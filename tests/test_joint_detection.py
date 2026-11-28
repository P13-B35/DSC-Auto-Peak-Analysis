"""Control-flow checks with artificial arrays and stub softmax predictors.

These tests validate the implementation, not measured DSC performance.
"""

import csv
import importlib.util
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from peakbot.dsc_peakbot.data_loader import DSCCurve
from peakbot.dsc_peakbot import joint_detection as joint


def curve(width=1.0, step=0.05):
    item = DSCCurve(name="artificial.xlsx")
    item.temperature = np.arange(0, 100 + step / 2, step)
    item.heat_flow = 2 + 0.01 * item.temperature - 10 * np.exp(
        -(item.temperature - 50) ** 2 / (2 * width ** 2))
    return item


class FakeModel:
    def __init__(self, length, probabilities):
        self.window_size = length
        self.probabilities = np.asarray(probabilities, dtype=float)
        self.seen = []

    def predict(self, areas, features):
        self.seen.append((areas.copy(), features.copy()))
        if self.probabilities.ndim == 1:
            p = np.tile(self.probabilities, (len(areas), 1))
        else:
            start = sum(len(a) for a, _ in self.seen[:-1])
            p = self.probabilities[start:start + len(areas)]
        return p, np.zeros((len(areas), 1)), np.zeros((len(areas), 2))


def models(narrow=True, broad=True):
    yes, no = [0.98, 0.01, 0.01], [0.01, 0.98, 0.01]
    return ([FakeModel(120, yes if narrow else no)],
            [FakeModel(240, yes if broad else no)])


class JointDetectionTests(unittest.TestCase):
    def test_import_does_not_require_tensorflow(self):
        code = ("import sys; sys.path.insert(0, sys.argv[1]); "
                "import peakbot.dsc_peakbot.joint_detection; "
                "assert 'tensorflow' not in sys.modules")
        result = subprocess.run([sys.executable, "-B", "-c", code, str(ROOT / "src")],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_identical_candidates_at_both_scales_and_dynamic_sampling(self):
        for step in (0.025, 0.1):
            item = curve(step=step)
            temperature, _, indices, inputs = joint.prepare_shared_candidates(item)
            self.assertGreater(len(indices), 0)
            self.assertTrue(np.any(temperature[indices] < 80))
            self.assertEqual(inputs["narrow"][0].shape, (len(indices), 120))
            self.assertEqual(inputs["broad"][0].shape, (len(indices), 240))
            self.assertEqual(inputs["narrow"][1].shape, (len(indices), 3))
            self.assertEqual(inputs["broad"][1].shape, (len(indices), 3))
            self.assertTrue(np.all(np.isfinite(inputs["narrow"][0])))

    def test_softmax_averaged_before_screening_and_batched(self):
        area, feature = np.zeros((7, 120)), np.zeros((7, 3))
        first = FakeModel(120, [0.8, 0.1, 0.1])
        second = FakeModel(120, [0.2, 0.7, 0.1])
        result = joint.mean_class_probabilities([first, second], area, feature, batch_size=3)
        np.testing.assert_allclose(result, np.tile([0.5, 0.4, 0.1], (7, 1)))
        self.assertEqual([len(a) for a, _ in first.seen], [3, 3, 1])
        with self.assertRaisesRegex(ValueError, "input length"):
            joint.mean_class_probabilities([FakeModel(240, [1, 0, 0])], area, feature)
        with self.assertRaisesRegex(ValueError, "invalid three-class"):
            joint.mean_class_probabilities([FakeModel(120, [0.5, 0.5, 0.5])], area, feature)

    def test_one_route_uses_its_own_boundary_rule(self):
        for narrow, broad, expected in ((True, False, "narrow"), (False, True, "broad")):
            n, b = models(narrow, broad)
            result = joint.detect_curve_joint(curve(), n, b, 10, 0.5)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["accepted_channels"], expected)
            self.assertEqual(result[0]["boundary_rule"], expected)
            self.assertEqual(result[0]["boundary_status"], "ok")
            self.assertEqual(result[0]["width_status"], "not_needed_single_scale")

    def test_both_routes_merge_and_width_selects_rule(self):
        for width, threshold, expected in ((1.0, 0.5, "narrow"), (4.0, 0.5, "broad")):
            n, b = models()
            result = joint.detect_curve_joint(curve(width=width), n, b, 10, threshold)
            self.assertEqual(len(result), 1)
            row = result[0]
            self.assertEqual(row["accepted_channels"], "both")
            self.assertEqual(row["boundary_rule"], expected)
            self.assertEqual(row["width_status"], "ok")
            self.assertEqual(row["boundary_status"], "ok")
            self.assertAlmostEqual(row["width_min"], row["width_C"] / 10)

    def test_temporal_width_unit_and_missing_crossings(self):
        item = curve(width=2.0)
        result = joint.measure_temporal_width(*item.to_numpy(), 50, 10)
        self.assertEqual(result["status"], "ok")
        self.assertAlmostEqual(result["width_C"], 4.70964, delta=0.15)
        self.assertAlmostEqual(result["width_min"], result["width_C"] / 10)
        with patch.object(joint, "measure_temporal_width", return_value=dict(
                width_C=float("nan"), width_min=float("nan"),
                status="missing_half_height_crossing")):
            n, b = models()
            row = joint.detect_curve_joint(item, n, b, 10, 0.5)[0]
        self.assertEqual(row["boundary_rule"], "unresolved")
        self.assertEqual(row["boundary_status"],
                         "unresolved_width:missing_half_height_crossing")
        self.assertTrue(math.isnan(row["temp_start"]))
        self.assertTrue(math.isnan(row["temp_end"]))

    def test_filters_not_bypassed_by_high_probability(self):
        n, b = models()
        result = joint.detect_curve_joint(curve(), n, b, 10, 0.5, min_peak_snr=1e30)
        self.assertEqual(result, [])
        n = [FakeModel(120, [0.45, 0.3, 0.25])]
        b = [FakeModel(240, [0.45, 0.3, 0.25])]
        self.assertEqual(joint.detect_curve_joint(curve(), n, b, 10, 0.5), [])

    def test_broad_context_rejects_pure_slope_and_accepts_peak(self):
        t = np.linspace(0, 100, 2001)
        self.assertFalse(joint.broad_two_sided_check(t, 1 + 0.1 * t, 50)[0])
        item = curve(width=4)
        self.assertTrue(joint.broad_two_sided_check(*item.to_numpy(), 50)[0])
        self.assertFalse(joint.broad_two_sided_check(*item.to_numpy(), 4)[0])

    def test_accepted_noisy_raw_maximum_localizes_to_common_minimum(self):
        item = curve(width=3)
        item.heat_flow += 0.15 * np.cos((item.temperature - 50) * 12)
        _, _, candidates, _ = joint.prepare_shared_candidates(item)
        maxima = [i for i in candidates if abs(item.temperature[i] - 50) < 1
                  and item.heat_flow[i] > item.heat_flow[i-1]
                  and item.heat_flow[i] > item.heat_flow[i+1]]
        self.assertTrue(maxima)
        accepted = maxima[0]
        probs = np.tile([0.01, 0.98, 0.01], (len(candidates), 1))
        probs[np.flatnonzero(candidates == accepted)[0]] = [0.98, 0.01, 0.01]
        n, b = [FakeModel(120, probs)], [FakeModel(240, probs)]
        result = joint.detect_curve_joint(item, n, b, 10, 0.5, min_peak_snr=0)
        self.assertEqual(len(result), 1)
        row = result[0]
        self.assertIn(accepted, row["source_candidate_indices"])
        self.assertNotEqual(row["candidate_index"], accepted)
        self.assertAlmostEqual(row["center_temp"], row["common_minimum_C"])

    def test_nms_and_no_candidates(self):
        t = np.array([0., 1., 1.4, 6.])
        probabilities = np.array([[.8, .1, .1], [.9, .05, .05], [.7, .2, .1]])
        selected = joint._screen_scale(np.array([1, 2, 3]), t, probabilities,
                                       np.ones(3)*10, .5, 4, 1.5)
        self.assertEqual(selected, [1, 2])
        item = curve()
        item.heat_flow = np.arange(len(item.temperature), dtype=float)
        n, b = models()
        self.assertEqual(joint.detect_curve_joint(item, n, b, 10, .5), [])

    def test_invalid_configuration_fails(self):
        for rate, gate in ((0, .5), (10, 0), (float("nan"), .5)):
            n, b = models()
            with self.assertRaises(ValueError):
                joint.detect_curve_joint(curve(), n, b, rate, gate)
        n, b = models()
        with self.assertRaises(ValueError):
            joint.detect_curve_joint(curve(), n, b, 10, .5, broad_min_probability=1.1)
        with self.assertRaises(ValueError):
            joint.detect_curve_joint(curve(), [], b, 10, .5)
        item = curve()
        item.temperature[3] = item.temperature[2]
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            joint.detect_curve_joint(item, n, b, 10, .5)

    def test_export_and_quantify_reject_unresolved_boundary(self):
        spec = importlib.util.spec_from_file_location("joint_test_cli", ROOT / "dsc_peakbot_cli.py")
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as folder:
            output = Path(folder) / "joint.tsv"
            n, b = models()
            rows = joint.detect_curve_joint(curve(), n, b, 10, .5)
            rows[0]["temp_start"] = rows[0]["temp_end"] = float("nan")
            rows[0]["boundary_status"] = "unresolved_width:test"
            joint.export_joint_results(rows, output)
            with output.open(newline="", encoding="utf-8") as handle:
                exported = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual(exported[0]["BoundaryStatus"], "unresolved_width:test")
            args = type("Args", (), dict(detections=output, curves=Path(folder),
                metadata=None, mass_mg=10., heating_rate=10., output=Path(folder)/"thermal.csv"))()
            fake_dataset = type("Dataset", (), {"curves": [curve()], "load_folder": lambda s, p: s})
            with patch("peakbot.dsc_peakbot.data_loader.DSCDataset", fake_dataset):
                with self.assertRaisesRegex(ValueError, "Unresolved joint boundary"):
                    cli.quantify(args)
            self.assertFalse(args.output.exists())


if __name__ == "__main__":
    unittest.main()
