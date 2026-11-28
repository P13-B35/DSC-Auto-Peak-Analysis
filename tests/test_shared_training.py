"""Shared curve identity, split validation, and two-scale sample tests."""
import csv
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from shared_dataset import build_dataset_plan, migrate_shared_dataset


def make_curve(folder, name, shift=0.0, broad=False):
    folder.mkdir(parents=True, exist_ok=True)
    t = np.linspace(20.0, 120.0, 2001)
    width = 12.0 if broad else 1.0
    signal = 0.01 * t - np.exp(-4 * np.log(2) * ((t - 75.0 - shift) / width) ** 2)
    signal += 0.1 * np.exp(-((t - 40.0 - shift) / 0.6) ** 2)
    pd.DataFrame({"temperature": t, "heatflow": signal}).to_excel(folder / name, index=False)
    halfwidth = 18.0 if broad else 2.0
    return [dict(filename=name, temp_center=75.0 + shift,
                 temp_start=75.0 + shift - halfwidth,
                 temp_end=75.0 + shift + halfwidth, type="melt_peak"),
            dict(filename=name, temp_center=30.0, temp_start=29.0,
                 temp_end=31.0, type="baseline"),
            dict(filename=name, temp_center=40.0 + shift,
                 temp_start=39.0 + shift, temp_end=41.0 + shift, type="artifact")]


def write_refs(folder, rows):
    with (folder / "references.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "filename", "temp_center", "temp_start", "temp_end", "type"), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


class SharedDatasetTests(unittest.TestCase):
    def setUp(self):
        work = ROOT / "workspace" / "test_tmp"
        work.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=str(work))
        self.addCleanup(self.temp.cleanup)
        self.data = Path(self.temp.name) / "data"
        self.train = self.data / "splits" / "shared" / "real_train"
        self.val = self.data / "splits" / "shared" / "real_val"
        self.train_refs = make_curve(self.train, "sharp.xlsx")
        self.train_refs += make_curve(self.train, "wide.xlsx", shift=2.0, broad=True)
        self.val_refs = make_curve(self.val, "held_out.xlsx", shift=-2.0)
        write_refs(self.train, self.train_refs)
        write_refs(self.val, self.val_refs)

    def test_all_shapes_share_one_plan(self):
        plan = build_dataset_plan(self.data)
        self.assertEqual([r["filename"] for r in plan["real_train"]],
                         ["sharp.xlsx", "wide.xlsx"])
        self.assertEqual(plan["scales"]["narrow"]["window_size"], 120)
        self.assertEqual(plan["scales"]["broad"]["target_span_c"], 4.0)
        self.assertEqual(plan["fingerprint"], build_dataset_plan(self.data)["fingerprint"])

    def test_renamed_curve_cannot_leak_to_validation(self):
        shutil.copy2(self.train / "sharp.xlsx", self.val / "renamed.xlsx")
        copied = [dict(ref, filename="renamed.xlsx") for ref in self.train_refs
                  if ref["filename"] == "sharp.xlsx"]
        write_refs(self.val, self.val_refs + copied)
        with self.assertRaisesRegex(ValueError, "both training and validation"):
            build_dataset_plan(self.data)

    def test_rejects_missing_or_out_of_range_annotations(self):
        write_refs(self.train, self.train_refs[:3])
        with self.assertRaisesRegex(ValueError, "no annotations"):
            build_dataset_plan(self.data)
        self.train_refs[0]["temp_end"] = 130.0
        write_refs(self.train, self.train_refs)
        with self.assertRaisesRegex(ValueError, "outside curve"):
            build_dataset_plan(self.data)

    def test_rejects_nonfinite_and_nonmonotonic_curves(self):
        frame = pd.read_excel(self.train / "sharp.xlsx")
        frame.iloc[3, 1] = np.nan
        frame.to_excel(self.train / "sharp.xlsx", index=False)
        with self.assertRaisesRegex(ValueError, "finite"):
            build_dataset_plan(self.data)
        frame.iloc[3, 1] = 0.0
        frame.iloc[3, 0] = frame.iloc[2, 0]
        frame.to_excel(self.train / "sharp.xlsx", index=False)
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            build_dataset_plan(self.data)

    def test_fs_parent_validation_and_missing_lineage(self):
        synth = self.data / "synthetic"
        write_refs(synth, make_curve(synth, "FS.xlsx", shift=4.0))
        plan = build_dataset_plan(self.data, "real_fs")
        self.assertEqual(plan["synthetic_lineage_status"], "unverified")
        pd.DataFrame([dict(filename="FS.xlsx", template_curve_id="held_out")]).to_csv(
            synth / "synthetic_metadata.csv", index=False)
        with self.assertRaisesRegex(ValueError, "source belongs to validation"):
            build_dataset_plan(self.data, "real_fs")
        pd.DataFrame([dict(filename="FS.xlsx", source_curve_ids=json.dumps(["sharp", "wide"]))]).to_csv(
            synth / "synthetic_metadata.csv", index=False)
        self.assertEqual(build_dataset_plan(self.data, "real_fs")["synthetic_lineage_status"], "verified")
        migrated = migrate_shared_dataset(self.data, Path(self.temp.name) / "migrated", "real_fs")
        self.assertEqual(migrated["synthetic_lineage_status"], "verified")

    def test_legacy_union_and_migration_keep_all_curves(self):
        legacy = Path(self.temp.name) / "legacy"
        for scale, name, refs in (("narrow", "sharp.xlsx", self.train_refs[:3]),
                                   ("broad", "wide.xlsx", self.train_refs[3:])):
            folder = legacy / "splits" / scale / "real_train"
            folder.mkdir(parents=True)
            shutil.copy2(self.train / name, folder / name)
            write_refs(folder, refs)
            val = legacy / "splits" / scale / "real_val"
            val.mkdir(parents=True)
            shutil.copy2(self.val / "held_out.xlsx", val / "held_out.xlsx")
            write_refs(val, self.val_refs)
        original = build_dataset_plan(legacy)
        migrated = migrate_shared_dataset(legacy, Path(self.temp.name) / "migrated")
        self.assertEqual(original["fingerprint"], migrated["fingerprint"])
        self.assertEqual(len(migrated["real_train"]), 2)
        self.assertEqual(len(migrated["real_val"]), 1)
        with self.assertRaisesRegex(ValueError, "must not already exist"):
            migrate_shared_dataset(legacy, Path(self.temp.name) / "migrated")

    def test_all_base_samples_survive_balancing(self):
        from train_family_models import balance_instances
        instances = [{"type": np.eye(3)[i % 3], "id": i} for i in range(30)]
        balanced = balance_instances(instances, base_target=2, seed=17, preserve_all=True)
        self.assertEqual({item["id"] for item in balanced}, set(range(30)))

    def test_hard_negatives_exclude_entire_broad_response(self):
        from train_family_models import mine_hard_negative_references
        from peakbot.dsc_peakbot.data_loader import DSCCurve, DSCDataset
        from peakbot.dsc_peakbot.config import Config
        temp = np.linspace(20.0, 120.0, 4001)
        curve = DSCCurve(name="wide.xlsx")
        curve.temperature = temp
        curve.heat_flow = -np.exp(-((temp - 77.0) / 12.0) ** 2) + 0.03 * np.sin(20 * temp)
        dataset = DSCDataset()
        dataset.curves = [curve]
        dataset.references = [dict(filename=curve.name, type="melt_peak",
                                   temp_center=77.0, temp_start=59.0, temp_end=95.0)]
        previous = Config.MIN_TEMPERATURE
        try:
            Config.MIN_TEMPERATURE = float("-inf")
            mined = mine_hard_negative_references(
                dataset, 240, per_curve=24, family="shared", target_span_c=4.0)
        finally:
            Config.MIN_TEMPERATURE = previous
        self.assertTrue(mined)
        self.assertTrue(all(not 57.0 <= ref["temp_center"] <= 97.0 for ref in mined))

    def test_scales_use_same_positions_and_protect_broad_melt(self):
        from train_family_models import load_shared_instances
        plan = build_dataset_plan(self.data)
        # A displaced annotation exposed the former scale-dependent snapping.
        for record in plan["real_train"]:
            for ref in record["references"]:
                if ref["type"] == "melt_peak":
                    ref["temp_center"] += 0.7
        _, small = load_shared_instances(plan["real_train"], 120, 0, 1, target_span_c=2.0)
        large_data, large = load_shared_instances(plan["real_train"], 240, 0, 1, target_span_c=4.0)
        self.assertEqual(len(small), len(large))
        np.testing.assert_allclose([x["temp_win"][60] for x in small],
                                   [x["temp_win"][120] for x in large], atol=1e-4)
        self.assertTrue(all(x["area"].shape == (120,) for x in small))
        self.assertTrue(all(x["area"].shape == (240,) for x in large))
        wide_refs = [r for r in large_data.references if r["filename"] == "wide.xlsx"]
        self.assertFalse(any(r["type"] == "artifact" and 59 <= r["temp_center"] <= 95
                             for r in wide_refs))
        # Evaluation retains the original manual center, not the snapped center.
        self.assertAlmostEqual(next(r["temp_center"] for r in wide_refs
                                    if r["type"] == "melt_peak"), 77.7)


if __name__ == "__main__":
    unittest.main()
