"""Joint inference with two observation scales and a shared candidate set.

The networks predict melting peak/baseline/artifact classes, not width classes.
Probabilities are averaged within each scale. Width is measured on the curve
only to choose a boundary rule after candidate screening.
"""

import csv
import math
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter

from .adaptive_boundary import (
    _odd_points, _upper_envelope_poly, find_adaptive_boundaries,
)
from .compensation import apply_compensation_events, exclusion_mask
from .config import Config, PAPER_WINDOWS
from .preprocessing import (
    _paper_model_window, find_local_maxima, temperature_window_size,
)


def _positive(value, name, allow_zero=False):
    value = float(value)
    if not math.isfinite(value) or (value < 0 if allow_zero else value <= 0):
        raise ValueError("{} must be finite and {}".format(
            name, "nonnegative" if allow_zero else "positive"))
    return value


def _curve_arrays(curve):
    temperature, signal = (np.asarray(a, dtype=float) for a in curve.to_numpy())
    if (temperature.ndim != 1 or signal.ndim != 1
            or len(temperature) != len(signal) or len(temperature) < 15
            or not np.all(np.isfinite(temperature))
            or not np.all(np.isfinite(signal))
            or not np.all(np.diff(temperature) > 0)):
        raise ValueError("Curve requires at least 15 finite, strictly increasing "
                         "temperature samples and matching heat flow")
    return temperature, signal


def prepare_shared_candidates(curve, min_temperature=None):
    """Return one raw-index candidate set and input arrays at both scales.

    Heat flow is in mW, oriented endothermic-down. Known compensation events
    are applied once; the original curve arrays are not modified.
    """
    temperature, signal = _curve_arrays(curve)
    events = getattr(curve, "compensation_events", None)
    exclusions = []
    if events:
        signal, diagnostics, exclusions = apply_compensation_events(
            temperature, signal, events)
        if any(d.get("status", "").startswith("skipped") for d in diagnostics):
            raise ValueError("A registered compensation event could not be applied")
    candidates = np.flatnonzero(find_local_maxima(signal, mode="both"))
    if min_temperature is not None:
        min_temperature = float(min_temperature)
        if not math.isfinite(min_temperature):
            raise ValueError("min_temperature must be finite")
        candidates = candidates[temperature[candidates] >= min_temperature]
    if exclusions:
        candidates = candidates[exclusion_mask(temperature[candidates], exclusions)]
    inputs = {}
    for scale, (length, span_C) in PAPER_WINDOWS.items():
        raw_size = temperature_window_size(temperature, span_C)
        areas = np.empty((len(candidates), length), dtype=np.float32)
        features = np.empty((len(candidates), 3), dtype=np.float32)
        for i, index in enumerate(candidates):
            _, _, areas[i], features[i] = _paper_model_window(
                temperature, signal, int(index), length, span_C, raw_size)
        inputs[scale] = (areas, features)
    return temperature, signal, candidates, inputs


def mean_class_probabilities(models, areas, features, batch_size=128):
    """Average raw three-class softmax outputs before selecting a class."""
    models = list(models)
    if not models:
        raise ValueError("Each observation scale requires at least one model")
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    n = len(areas)
    if areas.ndim != 2 or features.shape != (n, 3):
        raise ValueError("Expected areas[N,L] and features[N,3]")
    average = np.zeros((n, 3), dtype=float)
    for model in models:
        keras_model = getattr(model, "model", None)
        shapes = getattr(keras_model, "input_shape", None)
        model_length = (shapes[0][1] if isinstance(shapes, (list, tuple))
                        and shapes and isinstance(shapes[0], (list, tuple))
                        else getattr(model, "window_size", None))
        if model_length is not None and model_length != areas.shape[1]:
            raise ValueError("Model input length {} does not match {} points".format(
                model_length, areas.shape[1]))
        for start in range(0, n, batch_size):
            end = min(n, start + batch_size)
            prediction = model.predict(areas[start:end], features=features[start:end])
            probabilities = np.asarray(prediction[0], dtype=float)
            if (probabilities.shape != (end - start, 3)
                    or not np.all(np.isfinite(probabilities))
                    or np.any(probabilities < 0) or np.any(probabilities > 1)
                    or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-4)):
                raise ValueError("Model returned invalid three-class probabilities")
            average[start:end] += probabilities / len(models)
    return average


def _local_response(temperature, signal, center_C, span_C=35.0, smooth_C=2.0):
    mask = np.abs(temperature - center_C) <= span_C
    indices = np.flatnonzero(mask)
    if len(indices) < 30:
        return None
    x, raw = temperature[indices], signal[indices]
    window = _odd_points(x, smooth_C)
    smooth = savgol_filter(raw, window, 3 if window >= 7 else 2)
    background = _upper_envelope_poly(x, smooth, order=2)
    net = np.maximum(background - smooth, 0.0)
    return x, raw, smooth, background, net


def measure_temporal_width(temperature, signal, center_C, heating_rate_C_min):
    """Measure quadratic-background net FWHM; return explicit failure status."""
    heating_rate_C_min = _positive(heating_rate_C_min, "heating_rate_C_min")
    local = _local_response(temperature, signal, center_C)
    if local is None:
        return dict(width_C=float("nan"), width_min=float("nan"),
                    status="too_few_points")
    x, _, _, _, net = local
    nearby = np.flatnonzero(np.abs(x - center_C) <= 1.5)
    if len(nearby) < 3:
        status = "too_few_apex_samples"
    else:
        peak = int(nearby[np.argmax(net[nearby])])
        if peak in (nearby[0], nearby[-1]) or net[peak] <= 0:
            status = "no_local_apex"
        else:
            level = float(net[peak] / 2)
            left = right = peak
            while left > 0 and net[left] > level:
                left -= 1
            while right < len(x) - 1 and net[right] > level:
                right += 1
            if net[left] > level or net[right] > level:
                status = "missing_half_height_crossing"
            else:
                left_C = float(np.interp(level, net[left:left+2], x[left:left+2]))
                right_C = float(np.interp(level, net[right-1:right+1][::-1],
                                          x[right-1:right+1][::-1]))
                width_C = right_C - left_C
                if math.isfinite(width_C) and width_C > 0:
                    return dict(width_C=width_C,
                                width_min=width_C / heating_rate_C_min, status="ok")
                status = "invalid_half_height_width"
    return dict(width_C=float("nan"), width_min=float("nan"), status=status)


def _candidate_snr(temperature, signal, index):
    """Common full-curve depth/noise measure, independent of observation scale."""
    local = _local_response(temperature, signal, float(temperature[index]))
    if local is None:
        return 0.0, 0.0
    x, raw, _, _, net = local
    center = int(np.argmin(np.abs(x - temperature[index])))
    edge_n = max(6, len(raw) // 8)
    edge_x = np.r_[x[:edge_n], x[-edge_n:]]
    edge_y = np.r_[raw[:edge_n], raw[-edge_n:]]
    coef = np.polyfit(edge_x - np.mean(edge_x), edge_y, 1)
    residual = edge_y - np.polyval(coef, edge_x - np.mean(edge_x))
    noise = 1.4826 * float(np.median(np.abs(residual - np.median(residual))))
    # Estimate differences within the two shoulders, excluding the artificial
    # jump introduced by joining separated left and right edge arrays.
    differences = np.r_[np.diff(residual[:edge_n]), np.diff(residual[edge_n:])]
    if len(differences):
        noise = max(noise, float(np.std(differences) / np.sqrt(2.0)))
    depth = float(net[center])
    return depth, depth / max(noise, np.finfo(float).eps)


def _screen_scale(candidates, temperature, probabilities, snr, min_probability,
                  min_snr, nms_C, physical_keep=None):
    eligible = np.flatnonzero((probabilities.argmax(axis=1) == 0)
                              & (probabilities[:, 0] >= min_probability)
                              & (snr >= min_snr))
    if physical_keep is not None:
        eligible = eligible[physical_keep[eligible]]
    selected = []
    for row in sorted(eligible, key=lambda i: (-probabilities[i, 0], candidates[i])):
        if all(abs(temperature[candidates[row]] - temperature[candidates[j]])
               >= nms_C for j in selected):
            selected.append(int(row))
    return selected


def broad_two_sided_check(temperature, signal, center_C, inner_C=8.0, outer_C=18.0):
    """Require a net endothermic response recovering on both sides.

    The 8--18 C context follows the existing broad-screening context function
    in evaluate_broad_v3_retrospective.py. Here it is measured against the
    common quadratic background, rather than treating background slope as
    part of the peak. No fitted probability calibration is implied.
    """
    local = _local_response(temperature, signal, center_C)
    if local is None:
        return False, float("nan")
    x, raw, _, _, net = local
    left = net[(x >= center_C - outer_C) & (x <= center_C - inner_C)]
    right = net[(x >= center_C + inner_C) & (x <= center_C + outer_C)]
    nearby = np.flatnonzero(np.abs(x - center_C) <= 1.5)
    if min(len(left), len(right)) < 3 or len(nearby) < 3:
        return False, float("nan")
    apex = int(nearby[np.argmax(net[nearby])])
    numeric_floor = 128 * np.finfo(float).eps * max(1.0, float(np.max(np.abs(raw))))
    if apex in (nearby[0], nearby[-1]) or net[apex] <= numeric_floor:
        return False, 0.0
    center_depth = float(net[apex])
    recovery = min(center_depth - float(np.median(left)),
                   center_depth - float(np.median(right))) / center_depth
    return bool(math.isfinite(recovery) and recovery > 0), float(recovery)


def detect_curve_joint(curve, narrow_models, broad_models, heating_rate_C_min,
                       width_threshold_min, narrow_min_probability=0.5,
                       broad_min_probability=0.5, min_peak_snr=4.0,
                       narrow_nms_C=1.5, broad_nms_C=3.0,
                       min_temperature=None, batch_size=128,
                       broad_physics_check=True):
    """Return the screened union and a boundary-rule decision for every peak.

    A required time-width threshold is an explicit routing setting, not a
    network output. The result records failed width/boundary estimates with
    NaN endpoints and a non-ok boundary_status; these cannot be quantified.
    """
    heating_rate_C_min = _positive(heating_rate_C_min, "heating_rate_C_min")
    width_threshold_min = _positive(width_threshold_min, "width_threshold_min")
    min_peak_snr = _positive(min_peak_snr, "min_peak_snr", allow_zero=True)
    limits = {"narrow": narrow_min_probability, "broad": broad_min_probability}
    spacing = {"narrow": narrow_nms_C, "broad": broad_nms_C}
    for scale in PAPER_WINDOWS:
        limits[scale] = _positive(limits[scale], scale + "_min_probability",
                                  allow_zero=True)
        if limits[scale] > 1:
            raise ValueError("Class probability thresholds must be between 0 and 1")
        spacing[scale] = _positive(spacing[scale], scale + "_nms_C", allow_zero=True)
    temperature, signal, candidates, inputs = prepare_shared_candidates(
        curve, min_temperature=min_temperature)
    probabilities = {}
    for scale, models in (("narrow", narrow_models), ("broad", broad_models)):
        probabilities[scale] = mean_class_probabilities(models, *inputs[scale],
                                                       batch_size=batch_size)
    if not len(candidates):
        return []
    depth_snr = np.array([_candidate_snr(temperature, signal, i) for i in candidates])
    broad_physics = [broad_two_sided_check(temperature, signal, float(temperature[i]))
                     for i in candidates]
    physical_keep = np.array([value[0] for value in broad_physics], dtype=bool)
    accepted = {
        scale: _screen_scale(candidates, temperature, probabilities[scale],
                             depth_snr[:, 1], limits[scale], min_peak_snr,
                             spacing[scale], physical_keep=(physical_keep
                                 if scale == "broad" and broad_physics_check else None))
        for scale in PAPER_WINDOWS
    }
    # Nearby noisy extrema from both routes may represent one physical trough.
    # Group only those descending to the same minimum on the common smoothed
    # curve. Never rank a narrow probability against a broad probability.
    smooth = savgol_filter(signal, _odd_points(temperature, 2.0), 3)
    basins = {}
    def minimum_for(index):
        visited = []
        while index not in basins:
            visited.append(index)
            neighbors = [j for j in (index-1, index+1)
                         if 0 <= j < len(smooth) and smooth[j] < smooth[index]]
            if not neighbors:
                break
            index = min(neighbors, key=lambda j: (smooth[j], j))
        target = basins.get(index, index)
        for j in visited:
            basins[j] = target
        return target
    grouped = {}
    for scale, rows in accepted.items():
        for row in rows:
            minimum = minimum_for(int(candidates[row]))
            grouped.setdefault(minimum, []).append((scale, row))
    outputs = []
    for minimum, entries in sorted(grouped.items()):
        # A selected upward disturbance on a monotone region may descend to a
        # record edge, which is not a local trough and cannot define a peak.
        if minimum == 0 or minimum == len(smooth) - 1:
            continue
        rows = sorted({row for _, row in entries})
        representative = min(rows, key=lambda row: (
            abs(temperature[candidates[row]] - temperature[minimum]), candidates[row]))
        # Keep the common smoothed local minimum as the final location, even
        # when the only accepted raw candidate was a small maximum on a noisy
        # trough. Candidate identities/probabilities remain separate below.
        index = int(minimum)
        center = float(temperature[index])
        channels = sorted({scale for scale, _ in entries})
        width = dict(width_C=float("nan"), width_min=float("nan"),
                     status="not_needed_single_scale")
        boundary_rule = channels[0]
        status = "ok"
        start = end = float("nan")
        if len(channels) == 2:
            width = measure_temporal_width(temperature, signal, center,
                                            heating_rate_C_min)
            if width["status"] == "ok":
                boundary_rule = ("narrow" if width["width_min"] <= width_threshold_min
                                 else "broad")
            else:
                boundary_rule = "unresolved"
                status = "unresolved_width:" + width["status"]
        if boundary_rule != "unresolved":
            start, end = find_adaptive_boundaries(
                temperature, signal, center, family=boundary_rule)
            if not (math.isfinite(start) and math.isfinite(end)
                    and temperature[0] <= start < center < end <= temperature[-1]):
                start = end = float("nan")
                status = "invalid_boundary_crossings"
        scale_probabilities = {
            scale: (max(float(probabilities[scale][row, 0])
                        for entry_scale, row in entries if entry_scale == scale)
                    if scale in channels else float(probabilities[scale][representative, 0]))
            for scale in PAPER_WINDOWS
        }
        outputs.append(dict(
            filename=getattr(curve, "name", "unknown"),
            candidate_index=index, source_candidate_indices=[int(candidates[row]) for row in rows],
            center_temp=center, center_hf=float(signal[index]),
            temp_start=float(start), temp_end=float(end),
            type_idx=0, type_name="melt_peak",
            accepted_channels="both" if len(channels) == 2 else channels[0],
            boundary_rule=boundary_rule, boundary_status=status,
            width_status=width["status"], width_C=width["width_C"],
            width_min=width["width_min"], width_threshold_min=width_threshold_min,
            heating_rate_C_min=heating_rate_C_min,
            narrow_probability=scale_probabilities["narrow"],
            broad_probability=scale_probabilities["broad"],
            full_depth=float(depth_snr[representative, 0]),
            full_snr=float(depth_snr[representative, 1]),
            broad_two_sided_recovery=float(broad_physics[representative][1]),
            common_minimum_C=float(temperature[minimum]), merged_count=len(rows),
        ))
    return sorted(outputs, key=lambda row: row["center_temp"])


def load_model_ensemble(paths, scale):
    """Import TensorFlow only at the point actual saved weights are loaded."""
    paths = [Path(path).resolve() for path in paths]
    if not paths or len(set(paths)) != len(paths) or any(not p.is_file() for p in paths):
        raise ValueError("{} needs distinct, existing model files".format(scale))
    from .model import DSCPeakBot
    models = []
    for path in paths:
        model = DSCPeakBot()
        model.load(str(path))
        shapes = model.model.input_shape
        if shapes[0][1] != PAPER_WINDOWS[scale][0] or shapes[1][-1] != 3:
            raise ValueError("{} model has incompatible inputs: {}".format(scale, path))
        models.append(model)
    return models


def export_joint_results(peaks, path):
    """Keep the original quantify input columns and append routing diagnostics."""
    fields = ["PeakID", "Filename", "CenterTemp", "CenterHF", "TempStart",
              "TempEnd", "PeakType", "CandidateIndex", "SourceCandidateIndices",
              "AcceptedChannels", "BoundaryRule", "BoundaryStatus", "WidthStatus",
              "Width_C", "Width_min", "WidthThreshold_min", "HeatingRate_C_min",
              "NarrowProbability", "BroadProbability", "FullSNR", "MergedCount",
              "CommonMinimum_C", "BroadTwoSidedRecovery"]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for peak_id, row in enumerate(peaks):
            writer.writerow(dict(
                PeakID=peak_id, Filename=row["filename"], CenterTemp=row["center_temp"],
                CenterHF=row["center_hf"], TempStart=row["temp_start"], TempEnd=row["temp_end"],
                PeakType=row["type_name"], CandidateIndex=row["candidate_index"],
                SourceCandidateIndices=";".join(map(str, row["source_candidate_indices"])),
                AcceptedChannels=row["accepted_channels"], BoundaryRule=row["boundary_rule"],
                BoundaryStatus=row["boundary_status"], WidthStatus=row["width_status"],
                Width_C=row["width_C"], Width_min=row["width_min"],
                WidthThreshold_min=row["width_threshold_min"], HeatingRate_C_min=row["heating_rate_C_min"],
                NarrowProbability=row["narrow_probability"], BroadProbability=row["broad_probability"],
                FullSNR=row["full_snr"], MergedCount=row["merged_count"],
                CommonMinimum_C=row["common_minimum_C"],
                BroadTwoSidedRecovery=row["broad_two_sided_recovery"],
            ))
