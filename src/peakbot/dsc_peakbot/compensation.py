"""Deterministic correction for known upper-computer compensation events.

The acquisition software can enable a compensation table at a known
temperature.  The switch creates a non-thermal transient and a baseline
offset.  These events are metadata, not material transitions, and should be
handled before peak candidate generation rather than learned by the CNN.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def _as_float(value, default=None):
    if value is None or value == "":
        return default
    return float(value)


def load_compensation_registry(path):
    """Load a CSV registry and return ``{filename: [event, ...]}``."""
    registry = {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            event = dict(row)
            for key in (
                    "event_temp_C", "transition_start_C", "transition_end_C",
                    "left_fit_start_C", "left_fit_end_C", "right_fit_start_C",
                    "right_fit_end_C", "candidate_guard_C"):
                event[key] = _as_float(event.get(key))
            registry.setdefault(row["filename"], []).append(event)
    return registry


def attach_compensation_events(curves, registry):
    """Attach registered events to matching ``DSCCurve`` objects."""
    attached = 0
    for curve in curves:
        curve.compensation_events = list(registry.get(curve.name, []))
        attached += len(curve.compensation_events)
    return attached


def _robust_linear_fit(x, y, max_iter=5, sigma=3.5):
    """Iteratively sigma-clipped first-order fit."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    keep = np.isfinite(x) & np.isfinite(y)
    if np.sum(keep) < 3:
        raise ValueError("At least three finite samples are required for fitting")
    for _ in range(max_iter):
        coef = np.polyfit(x[keep], y[keep], 1)
        residual = y - np.polyval(coef, x)
        med = np.median(residual[keep])
        mad = 1.4826 * np.median(np.abs(residual[keep] - med))
        if not np.isfinite(mad) or mad <= np.finfo(float).eps:
            break
        new_keep = keep & (np.abs(residual - med) <= sigma * mad)
        if np.array_equal(new_keep, keep) or np.sum(new_keep) < 3:
            break
        keep = new_keep
    return coef, keep


def _event_defaults(event):
    t0 = float(event["event_temp_C"])
    return {
        **event,
        "event_temp_C": t0,
        "transition_start_C": _as_float(
            event.get("transition_start_C"), t0 - 1.5),
        "transition_end_C": _as_float(
            event.get("transition_end_C"), t0 + 8.0),
        "left_fit_start_C": _as_float(
            event.get("left_fit_start_C"), t0 - 12.0),
        "left_fit_end_C": _as_float(
            event.get("left_fit_end_C"), t0 - 2.0),
        "right_fit_start_C": _as_float(
            event.get("right_fit_start_C"), t0 + 9.0),
        "right_fit_end_C": _as_float(
            event.get("right_fit_end_C"), t0 + 20.0),
        "candidate_guard_C": _as_float(
            event.get("candidate_guard_C"), 0.5),
    }


def apply_compensation_events(temperature, signal, events):
    """Stitch known compensation offsets and suppress their transition zones.

    For each event, robust linear trends are fitted on stable data before and
    after the transition.  The post-event branch is shifted to meet the
    extrapolated pre-event trend at the end of the transition.  Samples inside
    the non-thermal transition are replaced by a straight bridge.  Candidate
    generation should exclude the returned intervals.

    Returns:
        corrected, diagnostics, exclusion_intervals
    """
    temperature = np.asarray(temperature, dtype=np.float64)
    corrected = np.asarray(signal, dtype=np.float64).copy()
    diagnostics = []
    exclusions = []

    for raw_event in sorted(events or [], key=lambda item: float(item["event_temp_C"])):
        event = _event_defaults(raw_event)
        t0 = event["event_temp_C"]
        start = event["transition_start_C"]
        end = event["transition_end_C"]
        if not start < end:
            raise ValueError("transition_start_C must be smaller than transition_end_C")

        left_mask = ((temperature >= event["left_fit_start_C"])
                     & (temperature <= event["left_fit_end_C"]))
        right_mask = ((temperature >= event["right_fit_start_C"])
                      & (temperature <= event["right_fit_end_C"]))
        transition_mask = (temperature >= start) & (temperature <= end)
        post_mask = temperature > end
        if np.sum(left_mask) < 3 or np.sum(right_mask) < 3 or not np.any(transition_mask):
            diagnostics.append({
                "event_temp_C": t0,
                "status": "skipped_insufficient_data",
                "transition_start_C": start,
                "transition_end_C": end,
            })
            continue

        left_coef, left_keep = _robust_linear_fit(
            temperature[left_mask], corrected[left_mask])
        right_coef, right_keep = _robust_linear_fit(
            temperature[right_mask], corrected[right_mask])

        # Align the stable post-event branch with the extrapolated pre-event
        # baseline at the transition end.  A constant shift preserves peak
        # centres, widths, areas relative to a local baseline, and derivatives.
        left_at_end = float(np.polyval(left_coef, end))
        right_at_end = float(np.polyval(right_coef, end))
        offset = right_at_end - left_at_end
        corrected[post_mask] -= offset

        idx = np.where(transition_mask)[0]
        y0 = float(np.polyval(left_coef, start))
        y1 = float(np.polyval(right_coef, end) - offset)
        alpha = ((temperature[idx] - start) / (end - start)).clip(0.0, 1.0)
        # Smoothstep bridge gives zero slope contribution at both endpoints and
        # avoids manufacturing a local extremum in the compensation interval.
        blend = alpha * alpha * (3.0 - 2.0 * alpha)
        corrected[idx] = y0 + (y1 - y0) * blend

        guard = event["candidate_guard_C"]
        exclusions.append((start - guard, end + guard))
        diagnostics.append({
            "event_temp_C": t0,
            "status": "corrected",
            "transition_start_C": start,
            "transition_end_C": end,
            "estimated_post_minus_pre_offset": float(offset),
            "applied_post_shift": float(-offset),
            "left_fit_points": int(np.sum(left_keep)),
            "right_fit_points": int(np.sum(right_keep)),
            "candidate_exclusion_start_C": float(start - guard),
            "candidate_exclusion_end_C": float(end + guard),
        })

    return corrected.astype(np.float32), diagnostics, exclusions


def exclusion_mask(temperature, intervals):
    """Return True for temperatures outside all excluded intervals."""
    temperature = np.asarray(temperature)
    keep = np.ones(temperature.shape, dtype=bool)
    for start, end in intervals or []:
        keep &= ~((temperature >= start) & (temperature <= end))
    return keep
