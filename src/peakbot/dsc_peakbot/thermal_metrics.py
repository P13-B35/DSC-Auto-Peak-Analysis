"""Numerically robust thermal metrics for DSC peak regions."""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter


def _odd_window(temperature, width_C, minimum=9, maximum=301):
    dt = float(np.median(np.diff(temperature)))
    n = int(round(width_C / max(dt, 1e-9)))
    n = max(minimum, min(maximum, n))
    if n % 2 == 0:
        n += 1
    if n >= len(temperature):
        n = len(temperature) - 1 if len(temperature) % 2 == 0 else len(temperature)
    return max(5, n)


def _interpolated_roots(x, y):
    roots = []
    for i in range(1, len(y)):
        y0, y1 = float(y[i - 1]), float(y[i])
        if y0 == 0.0:
            roots.append(float(x[i - 1]))
        elif y0 * y1 < 0.0:
            roots.append(float(x[i - 1] - y0 * (x[i] - x[i - 1]) / (y1 - y0)))
    return roots


def calculate_extrapolated_onset_endset(temperature, signal, boundaries,
                                         baseline):
    """Calculate tangent-baseline onset/endset using derivatives in signal/degC.

    Unlike the legacy implementation, ``delta`` is supplied to Savitzky-Golay
    differentiation, so tangent slopes and temperature are dimensionally
    consistent.  The onset is the closest valid intersection before the
    leading-edge inflection; endset is defined analogously after the trailing
    inflection.  Fallbacks are marked explicitly rather than silently treated
    as successful extrapolated temperatures.
    """
    temperature = np.asarray(temperature, dtype=float)
    signal = np.asarray(signal, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    outputs = []

    for t_start, t_end in boundaries:
        indices = np.where((temperature >= t_start) & (temperature <= t_end))[0]
        if len(indices) < 15:
            outputs.append({
                "t_onset": float(t_start), "t_endset": float(t_end),
                "onset_status": "fallback_short_region",
                "endset_status": "fallback_short_region",
            })
            continue

        x = temperature[indices]
        y = signal[indices]
        b = baseline[indices]
        dt = float(np.median(np.diff(x)))
        peak_width = float(t_end - t_start)
        smooth_width = min(1.2, max(0.25, 0.03 * peak_width))
        window = _odd_window(x, smooth_width)
        polyorder = min(3, window - 2)
        smooth = savgol_filter(y, window, polyorder)
        derivative = savgol_filter(y, window, polyorder, deriv=1, delta=dt)
        center = int(np.argmin(smooth))

        margin = max(3, int(round(0.03 * len(indices))))
        left_candidates = np.arange(margin, max(margin + 1, center - margin + 1))
        right_candidates = np.arange(min(len(indices) - margin - 1, center + margin),
                                     len(indices) - margin)

        if len(left_candidates):
            left_inf = int(left_candidates[np.argmin(derivative[left_candidates])])
            left_slope = float(derivative[left_inf])
            tangent = left_slope * (x[:left_inf + 1] - x[left_inf]) + smooth[left_inf]
            roots = _interpolated_roots(x[:left_inf + 1], tangent - b[:left_inf + 1])
            valid = [root for root in roots if t_start <= root <= x[left_inf]]
        else:
            left_inf, left_slope, valid = 0, np.nan, []
        if valid and np.isfinite(left_slope) and left_slope < 0.0:
            onset = max(valid)
            onset_status = "tangent_intersection"
        else:
            onset = float(t_start)
            onset_status = "fallback_boundary"

        if len(right_candidates):
            right_inf = int(right_candidates[np.argmax(derivative[right_candidates])])
            right_slope = float(derivative[right_inf])
            tangent = (right_slope * (x[right_inf:] - x[right_inf])
                       + smooth[right_inf])
            roots = _interpolated_roots(x[right_inf:], tangent - b[right_inf:])
            valid = [root for root in roots if x[right_inf] <= root <= t_end]
        else:
            right_inf, right_slope, valid = len(x) - 1, np.nan, []
        if valid and np.isfinite(right_slope) and right_slope > 0.0:
            endset = min(valid)
            endset_status = "tangent_intersection"
        else:
            endset = float(t_end)
            endset_status = "fallback_boundary"

        outputs.append({
            "t_onset": round(float(onset), 4),
            "t_endset": round(float(endset), 4),
            "onset_slope_mW_per_C": round(float(left_slope), 6)
                if np.isfinite(left_slope) else None,
            "endset_slope_mW_per_C": round(float(right_slope), 6)
                if np.isfinite(right_slope) else None,
            "onset_status": onset_status,
            "endset_status": endset_status,
            "smooth_width_C": round(float(smooth_width), 4),
        })
    return outputs
