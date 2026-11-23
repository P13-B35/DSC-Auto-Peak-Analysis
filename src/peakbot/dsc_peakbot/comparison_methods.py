"""Transparent DSC baselines used only for method comparison."""

from __future__ import annotations

import numpy as np
from scipy.integrate import cumulative_trapezoid


def _select_roi(temperature, start_C, end_C):
    width = float(end_C - start_C)
    margin = max(10.0, 0.5 * width)
    mask = ((temperature >= start_C - margin)
            & (temperature <= end_C + margin))
    return np.where(mask)[0]


def _normalized_heat_capacity(heatflow_mW, mass_mg, heating_rate_C_min):
    beta = float(heating_rate_C_min) / 60.0
    return np.asarray(heatflow_mW, dtype=float) / (beta * float(mass_mg))


def pydsc_sigmoid_enthalpy(temperature, heatflow_mW, start_C, end_C,
                           mass_mg, heating_rate_C_min, iterations=100):
    """Reproduce the published PyDSC sigmoid-baseline calculation.

    ``start_C`` and ``end_C`` correspond to the user-supplied PyDSC ROP_h.
    A deterministic ROI margin is used so that the comparison cannot be tuned
    curve by curve.  Returned enthalpy is positive for a downward endotherm.
    """
    temperature = np.asarray(temperature, dtype=float)
    cp = _normalized_heat_capacity(heatflow_mW, mass_mg, heating_rate_C_min)
    roi = _select_roi(temperature, start_C, end_C)
    t, y = temperature[roi], cp[roi]
    pre = t < start_C
    post = t > end_C
    if pre.sum() < 5 or post.sum() < 5:
        raise ValueError("insufficient pre/post points for PyDSC baseline")
    pre_s, pre_i = np.polyfit(t[pre], y[pre], 1)
    post_s, post_i = np.polyfit(t[post], y[post], 1)

    outside = pre | post
    initial = np.interp(t, t[outside], y[outside])
    cumulative = cumulative_trapezoid(y - initial, t, initial=0.0)
    baseline = initial
    for _ in range(iterations):
        total = float(cumulative[-1])
        if abs(total) < 1e-12:
            break
        alpha = cumulative / total
        baseline = (pre_i + pre_s * t
                    - alpha * ((pre_i - post_i) + (pre_s - post_s) * t))
        cumulative = cumulative_trapezoid(y - baseline, t, initial=0.0)
    return {
        "deltaH_J_g": abs(float(cumulative[-1])),
        "signed_deltaH_J_g": float(cumulative[-1]),
        "roi_start_C": float(t[0]),
        "roi_end_C": float(t[-1]),
        "rop_start_C": float(start_C),
        "rop_end_C": float(end_C),
    }


def manual_linear_enthalpy(temperature, heatflow_mW, start_C, end_C,
                           mass_mg, heating_rate_C_min,
                           anchor_half_width_C=0.25):
    """Integrate against a straight line through robust endpoint anchors."""
    temperature = np.asarray(temperature, dtype=float)
    y = _normalized_heat_capacity(heatflow_mW, mass_mg, heating_rate_C_min)
    left = np.abs(temperature - start_C) <= anchor_half_width_C
    right = np.abs(temperature - end_C) <= anchor_half_width_C
    peak = (temperature >= start_C) & (temperature <= end_C)
    if left.sum() < 3 or right.sum() < 3 or peak.sum() < 5:
        raise ValueError("insufficient points for manual linear baseline")
    y_left, y_right = float(np.median(y[left])), float(np.median(y[right]))
    t_peak, y_peak = temperature[peak], y[peak]
    baseline = y_left + (y_right - y_left) * (
        (t_peak - start_C) / max(1e-12, end_C - start_C))
    signed = float(np.trapz(y_peak - baseline, t_peak))
    return {
        "deltaH_J_g": abs(signed),
        "signed_deltaH_J_g": signed,
        "rop_start_C": float(start_C),
        "rop_end_C": float(end_C),
    }
