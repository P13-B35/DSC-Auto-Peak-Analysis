"""Adaptive full-curve DSC peak boundaries based on baseline return."""

import numpy as np
from scipy.signal import savgol_filter


BOUNDARY_POLICIES = {
    "narrow": {
        # Expert-annotation policy selected with leave-one-whole-curve-out CV.
        "span_C": 35.0, "smooth_C": 2.0, "fraction": 0.005,
        "n_sigma": 1.0, "poly_order": 2,
    },
    "broad": {
        "span_C": 35.0, "smooth_C": 2.0, "fraction": 0.05,
        "n_sigma": 2.0, "poly_order": 1,
    },
}


def _odd_points(temperature, width_C, minimum=7, maximum=501):
    dt = float(np.median(np.diff(temperature)))
    points = int(round(width_C / max(dt, 1e-6)))
    points = min(maximum, max(minimum, points))
    if points % 2 == 0:
        points += 1
    if points >= len(temperature):
        points = len(temperature) - 1 if len(temperature) % 2 == 0 \
            else len(temperature)
    return max(5, points)


def _upper_envelope_poly(x, y, order=2, asymmetry=0.03, iterations=12):
    x0 = x - np.mean(x)
    scale = max(np.std(x0), 1e-9)
    z = x0 / scale
    weights = np.ones(len(x), dtype=float)
    coef = np.polyfit(z, y, order)
    for _ in range(iterations):
        coef = np.polyfit(z, y, order, w=np.sqrt(weights))
        residual = y - np.polyval(coef, z)
        new_weights = np.where(residual >= 0.0, 1.0, asymmetry)
        if np.array_equal(new_weights, weights):
            break
        weights = new_weights
    return np.polyval(coef, z)


def find_adaptive_boundaries(temperature, signal, center_temperature,
                             family="narrow", **overrides):
    """Return temperature boundaries where a peak returns to local baseline.

    The signal is smoothed using a temperature-defined width.  An asymmetric
    IRLS polynomial estimates the upper envelope of the oriented endothermic
    trough.  Boundaries are the first stable crossings of a combined relative
    depth and noise threshold on either side.
    """
    params = dict(BOUNDARY_POLICIES[family])
    params.update(overrides)
    span_C = params["span_C"]
    smooth_C = params["smooth_C"]
    fraction = params["fraction"]
    n_sigma = params["n_sigma"]
    poly_order = params["poly_order"]

    temperature = np.asarray(temperature, dtype=float)
    signal = np.asarray(signal, dtype=float)
    mask = ((temperature >= center_temperature - span_C)
            & (temperature <= center_temperature + span_C))
    indices = np.where(mask)[0]
    if len(indices) < 30:
        return float(center_temperature), float(center_temperature)
    x = temperature[indices]
    y = signal[indices]
    window = _odd_points(x, smooth_C)
    smooth = savgol_filter(y, window, 3 if window >= 7 else 2)
    center_idx = int(np.argmin(np.abs(x - center_temperature)))
    local = smooth[max(0, center_idx - 10):min(len(smooth), center_idx + 11)]
    if smooth[center_idx] > np.median(local):
        smooth = -smooth

    baseline = _upper_envelope_poly(x, smooth, order=poly_order)
    residual = np.maximum(baseline - smooth, 0.0)
    edge_n = max(10, len(smooth) // 8)
    edge = np.r_[smooth[:edge_n], smooth[-edge_n:]]
    noise = np.std(np.diff(edge)) / np.sqrt(2.0) if len(edge) > 3 else 0.0
    depth = float(residual[center_idx])
    threshold = max(float(fraction * depth), float(n_sigma * noise))

    stable_n = max(3, _odd_points(x, 0.35, minimum=3, maximum=101))
    stable = np.convolve(residual, np.ones(stable_n) / stable_n, mode="same")
    left = center_idx
    while left > 1 and stable[left] > threshold:
        left -= 1
    right = center_idx
    while right < len(x) - 2 and stable[right] > threshold:
        right += 1
    return float(x[left]), float(x[right])
