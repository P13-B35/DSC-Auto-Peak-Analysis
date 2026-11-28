"""Preprocessing for DSC curves — local maxima, window extraction, instance generation."""

import os
import pickle
import math

import numpy as np
from scipy.signal import savgol_filter

from .config import Config
from .compensation import apply_compensation_events, exclusion_mask
from .core import tic, toc, timeit


# ---------------------------------------------------------------------------
# 1D local maxima detection
# ---------------------------------------------------------------------------

def _label_runs(bool_mask):
    """Label contiguous True runs in a boolean array.

    Returns:
        labels: int array of same length, -1 where mask is False,
                0..n_runs-1 for each run.
        n_runs: number of runs.
    """
    n = len(bool_mask)
    labels = np.full(n, -1, dtype=int)
    run_id = 0
    in_run = False
    for i in range(n):
        if bool_mask[i]:
            if not in_run:
                in_run = True
                run_id += 1
            labels[i] = run_id - 1
        else:
            in_run = False
    return labels, run_id


def find_local_maxima(heat_flow, min_intensity=None, mode='max'):
    """Find local maxima and/or minima in a 1D heat flow signal.

    DSC data contains both endothermic (downward, local minima) and exothermic
    (upward, local maxima) features.  This function supports three modes:

        'max'  — only local maxima (original behaviour)
        'min'  — only local minima (endothermic melt peaks)
        'both' — both maxima and minima

    Args:
        heat_flow: 1D numpy array of heat flow values.
        min_intensity: Optional minimum intensity threshold.
        mode: 'max', 'min', or 'both'.

    Returns:
        Boolean mask of same length as heat_flow.
    """
    n = len(heat_flow)
    mask = np.zeros(n, dtype=bool)
    if n < 3:
        return mask

    if min_intensity is None:
        min_intensity = -np.inf

    for i in range(1, n - 1):
        val = heat_flow[i]
        left = heat_flow[i - 1]
        right = heat_flow[i + 1]

        is_max = (mode in ('max', 'both')) and val >= min_intensity and val >= left and val >= right
        is_min = (mode in ('min', 'both')) and val <= -min_intensity and val <= left and val <= right

        # Avoid flagging flat regions: require at least one strict inequality
        if is_max and val == left and val == right:
            is_max = False
        if is_min and val == left and val == right:
            is_min = False

        if is_max or is_min:
            mask[i] = True

    # Merge consecutive detections into the centre of each flat run
    labels, n_runs = _label_runs(mask)
    final_mask = np.zeros(n, dtype=bool)
    for run_id in range(n_runs):
        indices = np.where(labels == run_id)[0]
        centre = indices[len(indices) // 2]
        final_mask[centre] = True

    return final_mask


# ---------------------------------------------------------------------------
# Window extraction
# ---------------------------------------------------------------------------

def extract_window(temperature, heat_flow, center_idx, window_size=None):
    """Extract a standardized 1D window around a centre index.

    Pads with edge values when the window extends beyond the array bounds.

    Args:
        temperature: Full temperature array.
        heat_flow: Full heat flow array.
        center_idx: Index of the window centre.
        window_size: Size of the output window (default Config.WINDOW_SIZE).

    Returns:
        (temp_window, hf_window) — both 1D float32 arrays of length window_size.
    """
    if window_size is None:
        window_size = Config.WINDOW_SIZE

    half = window_size // 2
    n = len(heat_flow)

    temp_window = np.zeros(window_size, dtype=np.float32)
    hf_window   = np.zeros(window_size, dtype=np.float32)

    src_start = max(0, center_idx - half)
    src_end   = min(n, center_idx + half + (window_size % 2))
    dst_start = max(0, half - center_idx)
    dst_end   = dst_start + (src_end - src_start)

    temp_window[dst_start:dst_end] = temperature[src_start:src_end]
    hf_window[dst_start:dst_end]   = heat_flow[src_start:src_end]

    # Fill padded regions with the nearest edge value
    if dst_start > 0:
        hf_window[:dst_start]   = heat_flow[0]
        temp_window[:dst_start] = temperature[0]
    if dst_end < window_size:
        hf_window[dst_end:]   = heat_flow[-1]
        temp_window[dst_end:] = temperature[-1]

    return temp_window, hf_window


def temperature_window_size(temperature, target_span_c):
    """Estimate the raw point count from the measured temperature increment."""
    temperature = np.asarray(temperature, dtype=np.float64)
    steps = np.diff(temperature)
    increasing_steps = steps[np.isfinite(steps) & (steps > 0)]
    if not increasing_steps.size:
        raise ValueError("A temperature window requires increasing temperature samples")
    return max(3, min(len(temperature),
                      int(round(target_span_c / np.median(increasing_steps)))))


def _paper_model_window(temperature, heat_flow, center_idx, window_size,
                        target_span_c, raw_size):
    """Use temperatures and unnormalized heat flow in mW for the paper inputs."""
    # Include one extra sample on each side for interpolation at the edges.
    start = center_idx - raw_size // 2 - 1
    stop = min(len(temperature), start + raw_size + 2)
    start = max(0, start)
    raw_t = np.asarray(temperature[start:stop], dtype=np.float64)
    raw_q = np.asarray(heat_flow[start:stop], dtype=np.float64)
    apex_t = float(temperature[center_idx])
    edge_count = max(1, min(raw_t.size // 8, 24))
    edge_idx = np.r_[np.arange(edge_count),
                     np.arange(raw_t.size - edge_count, raw_t.size)]
    centered_t = raw_t - apex_t
    fit_t = centered_t[edge_idx]
    fit_q = raw_q[edge_idx]
    if np.ptp(fit_t) > 0:
        slope, offset = np.polyfit(fit_t, fit_q, 1)
    else:
        slope, offset = 0.0, np.mean(fit_q)
    residual = raw_q - (slope * centered_t + offset)

    apex = float(residual[center_idx - start])
    edge_level = float(np.median(residual[edge_idx]))
    features = np.array([max(0.0, -apex), abs(apex - edge_level),
                         np.sqrt(np.mean(residual ** 2))], dtype=np.float32)

    # Keep the candidate at window_size // 2, as expected by the regressors.
    grid = apex_t + (np.arange(window_size) - window_size // 2) * (
        target_span_c / window_size)
    unique_t, unique_idx = np.unique(raw_t, return_index=True)
    normalized = residual / max(np.max(np.abs(residual)), np.finfo(float).eps)
    area = np.interp(grid, unique_t, normalized[unique_idx])
    resampled_heat_flow = np.interp(grid, unique_t, raw_q[unique_idx])
    return (grid.astype(np.float32), resampled_heat_flow.astype(np.float32),
            area.astype(np.float32), features)


def _detrend_and_orient_window(hf_window):
    """Remove local offset and slope while retaining physical peak sign.

    Voltage-domain development curves are globally polarity-aligned with
    heat-flow curves before this function is called.  Local detrending then
    removes offset and slope without turning upward disturbances into
    apparent endothermic peaks.
    """
    y = np.asarray(hf_window, dtype=np.float64)
    n = y.size
    if n < 4:
        return y.copy()

    edge_n = max(4, min(n // 8, 24))
    edge_idx = np.r_[np.arange(edge_n), np.arange(n - edge_n, n)]
    x = np.arange(n, dtype=np.float64)
    slope, intercept = np.polyfit(x[edge_idx], y[edge_idx], 1)
    residual = y - (slope * x + intercept)

    return residual


def normalize_window(hf_window):
    """Return a unit-invariant, locally detrended, sign-preserving window."""
    residual = _detrend_and_orient_window(hf_window)
    max_val = np.max(np.abs(residual))
    if max_val > np.finfo(np.float64).eps:
        residual = residual / max_val
    return residual.astype(np.float32)


def extract_scalar_features(hf_win):
    """Extract three dimensionless strength features.

    Absolute uV/mW values would leak the acquisition domain into the model.
    The features are therefore expressed relative to robust edge noise:
    log depth-to-noise, log prominence-to-noise and log depth-to-RMS.
    """
    residual = _detrend_and_orient_window(hf_win)
    n = residual.size
    edge_n = max(4, min(n // 8, 24))
    edges = np.r_[residual[:edge_n], residual[-edge_n:]]

    edge_median = float(np.median(edges))
    edge_mad = float(np.median(np.abs(edges - edge_median)))
    noise = 1.4826 * edge_mad
    if edges.size > 2:
        diff_noise = float(np.std(np.diff(edges), ddof=1) / np.sqrt(2.0))
        noise = max(noise, diff_noise)

    scale_floor = max(float(np.max(np.abs(residual))) * 1e-4, 1e-12)
    noise = max(noise, scale_floor)

    centre = n // 2
    half = max(2, min(5, n // 16))
    centre_val = float(np.median(
        residual[max(0, centre - half):min(n, centre + half + 1)]))
    depth = max(0.0, -centre_val)
    prominence = abs(centre_val - edge_median)
    rms = max(float(np.sqrt(np.mean(np.square(residual)))), scale_floor)

    features = np.array([
        np.log1p(depth / noise),
        np.log1p(prominence / noise),
        np.log1p(depth / rms),
    ], dtype=np.float64)
    return np.clip(features, 0.0, 12.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Threshold-crossing boundary detection
# ---------------------------------------------------------------------------

def gradient_descend_boundaries(hf_window, center_idx,
                                 peak_fraction=0.90,
                                 smooth=True,
                                 sg_window=31,
                                 sg_order=3):
    """Find left and right boundaries by threshold crossing from the peak apex.

    Starting from the centre, walk outward until the smoothed signal has
    returned to within ``(1 - peak_fraction)`` of the baseline.  This gives
    wider, more physically meaningful boundaries than gradient descent for
    sharp DSC peaks.

    Works for both endothermic valleys and exothermic peaks — the direction
    is detected automatically.

    Args:
        hf_window: 1D window of heat flow values (may be un-normalised).
        center_idx: Index of the peak apex within the window.
        peak_fraction: Fraction of peak depth/height to cross before stopping.
                       0.9 = stop when signal returns to 90% of baseline.
        smooth: Whether to smooth before boundary search.
        sg_window: Savitzky-Golay window length (must be odd).
        sg_order: Savitzky-Golay polynomial order.

    Returns:
        (left_idx, right_idx) — boundary indices in window coordinates.
    """
    window_size = len(hf_window)

    if smooth and window_size >= sg_window:
        work = savgol_filter(hf_window, sg_window, sg_order)
    else:
        work = hf_window.copy()

    apex_val = work[center_idx]

    # Detect direction: peak (maximum) or valley (minimum)?
    left_nbr  = work[max(0, center_idx - 1)]
    right_nbr = work[min(window_size - 1, center_idx + 1)]
    is_valley = (apex_val < left_nbr) and (apex_val < right_nbr)

    if is_valley:
        # Valley: threshold rises from bottom toward baseline (~0)
        depth = abs(apex_val)
        threshold = apex_val + depth * peak_fraction

        # --- left boundary ---
        left_idx = 0
        for i in range(center_idx, -1, -1):
            if work[i] < threshold:
                left_idx = i
            else:
                break

        # --- right boundary ---
        right_idx = window_size - 1
        for i in range(center_idx, window_size):
            if work[i] < threshold:
                right_idx = i
            else:
                break
    else:
        # Peak: threshold falls from top toward baseline (~0)
        height = apex_val
        threshold = apex_val - height * peak_fraction

        # --- left boundary ---
        left_idx = 0
        for i in range(center_idx, -1, -1):
            if work[i] > threshold:
                left_idx = i
            else:
                break

        # --- right boundary ---
        right_idx = window_size - 1
        for i in range(center_idx, window_size):
            if work[i] > threshold:
                right_idx = i
            else:
                break

    return left_idx, right_idx


def find_boundaries_full_curve(temp, hf, center_idx,
                                peak_fraction=0.90,
                                sg_window=31,
                                sg_order=3,
                                n_std=3.0):
    """Find peak boundaries at the nearest local maxima on each side.

    For a valley (melt peak), walk outward from centre along the smoothed
    curve and stop at the first local maximum on each side — the natural
    "shoulders" of the valley.

    Args:
        temp: Full temperature array.
        hf: Full heat flow array.
        center_idx: Index of the peak centre.
        peak_fraction: Unused (kept for compatibility).
        sg_window: Savitzky-Golay window for smoothing.
        sg_order: Savitzky-Golay polynomial order.
        n_std: Unused (kept for compatibility).

    Returns:
        (left_temp, right_temp) — boundary temperatures.
    """
    n = len(hf)

    if n >= sg_window:
        work = savgol_filter(hf, sg_window, sg_order)
    else:
        work = hf.astype(np.float64)

    # Snap to the nearest SG-smoothed extremum of the candidate's sign.
    # Voltage-domain development peaks and heat-flow-domain test peaks can
    # point in opposite directions, so a minimum-only snap is not valid.
    snap_win = 5
    snap_start = max(0, center_idx - snap_win)
    snap_end = min(n, center_idx + snap_win + 1)
    local = work[snap_start:snap_end]
    if work[center_idx] <= np.median(local):
        center_idx = int(snap_start + np.argmin(local))
    else:
        center_idx = int(snap_start + np.argmax(local))

    apt_val = work[center_idx]
    left_neighbor = work[max(0, center_idx - 1)]
    right_neighbor = work[min(n - 1, center_idx + 1)]
    # Use <= so flat-bottomed valleys (equal consecutive minima) are still
    # treated as valleys.  Require at least one neighbour to be strictly
    # higher so that true plateaus are not misidentified.
    is_valley = (apt_val <= left_neighbor
                 and apt_val <= right_neighbor
                 and apt_val < max(left_neighbor, right_neighbor))

    max_search = min(n // 3, 800)  # don't search past ~12 degC

    if is_valley:
        # Walk LEFT from centre, find first local maximum (point higher
        # than both neighbours) — the left shoulder of the valley.
        left_idx = center_idx - 1
        while left_idx > 2:
            if (work[left_idx] > work[left_idx - 1]
                    and work[left_idx] > work[left_idx + 1]):
                break
            left_idx -= 1
            if center_idx - left_idx > max_search:
                left_idx = center_idx - max_search
                break

        # Walk RIGHT from centre, find first local maximum.
        right_idx = center_idx + 1
        while right_idx < n - 3:
            if (work[right_idx] > work[right_idx - 1]
                    and work[right_idx] > work[right_idx + 1]):
                break
            right_idx += 1
            if right_idx - center_idx > max_search:
                right_idx = center_idx + max_search
                break
    else:
        # For a peak (upward), find nearest local minima as shoulders.
        left_idx = center_idx - 1
        while left_idx > 2:
            if (work[left_idx] < work[left_idx - 1]
                    and work[left_idx] < work[left_idx + 1]):
                break
            left_idx -= 1
            if center_idx - left_idx > max_search:
                left_idx = center_idx - max_search
                break

        right_idx = center_idx + 1
        while right_idx < n - 3:
            if (work[right_idx] < work[right_idx - 1]
                    and work[right_idx] < work[right_idx + 1]):
                break
            right_idx += 1
            if right_idx - center_idx > max_search:
                right_idx = center_idx + max_search
                break

    # Clamp to valid array bounds
    left_idx = max(0, min(n - 1, left_idx))
    right_idx = max(0, min(n - 1, right_idx))
    return temp[left_idx], temp[right_idx]


# ---------------------------------------------------------------------------
# Baseline construction — iterative polynomial fitting
# ---------------------------------------------------------------------------

def auto_detect_baseline_start(temp, hf, sg_window=51, safety_margin=10.0):
    """Automatically detect where the real baseline begins.

    Uses rolling variance of the SG-smoothed residual: when the variance
    drops and stabilises at a low level, the instrument has finished its
    initial warm-up / stabilisation phase.

    Args:
        temp: (N,) temperature array.
        hf: (N,) heat flow array.
        sg_window: SG window for smoothing.
        safety_margin: extra °C added after detection to ensure robust exclusion.

    Returns:
        float: recommended minimum temperature for baseline fitting.
    """
    from scipy.signal import savgol_filter
    n = len(temp)
    if n < 200:
        return 80.0  # too short to auto-detect

    # Smooth and extract high-frequency residual
    smooth = savgol_filter(hf, min(sg_window, n // 2 * 2 + 1), 3)
    trend = savgol_filter(hf, min(sg_window * 5, n // 2 * 2 + 1), 2)
    residual = np.abs(smooth - trend)

    # Rolling std of residual
    window = min(200, n // 10)
    half = window // 2
    roll_std = np.array([
        np.std(residual[max(0, i - half):min(n, i + half)])
        for i in range(n)
    ])

    # SG smooth the rolling std for cleaner detection
    roll_smooth = savgol_filter(roll_std, min(sg_window, n // 2 * 2 + 1), 2)

    # Only consider data above 50 °C (skip the very beginning)
    warm_mask = temp > 50.0
    if not warm_mask.any():
        return 80.0

    warm_start = np.where(warm_mask)[0][0]
    # The stable baseline std is roughly the minimum in the upper half
    mid = n // 2
    min_std = np.min(roll_smooth[mid:]) if mid < n else np.min(roll_smooth[warm_start:])
    threshold = min_std * 2.0  # twice the minimum = clearly stabilised

    # Find the first index after warm_start where rolling std stays below threshold
    candidates = np.where(warm_mask & (roll_smooth < threshold))[0]
    if len(candidates) == 0:
        return 80.0

    # Take the first candidate AFTER a clear downward trend
    # (skip isolated drops)
    for idx in candidates:
        if idx < warm_start + 50:
            continue
        # Check that the next 100 points are also mostly below threshold
        future = roll_smooth[idx:min(n, idx + 200)]
        if len(future) > 0 and np.mean(future < threshold) > 0.7:
            auto_temp = float(temp[idx]) + safety_margin
            # Clamp reasonable range
            auto_temp = max(50.0, min(auto_temp, temp[-1] * 0.6))
            return round(auto_temp, 1)

    return 80.0  # fallback


def construct_baseline_iterative_poly(temp, hf, peak_boundaries,
                                       poly_degree=None,
                                       max_iter=None,
                                       convergence_tol=None,
                                       n_std=3.0,
                                       min_temp=None,
                                       max_temp=None,
                                       verbose=False):
    """Construct baseline via iterative polynomial fitting.

    Iteratively fits a polynomial to the "baseline" points (those outside
    detected peak regions), then subtracts the baseline and re-identifies
    peak regions from the residual.  Repeats until convergence.

    Args:
        temp: (N,) temperature array.
        hf: (N,) heat flow array (raw, not normalised).
        peak_boundaries: list of (t_start, t_end) tuples.
        poly_degree: polynomial degree (default from Config).
        max_iter: maximum iterations (default from Config).
        convergence_tol: stop when mean baseline change < this (mW).
        n_std: number of std above residual noise to flag peak regions.
        min_temp: exclude data below this temperature from baseline fit.
                  Defaults to Config.MIN_TEMPERATURE.
        max_temp: exclude data above this temperature (optional).

    Returns:
        baseline: (N,) baseline heat flow array.
        net_signal: (N,) baseline-subtracted signal (baseline - hf).
    """
    if poly_degree is None:
        poly_degree = Config.BASELINE_POLY_DEGREE
    if max_iter is None:
        max_iter = Config.BASELINE_MAX_ITER
    if convergence_tol is None:
        convergence_tol = Config.BASELINE_CONVERGENCE
    if min_temp is None:
        min_temp = Config.MIN_TEMPERATURE

    n = len(temp)
    # Initial mask: exclude peak regions AND temperature-forbidden zones
    mask = np.ones(n, dtype=bool)
    mask[temp < min_temp] = False
    if max_temp is not None:
        mask[temp > max_temp] = False
    for t_start, t_end in peak_boundaries:
        mask[(temp >= t_start) & (temp <= t_end)] = False

    if not mask.any():
        # No baseline points — fallback to a flat line at min(hf)
        baseline = np.full(n, np.min(hf), dtype=np.float64)
        return baseline, baseline - hf

    baseline = np.zeros(n, dtype=np.float64)
    prev_baseline = np.full(n, np.inf, dtype=np.float64)

    for iteration in range(max_iter):
        # Fit polynomial to current baseline points
        coeffs = np.polyfit(temp[mask], hf[mask], poly_degree)
        baseline = np.polyval(coeffs, temp)

        # Net signal (positive = endothermic, i.e. baseline above signal)
        net = baseline - hf

        # Check convergence
        change = np.mean(np.abs(baseline - prev_baseline))
        if verbose:
            n_pts = mask.sum()
            print(f"  | .. baseline iter {iteration+1}: {n_pts} baseline pts, "
                  f"change={change:.6f} mW")
        if change < convergence_tol:
            break

        prev_baseline = baseline.copy()

        # Identify peak regions from residual — points where net signal
        # exceeds n_std of the baseline-region noise
        baseline_residual = net[mask]
        noise_std = np.std(baseline_residual) if len(baseline_residual) > 1 else 0.0
        threshold = n_std * max(noise_std, 1e-8)

        # Mark as non-baseline: temperature-forbidden zones, peak
        # boundaries, and points where residual is significant
        new_mask = np.ones(n, dtype=bool)
        new_mask[temp < min_temp] = False
        if max_temp is not None:
            new_mask[temp > max_temp] = False
        for t_start, t_end in peak_boundaries:
            new_mask[(temp >= t_start) & (temp <= t_end)] = False
        # Also exclude points with significant residual
        new_mask[net > threshold] = False

        if np.array_equal(mask, new_mask):
            if verbose:
                print(f"  | .. baseline converged at iter {iteration+1} "
                      f"(mask unchanged)")
            break
        mask = new_mask

    return baseline, net


def calculate_enthalpy(temp, hf, baseline, peak_boundaries,
                        mass_mg=None, heating_rate=None,
                        temp_col_name='temperature',
                        hf_col_name='heatflow'):
    """Calculate melting enthalpy for each detected peak.

    The net endothermic signal = baseline - hf (positive downward).
    Integrated over temperature and converted to J/g via heating rate
    and sample mass.

    Args:
        temp: (N,) temperature array (°C).
        hf: (N,) heat flow array (mW).
        baseline: (N,) baseline heat flow array (mW).
        peak_boundaries: list of (t_start, t_end) tuples.
        mass_mg: sample mass in mg.  If None, returns enthalpy in mJ
                 (not normalised by mass).
        heating_rate: °C/min.  Default from Config.

    Returns:
        list of dicts with keys:
          t_start, t_end, t_peak, hf_peak, area_mJ, deltaH_Jg
    """
    if heating_rate is None:
        heating_rate = Config.DEFAULT_HEATING_RATE

    # Heating rate: °C/min → °C/s
    beta_C_per_s = heating_rate / 60.0

    results = []
    for t_start, t_end in peak_boundaries:
        # Indices within this peak
        mask = (temp >= t_start) & (temp <= t_end)
        if not mask.any():
            results.append(dict(t_start=t_start, t_end=t_end, t_peak=np.nan,
                                hf_peak=np.nan, area_mJ=0.0, deltaH_Jg=0.0))
            continue

        t_region = temp[mask]
        net = baseline[mask] - hf[mask]  # positive = endothermic
        net = np.maximum(net, 0)  # clip negative (exothermic) contributions

        # Peak temperature = temperature at maximum net signal
        peak_idx = np.argmax(net)
        t_peak = t_region[peak_idx]
        hf_peak = hf[mask][peak_idx]

        # Integrate: ∫ net_signal dt = ∫ net_signal dT / β
        # Trapezoidal rule over temperature
        area_mJ = np.trapz(net, t_region) / beta_C_per_s  # mJ

        # Normalise by mass
        deltaH_Jg = area_mJ / mass_mg if (mass_mg and mass_mg > 0) else None

        results.append(dict(
            t_start=round(float(t_start), 3),
            t_end=round(float(t_end), 3),
            t_peak=round(float(t_peak), 3),
            hf_peak=round(float(hf_peak), 4),
            area_mJ=round(float(area_mJ), 4),
            deltaH_Jg=round(float(deltaH_Jg), 4) if deltaH_Jg is not None else None,
        ))

    return results


# ---------------------------------------------------------------------------
# Extrapolated onset / endset temperatures (ASTM standard melting point)
# ---------------------------------------------------------------------------

def calculate_onset_endset(temp, hf, peak_boundaries, baseline,
                            sg_window=11, verbose=False):
    """Calculate extrapolated onset and endset temperatures per ASTM.

    For each endothermic peak (valley), the onset is found by:
      1. Locating the point of maximum slope on the leading (left) edge.
      2. Drawing the tangent line at that point.
      3. Finding the intersection of this tangent with the constructed baseline.

    The endset is found analogously on the trailing (right) edge.

    Args:
        temp: (N,) temperature array.
        hf: (N,) heat flow array.
        peak_boundaries: list of (t_start, t_end) tuples.
        baseline: (N,) constructed baseline array.
        sg_window: Savitzky-Golay window for derivative smoothing (odd).

    Returns:
        list of dicts with keys: t_onset, t_endset, onset_slope, endset_slope.
    """
    from scipy.signal import savgol_filter

    n = len(temp)
    results = []

    for t_start, t_end in peak_boundaries:
        mask = (temp >= t_start) & (temp <= t_end)
        if not mask.any():
            results.append(dict(t_onset=None, t_endset=None))
            continue

        idx_in = np.where(mask)[0]
        idx_center = idx_in[np.argmin(hf[idx_in])]  # deepest point = peak center
        t_center = temp[idx_center]

        # ---- Onset (left / leading edge) ----
        left_mask = (temp >= t_start) & (temp <= t_center)
        left_idx = np.where(left_mask)[0]
        onset_result = None
        if len(left_idx) > sg_window + 5:
            # Smooth the left-edge HF and compute 1st derivative
            left_smooth = savgol_filter(hf[left_idx], sg_window, 2)
            left_deriv = savgol_filter(hf[left_idx], sg_window, 2, deriv=1)
            # Find point of maximum negative slope (steepest descent)
            steepest_i = np.argmin(left_deriv)  # most negative
            t_infl = temp[left_idx[steepest_i]]
            hf_infl = left_smooth[steepest_i]
            slope = left_deriv[steepest_i]

            # Tangent: y = slope * (T - t_infl) + hf_infl
            # Intersect with baseline: solve slope*(T - t_infl) + hf_infl = baseline(T)
            # Find where (tangent - baseline) crosses zero near the inflection
            search_start = max(0, left_idx[0] - 20)
            search_end = min(n - 1, left_idx[-1] + 10)
            search_idx = np.arange(search_start, search_end)
            if len(search_idx) > 0:
                tangent_vals = slope * (temp[search_idx] - t_infl) + hf_infl
                diff = tangent_vals - baseline[search_idx]
                # Find the first zero crossing (from baseline toward peak)
                cross = None
                for k in range(1, len(diff)):
                    if diff[k-1] >= 0 and diff[k] < 0:
                        cross = k
                        break
                    if diff[k-1] <= 0 and diff[k] > 0:
                        cross = k
                        break
                if cross is not None:
                    # Linear interpolation
                    t1, t2 = temp[search_idx[cross-1]], temp[search_idx[cross]]
                    d1, d2 = diff[cross-1], diff[cross]
                    t_onset = t1 - d1 * (t2 - t1) / (d2 - d1)
                    onset_result = dict(t_onset=round(float(t_onset), 3),
                                        onset_slope=round(float(slope), 4))
            # Fallback: use t_start if intersection not found
            if onset_result is None:
                onset_result = dict(t_onset=round(float(t_start), 3),
                                    onset_slope=None)

        else:
            onset_result = dict(t_onset=round(float(t_start), 3), onset_slope=None)

        # ---- Endset (right / trailing edge) ----
        right_mask = (temp >= t_center) & (temp <= t_end)
        right_idx = np.where(right_mask)[0]
        endset_result = None
        if len(right_idx) > sg_window + 5:
            right_smooth = savgol_filter(hf[right_idx], sg_window, 2)
            right_deriv = savgol_filter(hf[right_idx], sg_window, 2, deriv=1)
            # Find point of maximum positive slope (steepest ascent back to baseline)
            steepest_i = np.argmax(right_deriv)
            t_infl = temp[right_idx[steepest_i]]
            hf_infl = right_smooth[steepest_i]
            slope = right_deriv[steepest_i]

            search_start = max(0, right_idx[0] - 10)
            search_end = min(n - 1, right_idx[-1] + 20)
            search_idx = np.arange(search_start, search_end)
            if len(search_idx) > 0:
                tangent_vals = slope * (temp[search_idx] - t_infl) + hf_infl
                diff = tangent_vals - baseline[search_idx]
                cross = None
                for k in range(1, len(diff)):
                    if diff[k-1] <= 0 and diff[k] > 0:
                        cross = k
                        break
                    if diff[k-1] >= 0 and diff[k] < 0:
                        cross = k
                        break
                if cross is not None:
                    t1, t2 = temp[search_idx[cross-1]], temp[search_idx[cross]]
                    d1, d2 = diff[cross-1], diff[cross]
                    t_endset = t1 - d1 * (t2 - t1) / (d2 - d1)
                    endset_result = dict(t_endset=round(float(t_endset), 3),
                                          endset_slope=round(float(slope), 4))
            if endset_result is None:
                endset_result = dict(t_endset=round(float(t_end), 3), endset_slope=None)
        else:
            endset_result = dict(t_endset=round(float(t_end), 3), endset_slope=None)

        if verbose:
            print(f"  Peak {t_start:.1f}-{t_end:.1f}C: "
                  f"onset={onset_result['t_onset']:.3f}C, "
                  f"endset={endset_result['t_endset']:.3f}C")

        results.append(dict(
            t_onset=onset_result['t_onset'],
            t_endset=endset_result['t_endset'],
            onset_slope=onset_result.get('onset_slope'),
            endset_slope=endset_result.get('endset_slope'),
        ))

    return results


# ---------------------------------------------------------------------------
# Instance generation for training
# ---------------------------------------------------------------------------

def _get_best_match_index(temperature, target_temp):
    """Return the index in `temperature` closest to `target_temp`."""
    return int(np.argmin(np.abs(temperature - target_temp)))


def _map_temp_to_window_index(temp_window, target_temp):
    """Map a real temperature value to the closest index in the window."""
    valid = temp_window != 0  # skip padding
    if not np.any(valid):
        return 0
    diff = np.abs(temp_window - target_temp)
    diff[~valid] = 1e9
    return int(np.argmin(diff))


@timeit
def prepare_instances_from_references(dataset, window_size=None,
                                       augment_count=5, verbose=True,
                                       target_span_c=None):
    """Generate training instances from user-annotated references.

    For each reference annotation:
      1. Find the DSC curve it belongs to.
      2. Locate the local maximum nearest to the annotated centre.
      3. Extract a normalised window.
      4. Map the annotated boundaries to window indices.
      5. (Optionally) apply augmentation — slight shifts, noise, decoy mixing.

    Args:
        dataset: DSCDataset with loaded curves and references.
        window_size: Window size (default Config.WINDOW_SIZE).
        augment_count: Number of augmented copies per reference.
        target_span_c: Optional temperature span. Requires heat flow in mW.
        verbose: Print progress.

    Returns:
        List of dicts, each with keys:
          'area'     — normalised 1D window  (window_size,)
          'type'     — one-hot class vector  (NUM_CLASSES,)
          'center'   — peak centre index     (scalar)
          'box'      — [left_idx, right_idx] (2,)
          'temp_win' — temperature window    (window_size,)  [for reference]
    """
    if window_size is None:
        window_size = Config.WINDOW_SIZE

    type_map = {'melt_peak': 0, 'baseline': 1, 'artifact': 2}
    num_classes = Config.NUM_CLASSES
    instances = []
    half = window_size // 2

    for ref in dataset.references:
        curve = dataset.get_curve_by_name(ref['filename'])
        if curve is None:
            if verbose:
                print(f"  | Warning: curve '{ref['filename']}' not found, skipping")
            continue

        temp, hf = curve.to_numpy()
        t_center = ref['temp_center']
        t_start  = ref['temp_start']
        t_end    = ref['temp_end']
        cls_name = ref['type']
        cls_idx  = type_map.get(cls_name, 2)

        raw_size = (window_size if target_span_c is None
                    else temperature_window_size(temp, target_span_c))
        # Find local maximum nearest to the annotated centre
        center_idx = _get_best_match_index(temp, t_center)

        # Thermocouple-voltage development peaks and heat-flow test peaks
        # can have opposite signs. Snap a melt reference to the nearest local
        # extremum of either sign instead of assuming an endothermic minimum.
        if cls_name == 'melt_peak':
            extrema = np.where(
                find_local_maxima(hf, mode='min') |
                find_local_maxima(hf, mode='max'))[0]
            if len(extrema) > 0:
                best_ext = extrema[np.argmin(np.abs(extrema - center_idx))]
                if abs(best_ext - center_idx) < raw_size // 4:
                    center_idx = best_ext

        if target_span_c is None:
            temp_win, hf_win = extract_window(temp, hf, center_idx, window_size)
            norm_hf = normalize_window(hf_win)
            scalar_feats = extract_scalar_features(hf_win)
        else:
            temp_win, hf_win, norm_hf, scalar_feats = _paper_model_window(
                temp, hf, center_idx, window_size, target_span_c, raw_size)

        # Map boundaries to window indices
        center_win_idx = half
        box_left  = _map_temp_to_window_index(temp_win, t_start)
        box_right = _map_temp_to_window_index(temp_win, t_end)

        # Clamp box to valid range
        box_left  = max(0, min(window_size - 2, box_left))
        box_right = max(box_left + 1, min(window_size - 1, box_right))

        # One-hot type
        type_vec = np.zeros(num_classes, dtype=np.float32)
        type_vec[cls_idx] = 1.0

        # Base instance
        base = {
            'area':       norm_hf,
            'type':       type_vec,
            'center':     np.array([center_win_idx], dtype=np.float32),
            'box':        np.array([box_left, box_right], dtype=np.float32),
            'temp_win':   temp_win,
            'hf_win':     hf_win,
            'features':   scalar_feats,
        }
        instances.append(base)

        # --- Data augmentation ---
        for aug_i in range(augment_count):
            aug_area = norm_hf.copy()

            # Add small Gaussian noise
            noise = np.random.normal(0, 0.02, window_size).astype(np.float32)
            aug_area = aug_area + noise

            # Random intensity scaling
            scale = np.random.uniform(0.9, 1.1)
            aug_area = aug_area * scale

            # Small random shift in the window (simulate imprecise centring)
            shift = np.random.randint(-3, 4)
            if shift != 0:
                aug_area_shifted = np.zeros(window_size, dtype=np.float32)
                if shift > 0:
                    aug_area_shifted[shift:] = aug_area[:-shift]
                    aug_area_shifted[:shift] = aug_area[0]
                else:
                    aug_area_shifted[:shift] = aug_area[-shift:]
                    aug_area_shifted[shift:] = aug_area[-1]
                aug_area = aug_area_shifted

            # Re-normalise
            aug_area = normalize_window(aug_area)

            instances.append({
                'area':       aug_area,
                'type':       type_vec.copy(),
                'center':     np.array([center_win_idx], dtype=np.float32),
                'box':        np.array([box_left, box_right], dtype=np.float32),
                'temp_win':   temp_win.copy(),
                'hf_win':     hf_win.copy(),
                'features':   scalar_feats.copy(),
            })

    if verbose:
        print(f"  | Generated {len(instances)} training instances "
              f"from {len(dataset.references)} references")
        cls_counts = {}
        for inst in instances:
            c = np.argmax(inst['type'])
            cls_counts[c] = cls_counts.get(c, 0) + 1
        for c, count in cls_counts.items():
            names = {0: 'melt_peak', 1: 'baseline', 2: 'artifact'}
            print(f"  | .. {names.get(c, c)}: {count}")

    return instances


# ---------------------------------------------------------------------------
# Instance generation for prediction
# ---------------------------------------------------------------------------

@timeit
def prepare_instances_for_prediction(curve, window_size=None,
                                      min_intensity=None,
                                      min_temperature=None,
                                      verbose=False, target_span_c=None):
    """Generate prediction instances for a single DSC curve.

    Finds all local extrema (maxima + minima), extracts and normalises windows.
    Candidates below ``min_temperature`` are excluded (experimental prep phase).

    Args:
        curve: DSCCurve object.
        window_size: Window size (default Config.WINDOW_SIZE).
        min_intensity: Minimum intensity for local extrema.
        min_temperature: Minimum temperature for candidates.
        target_span_c: Optional temperature span. Requires heat flow in mW.
        verbose: Print progress.

    Returns:
        areas:      (N, window_size) float32 array.
        features:   (N, 3) float32 array — [peak_depth, prominence, rms].
        temp_win:   (N, window_size) float32 array.
        gd_props:   (N, 7) float32 array — [center_temp, center_hf,
                                             left_temp, right_temp, apex_idx_in_window,
                                             full_curve_depth, full_curve_SNR].
    """
    if window_size is None:
        window_size = Config.WINDOW_SIZE
    if min_temperature is None:
        min_temperature = Config.MIN_TEMPERATURE

    temp, hf = curve.to_numpy()
    compensation_events = getattr(curve, 'compensation_events', None)
    compensation_intervals = []
    if compensation_events:
        hf, compensation_diagnostics, compensation_intervals = \
            apply_compensation_events(temp, hf, compensation_events)
        # Retain diagnostics for GUI and reporting without changing the
        # public return signature of this preprocessing function.
        curve.compensation_diagnostics = compensation_diagnostics
        curve.compensation_exclusion_intervals = compensation_intervals
    half = window_size // 2

    lm_mask = find_local_maxima(hf, min_intensity, mode='both')
    lm_indices = np.where(lm_mask)[0]

    # Filter by minimum temperature
    if min_temperature is not None and len(lm_indices) > 0:
        keep = temp[lm_indices] >= min_temperature
        lm_indices = lm_indices[keep]

    # A compensation-table switch is known acquisition metadata, not an
    # unknown material peak.  Suppress its transient before CNN inference.
    if compensation_intervals and len(lm_indices) > 0:
        keep = exclusion_mask(temp[lm_indices], compensation_intervals)
        lm_indices = lm_indices[keep]

    n = len(lm_indices)
    if n == 0:
        if verbose:
            print(f"  | No local extrema found in '{curve.name}'")
        return None, None, None, None

    if target_span_c is not None:
        raw_size = temperature_window_size(temp, target_span_c)
    areas    = np.zeros((n, window_size), dtype=np.float32)
    features = np.zeros((n, 3), dtype=np.float32)
    temp_win = np.zeros((n, window_size), dtype=np.float32)
    gd_props = np.zeros((n, 7), dtype=np.float32)

    for i, center_idx in enumerate(lm_indices):
        if target_span_c is None:
            tw, hw = extract_window(temp, hf, center_idx, window_size)
            area = normalize_window(hw)
            scalar_feats = extract_scalar_features(hw)
        else:
            tw, hw, area, scalar_feats = _paper_model_window(
                temp, hf, center_idx, window_size, target_span_c, raw_size)
        areas[i] = area
        features[i] = scalar_feats
        temp_win[i] = tw

        # Boundary detection on FULL curve (not limited by window size)
        left_temp, right_temp = find_boundaries_full_curve(
            temp, hf, center_idx,
            peak_fraction=Config.PEAK_FRACTION,
            sg_window=Config.SG_BOUNDARY)
        gd_props[i, 0] = temp[center_idx]   # centre temperature
        gd_props[i, 1] = hf[center_idx]     # centre heat flow
        gd_props[i, 2] = left_temp           # left boundary (from full curve)
        gd_props[i, 3] = right_temp          # right boundary (from full curve)
        gd_props[i, 4] = half                # apex in window coords

        # Sign-invariant full-curve peak depth and dimensionless SNR.
        left_idx = int(np.argmin(np.abs(temp - left_temp)))
        right_idx = int(np.argmin(np.abs(temp - right_temp)))
        edge_hf = (hf[left_idx] + hf[right_idx]) / 2.0
        full_depth = abs(float(edge_hf - hf[center_idx]))
        gd_props[i, 5] = full_depth

        shoulder_n = max(6, min(40, window_size // 8))
        noise_segments = []
        if left_idx - shoulder_n >= 0:
            noise_segments.append(hf[left_idx - shoulder_n:left_idx])
        if right_idx + shoulder_n < len(hf):
            noise_segments.append(hf[right_idx + 1:right_idx + 1 + shoulder_n])
        if noise_segments:
            noise_values = np.concatenate(noise_segments).astype(np.float64)
        else:
            noise_values = hw.astype(np.float64)
        # Remove local shoulder drift before estimating high-frequency
        # noise; otherwise a broad peak slope is incorrectly counted as noise.
        nx = np.arange(noise_values.size, dtype=np.float64)
        if noise_values.size >= 4:
            nslope, nintercept = np.polyfit(nx, noise_values, 1)
            noise_residual = noise_values - (nslope * nx + nintercept)
        else:
            noise_residual = noise_values - np.mean(noise_values)
        med = float(np.median(noise_residual))
        noise = 1.4826 * float(np.median(np.abs(noise_residual - med)))
        if noise_residual.size > 2:
            diff_noise = float(np.std(np.diff(noise_residual), ddof=1) /
                               np.sqrt(2.0))
            noise = max(noise, diff_noise)
        noise = max(noise, np.finfo(np.float64).eps)
        gd_props[i, 6] = float(full_depth / noise)

    if verbose:
        print(f"  | Prepared {n} instances from '{curve.name}' "
              f"(min T = {min_temperature} degC)")

    return areas, features, temp_win, gd_props


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def save_instances(instances, folder, prefix=None):
    """Save training/prediction instances as pickle batches."""
    if prefix is None:
        prefix = Config.INSTANCE_PREFIX
    os.makedirs(folder, exist_ok=True)

    batch_size = Config.BATCH_SIZE
    n = len(instances)
    num_batches = math.ceil(n / batch_size)

    for bi in range(num_batches):
        start = bi * batch_size
        end   = min(start + batch_size, n)
        batch = instances[start:end]

        areas = np.stack([inst['area'] for inst in batch])
        types = np.stack([inst['type'] for inst in batch])
        centers = np.stack([inst['center'] for inst in batch])
        boxes   = np.stack([inst['box'] for inst in batch])
        feats = np.stack([inst['features'] if 'features' in inst
                          else extract_scalar_features(inst['hf_win'])
                          for inst in batch])

        out = {
            'area':     areas,
            'features': feats,
            'type':     types,
            'center':   centers,
            'box':      boxes,
        }

        fpath = os.path.join(folder, f"{prefix}{bi}.pickle")
        with open(fpath, 'wb') as f:
            pickle.dump(out, f)

    return num_batches


def load_instance_batch(filepath):
    """Load a single pickle batch. Returns dict or None."""
    if not os.path.isfile(filepath):
        return None
    with open(filepath, 'rb') as f:
        return pickle.load(f)
