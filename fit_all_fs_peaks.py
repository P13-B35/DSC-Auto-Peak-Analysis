from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, savgol_filter


WORK = Path(__file__).resolve().parent
MANIFEST = WORK / "dataset_manifest.csv"
DIAGNOSIS = WORK / "development_peak_diagnosis.csv"
OUT_DIR = WORK / "fs_fit"
PARAMS = OUT_DIR / "fs_fit_params.csv"
FAILURES = OUT_DIR / "fs_fit_failures.csv"
REFERENCES = WORK / "data" / "development_real" / "references.tsv"
STATS = OUT_DIR / "fs_fit_stats.json"


def fs_unit(x: np.ndarray, center: float, width: float, skew: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    width = max(float(width), 1e-8)
    skew = float(skew)
    if abs(skew) < 1e-6:
        z = (x - center) / width
        return np.exp(-math.log(2.0) * (2.0 * z) ** 2)
    arg = 1.0 + 2.0 * skew * (x - center) / width
    y = np.zeros_like(x)
    valid = arg > 1e-10
    z = np.log(arg[valid]) / skew
    y[valid] = np.exp(-math.log(2.0) * z ** 2)
    return y


def fs_with_baseline(x, c0, c1, c2, amplitude, center, width, skew):
    x0 = x - np.mean(x)
    return c0 + c1 * x0 + c2 * x0 * x0 + amplitude * fs_unit(x, center, width, skew)


def odd_window(n: int, preferred: int = 151) -> int:
    w = min(preferred, n if n % 2 else n - 1)
    w = max(7, w)
    return w if w % 2 else w - 1


def fit_bounds(family: str):
    if family == "narrow_metal":
        return 0.4, 12.0, [1.5, 3.0, 5.0, 8.0]
    if family == "broad_polymer":
        return 2.0, 35.0, [5.0, 9.0, 14.0, 22.0]
    return 1.0, 25.0, [3.0, 6.0, 10.0, 16.0]


def r_squared(y: np.ndarray, pred: np.ndarray) -> float:
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return float("nan") if ss_tot <= 0 else 1.0 - ss_res / ss_tot


def fit_one(row: pd.Series, center_hint: float) -> dict:
    df = pd.read_excel(row["derived_curve"])
    t = pd.to_numeric(df["temperature"], errors="coerce").to_numpy(float)
    y = pd.to_numeric(df["heatflow"], errors="coerce").to_numpy(float)
    valid = np.isfinite(t) & np.isfinite(y)
    t, y = t[valid], y[valid]
    order = np.argsort(t)
    t, y = t[order], y[order]

    lo = float(row["search_min_C"])
    hi = float(row["search_max_C"])
    mask = (t >= lo) & (t <= hi)
    x, yy = t[mask], y[mask]
    if len(x) < 50:
        raise ValueError(f"too few points in {lo}-{hi} C")
    ys = savgol_filter(yy, odd_window(len(yy)), 3)
    stride = max(1, int(np.ceil(len(x) / 1200)))
    xf, yf = x[::stride], ys[::stride]
    edge_n = max(10, int(0.12 * len(xf)))
    edge_idx = np.r_[0:edge_n, len(xf)-edge_n:len(xf)]
    coef = np.polyfit(xf[edge_idx], yf[edge_idx], 1)
    baseline0 = np.polyval(coef, xf)
    residual0 = yf - baseline0
    center0 = float(center_hint)
    apex_idx = int(np.argmin(np.abs(xf - center0)))
    amplitude0 = float(residual0[apex_idx])
    if amplitude0 <= 0:
        apex_idx = int(np.argmax(residual0))
        center0 = float(xf[apex_idx])
        amplitude0 = float(max(residual0[apex_idx], np.ptp(yf) * 0.2, 1e-3))

    scale = max(float(np.ptp(yf)), abs(amplitude0), 1e-3)
    span = max(float(np.ptp(xf)), 1.0)
    w_min, w_max, width_starts = fit_bounds(str(row["peak_family"]))
    lower = [float(np.min(yf) - 2.5 * scale), -4 * scale / span,
             -4 * scale / span**2, 1e-5, lo, w_min, -0.92]
    upper = [float(np.max(yf) + 2.5 * scale), 4 * scale / span,
             4 * scale / span**2, 6 * scale, hi, w_max, 0.92]
    c0 = float(np.median(np.r_[yf[:edge_n], yf[-edge_n:]]))
    c1 = float(coef[0])
    best = None
    for w0 in width_starts:
        for s0 in (-0.60, -0.25, -0.05, 0.20, 0.55):
            try:
                popt, _ = curve_fit(
                    fs_with_baseline, xf, yf,
                    p0=[c0, c1, 0.0, amplitude0, center0, w0, s0],
                    bounds=(lower, upper), maxfev=50000,
                )
                pred0 = fs_with_baseline(xf, *popt)
                rmse0 = float(np.sqrt(np.mean((yf - pred0) ** 2)))
                if best is None or rmse0 < best[0]:
                    best = (rmse0, popt)
            except Exception:
                continue
    if best is None:
        raise RuntimeError("all FS multistart fits failed")

    popt = best[1]
    pred = fs_with_baseline(x, *popt)
    c0, c1, c2, amplitude, center, width, skew = [float(v) for v in popt]
    peak = amplitude * fs_unit(x, center, width, skew)
    residual = ys - pred
    active = peak >= max(0.025 * amplitude, 1e-12)
    idx = np.where(active)[0]
    t_start = float(x[idx[0]]) if len(idx) else center - 2 * width
    t_end = float(x[idx[-1]]) if len(idx) else center + 2 * width
    return {
        "c0_uV": c0, "c1_uV_per_C": c1, "c2_uV_per_C2": c2,
        "amplitude_uV": amplitude, "center_C": center, "width_C": width,
        "skew": skew, "area_uV_C": float(np.trapz(peak, x)),
        "t_start_C": t_start, "t_end_C": t_end,
        "r2": r_squared(ys, pred),
        "rmse_uV": float(np.sqrt(np.mean(residual ** 2))),
        "nrmse_to_amplitude": float(np.sqrt(np.mean(residual ** 2)) / max(abs(amplitude), 1e-12)),
        "residual_rms_uV": float(np.std(residual)), "n_fit_points": int(len(x)),
    }


def auxiliary_refs(row: pd.Series, peak_ref: dict, max_baseline: int = 4, max_artifact: int = 4):
    df = pd.read_excel(row["derived_curve"])
    t = pd.to_numeric(df["temperature"], errors="coerce").to_numpy(float)
    y = pd.to_numeric(df["heatflow"], errors="coerce").to_numpy(float)
    valid = np.isfinite(t) & np.isfinite(y)
    t, y = t[valid], y[valid]
    order = np.argsort(t)
    t, y = t[order], y[order]
    ys = savgol_filter(y, odd_window(len(y), 101), 3)
    name = Path(row["derived_curve"]).name
    p0, p1 = peak_ref["temp_start"], peak_ref["temp_end"]
    refs = []
    for center in np.linspace(max(80.0, float(np.min(t)) + 5.0), float(np.max(t)) - 5.0, 10):
        if center < p0 - 12.0 or center > p1 + 12.0:
            refs.append({"filename": name, "temp_center": float(center),
                         "temp_start": float(center - 3.0), "temp_end": float(center + 3.0),
                         "type": "baseline"})
        if len(refs) >= max_baseline:
            break
    prominence = max(float(np.std(ys)) * 0.35, 1e-6)
    extrema = []
    for sign in (1.0, -1.0):
        idx, props = find_peaks(sign * ys, prominence=prominence, distance=max(5, len(ys) // 250))
        for k, i in enumerate(idx):
            tc = float(t[i])
            if tc >= 80.0 and not (p0 - 8.0 <= tc <= p1 + 8.0):
                extrema.append((float(props["prominences"][k]), tc))
    extrema.sort(reverse=True)
    used = []
    for _, center in extrema:
        if any(abs(center - u) < 5.0 for u in used):
            continue
        refs.append({"filename": name, "temp_center": center,
                     "temp_start": center - 2.5, "temp_end": center + 2.5,
                     "type": "artifact"})
        used.append(center)
        if len(used) >= max_artifact:
            break
    return refs


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(MANIFEST)
    diagnosis = pd.read_csv(DIAGNOSIS)
    dev = manifest[(manifest["role"] == "development") & (manifest["qc_status"] == "usable")].copy()
    dev = dev.merge(diagnosis[["curve_id", "diagnostic_center_C"]], on="curve_id", how="left")
    rows, failures, references = [], [], []
    for _, row in dev.iterrows():
        try:
            params = fit_one(row, float(row["diagnostic_center_C"]))
            accepted = bool(params["r2"] >= 0.96 and params["nrmse_to_amplitude"] <= 0.20)
            record = {"curve_id": row["curve_id"], "filename": Path(row["derived_curve"]).name,
                      "source_file": row["source_file"], "derived_curve": row["derived_curve"],
                      "material": row["material"], "peak_family": row["peak_family"],
                      "mass_mg": row["mass_mg"], "heating_rate_C_min": row["heating_rate_C_min"],
                      "fit_accepted": accepted, **params}
            rows.append(record)
            if accepted:
                peak_ref = {"filename": record["filename"], "temp_center": params["center_C"],
                            "temp_start": params["t_start_C"], "temp_end": params["t_end_C"],
                            "type": "melt_peak"}
                references.append(peak_ref)
                references.extend(auxiliary_refs(row, peak_ref))
        except Exception as exc:
            failures.append({"curve_id": row["curve_id"], "material": row["material"],
                             "error": f"{type(exc).__name__}: {exc}"})

    params_df = pd.DataFrame(rows)
    params_df.to_csv(PARAMS, index=False, encoding="utf-8-sig")
    pd.DataFrame(failures).to_csv(FAILURES, index=False, encoding="utf-8-sig")
    ref_df = pd.DataFrame(references)[["filename", "temp_center", "temp_start", "temp_end", "type"]]
    ref_df.to_csv(REFERENCES, sep="\t", index=False)
    accepted = params_df[params_df["fit_accepted"]].copy()
    summary = {
        "n_development_curves": int(len(dev)), "n_fit_success": int(len(params_df)),
        "n_fit_accepted": int(len(accepted)),
        "n_fit_rejected": int((~params_df["fit_accepted"]).sum()) if len(params_df) else 0,
        "n_fit_failures": int(len(failures)),
        "accepted_by_material": accepted.groupby("material").size().to_dict(),
        "fit_quality_by_material": accepted.groupby("material").agg(
            r2_median=("r2", "median"), nrmse_median=("nrmse_to_amplitude", "median"),
            width_median_C=("width_C", "median"), skew_median=("skew", "median"),
        ).round(5).to_dict("index"),
        "reference_counts": ref_df["type"].value_counts().to_dict(),
        "note": "All fitted curves are non-10 C/min voltage curves; locked test curves were not read.",
    }
    STATS.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if len(params_df):
        print("\nRejected fits")
        print(params_df.loc[~params_df["fit_accepted"], ["curve_id", "material", "center_C",
              "width_C", "skew", "r2", "nrmse_to_amplitude"]].to_string(index=False))
    if failures:
        print("\nFailures\n" + pd.DataFrame(failures).to_string(index=False))
    print(f"Parameters: {PARAMS}\nReferences: {REFERENCES}")


if __name__ == "__main__":
    main()
