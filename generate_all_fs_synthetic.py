from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


WORK = Path(__file__).resolve().parent
PARAMS = WORK / "fs_fit" / "fs_fit_params.csv"
SYN_ROOT = WORK / "data"


def fs_unit(x: np.ndarray, center: float, width: float, skew: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    width = max(float(width), 1e-8)
    if abs(skew) < 1e-6:
        z = (x - center) / width
        return np.exp(-math.log(2.0) * (2.0 * z) ** 2)
    arg = 1.0 + 2.0 * skew * (x - center) / width
    y = np.zeros_like(x)
    valid = arg > 1e-10
    z = np.log(arg[valid]) / skew
    y[valid] = np.exp(-math.log(2.0) * z ** 2)
    return y


def odd_window(n: int, preferred: int) -> int:
    w = min(preferred, n if n % 2 else n - 1)
    w = max(7, w)
    return w if w % 2 else w - 1


def load_template(row: pd.Series, rng: np.random.Generator):
    df = pd.read_excel(row["derived_curve"])
    t = pd.to_numeric(df["temperature"], errors="coerce").to_numpy(float)
    y = pd.to_numeric(df["heatflow"], errors="coerce").to_numpy(float)
    valid = np.isfinite(t) & np.isfinite(y)
    t, y = t[valid], y[valid]
    order = np.argsort(t)
    t, y = t[order], y[order]

    # Remove the measured target peak before borrowing baseline/noise. The
    # replacement is a straight interpolation across a padded peak interval.
    width = float(row["width_C"])
    left = float(row["t_start_C"] - 1.0 * width)
    right = float(row["t_end_C"] + 1.0 * width)
    i0 = int(np.searchsorted(t, left, side="left"))
    i1 = int(np.searchsorted(t, right, side="right"))
    clean = y.copy()
    if 0 < i0 < i1 < len(t) - 1:
        clean[i0:i1] = np.interp(t[i0:i1], [t[i0 - 1], t[i1]], [clean[i0 - 1], clean[i1]])
    preferred = max(301, min(1501, (len(clean) // 12) | 1))
    baseline = savgol_filter(clean, odd_window(len(clean), preferred), 2)
    residual = clean - baseline
    # Circularly shift real residuals so synthetic curves retain measured
    # coloured noise but are not pointwise copies of the source trace.
    residual = np.roll(residual, int(rng.integers(0, max(len(residual), 1))))
    residual *= float(rng.uniform(0.70, 1.35))
    return t, baseline, residual


def sample_peak(row: pd.Series, t: np.ndarray, rng: np.random.Generator):
    width = float(np.clip(float(row["width_C"]) * rng.lognormal(0.0, 0.10), 0.35, 40.0))
    skew = float(np.clip(float(row["skew"]) + rng.normal(0.0, 0.05), -0.90, 0.90))
    amplitude = float(max(0.05, float(row["amplitude_uV"]) * rng.lognormal(0.0, 0.18)))
    lower = float(np.min(t) + 3.0 * width)
    upper = float(np.max(t) - 3.0 * width)
    center = float(np.clip(float(row["center_C"]) + rng.normal(0.0, 0.35 * width), lower, upper))
    unit = fs_unit(t, center, width, skew)
    active = unit >= 0.025
    idx = np.where(active)[0]
    start = float(t[idx[0]]) if len(idx) else center - 2.5 * width
    end = float(t[idx[-1]]) if len(idx) else center + 2.5 * width
    return {"center": center, "width": width, "skew": skew,
            "amplitude": amplitude, "unit": unit, "start": start, "end": end}


def nonoverlapping_center(t: np.ndarray, occupied: list[tuple[float, float]], rng: np.random.Generator):
    lo, hi = max(82.0, float(np.min(t)) + 8.0), float(np.max(t)) - 8.0
    for _ in range(100):
        center = float(rng.uniform(lo, hi))
        if all(center < a - 7.0 or center > b + 7.0 for a, b in occupied):
            return center
    return None


def add_baseline_refs(filename: str, t: np.ndarray, occupied: list[tuple[float, float]], refs: list[dict]):
    lo, hi = max(85.0, float(np.min(t)) + 8.0), float(np.max(t)) - 8.0
    n = 0
    for center in np.linspace(lo, hi, 14):
        if all(center < a - 8.0 or center > b + 8.0 for a, b in occupied):
            refs.append({"filename": filename, "temp_center": float(center),
                         "temp_start": float(center - 3.0), "temp_end": float(center + 3.0),
                         "type": "baseline"})
            n += 1
        if n >= 4:
            break


def generate_curve(material_rows: pd.DataFrame, curve_number: int, family: str, rng: np.random.Generator):
    base = material_rows.iloc[int(rng.integers(0, len(material_rows)))]
    t, baseline, residual = load_template(base, rng)
    signal = baseline + residual
    filename = f"synthetic_{base['material']}_{curve_number:04d}.xlsx"
    refs = []
    occupied = []
    meta = {"filename": filename, "material": base["material"], "family": family,
            "template_curve_id": base["curve_id"]}

    primary = sample_peak(base, t, rng)
    signal += primary["amplitude"] * primary["unit"]
    refs.append({"filename": filename, "temp_center": primary["center"],
                 "temp_start": primary["start"], "temp_end": primary["end"],
                 "type": "melt_peak"})
    occupied.append((primary["start"], primary["end"]))
    meta.update({f"primary_{k}": v for k, v in primary.items() if k != "unit"})

    # Broad polymer traces occasionally receive a resolvable second melt peak,
    # producing shoulder/double-peak training examples without using test data.
    if family == "broad" and rng.random() < 0.30:
        other = material_rows.iloc[int(rng.integers(0, len(material_rows)))]
        second = sample_peak(other, t, rng)
        direction = float(rng.choice([-1.0, 1.0]))
        separation = rng.uniform(1.2, 2.1) * max(primary["width"], second["width"])
        second["center"] = float(np.clip(primary["center"] + direction * separation,
                                          np.min(t) + 3 * second["width"],
                                          np.max(t) - 3 * second["width"]))
        second["amplitude"] *= float(rng.uniform(0.35, 0.80))
        second["unit"] = fs_unit(t, second["center"], second["width"], second["skew"])
        idx = np.where(second["unit"] >= 0.025)[0]
        second["start"] = float(t[idx[0]]) if len(idx) else second["center"] - 2.5 * second["width"]
        second["end"] = float(t[idx[-1]]) if len(idx) else second["center"] + 2.5 * second["width"]
        signal += second["amplitude"] * second["unit"]
        refs.append({"filename": filename, "temp_center": second["center"],
                     "temp_start": second["start"], "temp_end": second["end"],
                     "type": "melt_peak"})
        occupied.append((second["start"], second["end"]))
        meta.update({f"secondary_{k}": v for k, v in second.items() if k != "unit"})

    # Add 1-2 peak-shaped hard negatives with lower SNR and varied sign/width.
    for art_i in range(int(rng.integers(1, 3))):
        center = nonoverlapping_center(t, occupied, rng)
        if center is None:
            continue
        width = float(np.clip(primary["width"] * rng.uniform(0.25, 1.40), 0.5, 22.0))
        skew = float(rng.uniform(-0.75, 0.75))
        amp = float(primary["amplitude"] * rng.uniform(0.04, 0.28))
        sign = float(rng.choice([-1.0, 1.0]))
        unit = fs_unit(t, center, width, skew)
        signal += sign * amp * unit
        idx = np.where(unit >= 0.04)[0]
        start = float(t[idx[0]]) if len(idx) else center - 2.0 * width
        end = float(t[idx[-1]]) if len(idx) else center + 2.0 * width
        refs.append({"filename": filename, "temp_center": center,
                     "temp_start": start, "temp_end": end, "type": "artifact"})
        occupied.append((start, end))
        meta.update({f"artifact{art_i+1}_center": center, f"artifact{art_i+1}_width": width,
                     f"artifact{art_i+1}_amplitude": amp, f"artifact{art_i+1}_sign": sign})

    add_baseline_refs(filename, t, occupied, refs)
    return pd.DataFrame({"temperature": t, "heatflow": signal}), refs, meta


def generate_family(params: pd.DataFrame, family: str, materials: list[str], n_per_material: int,
                    seed: int, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    refs, meta = [], []
    curve_no = 0
    for material in materials:
        sub = params[params["material"] == material].reset_index(drop=True)
        if sub.empty:
            continue
        for _ in range(n_per_material):
            curve_no += 1
            curve, curve_refs, curve_meta = generate_curve(sub, curve_no, family, rng)
            curve.to_excel(output_dir / curve_meta["filename"], index=False)
            refs.extend(curve_refs)
            meta.append(curve_meta)
    ref_df = pd.DataFrame(refs)[["filename", "temp_center", "temp_start", "temp_end", "type"]]
    ref_df.to_csv(output_dir / "references.tsv", sep="\t", index=False)
    pd.DataFrame(meta).to_csv(output_dir / "synthetic_metadata.csv", index=False, encoding="utf-8-sig")
    return {"family": family, "curves": curve_no, "references": len(ref_df),
            "reference_counts": ref_df["type"].value_counts().to_dict(), "output": str(output_dir)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-material", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260713)
    args = parser.parse_args()
    params = pd.read_csv(PARAMS)
    params = params[params["fit_accepted"].astype(str).str.lower().isin(["true", "1"])].copy()
    narrow = generate_family(params, "narrow", ["In", "Sn", "Pb", "Bi", "urea"],
                             args.n_per_material, args.seed, SYN_ROOT / "synthetic_narrow")
    broad = generate_family(params, "broad", ["PBT", "PET", "PA6", "PA66"],
                            args.n_per_material, args.seed + 1000, SYN_ROOT / "synthetic_broad")
    print(narrow)
    print(broad)


if __name__ == "__main__":
    main()
