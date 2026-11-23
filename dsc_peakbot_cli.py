"""Command-line wrappers around the retained DSC detection/thermal functions."""
import argparse
import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def positive(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return number

def load_rows(path, delimiter=","):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))

def check_curve(curve):
    import numpy as np
    t, h = curve.to_numpy()
    if len(t) < 15 or not np.all(np.isfinite(t)) or not np.all(np.isfinite(h)):
        raise ValueError("Insufficient or non-finite numeric input: " + curve.name)
    if not np.all(np.diff(t) > 0):
        raise ValueError("Temperature must be strictly increasing: " + curve.name)

def detect(args):
    from peakbot.dsc_peakbot.data_loader import DSCDataset
    from peakbot.dsc_peakbot.main import run_detection
    folder = args.curves.resolve()
    model = args.model.resolve()
    if not folder.is_dir() or not model.is_file():
        raise ValueError("Input curve directory or trained model is missing")
    expected = {p.name for p in folder.iterdir() if p.suffix.lower() in (".xlsx", ".xls") and not p.name.startswith("~")}
    dataset = DSCDataset().load_folder(str(folder))
    loaded = {curve.name for curve in dataset.curves}
    if not expected or loaded != expected:
        raise ValueError("No input curves or some workbooks could not be read: " + repr(sorted(expected - loaded)))
    for curve in dataset.curves:
        check_curve(curve)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Broad smoothing follows the retained family-training configuration.
    from peakbot.dsc_peakbot.config import Config
    Config.SG_BOUNDARY = 151 if args.family == "broad" else 31
    run_detection(str(folder), str(model), str(args.output), paper_family=args.family)

def quantify(args):
    import numpy as np
    from peakbot.dsc_peakbot.data_loader import DSCDataset
    from peakbot.dsc_peakbot.preprocessing import construct_baseline_iterative_poly, calculate_enthalpy
    from peakbot.dsc_peakbot.thermal_metrics import calculate_extrapolated_onset_endset
    if args.metadata:
        if args.mass_mg is not None or args.heating_rate is not None:
            raise ValueError("Use --metadata OR both scalar values, not both modes")
        metadata = {}
        for row in load_rows(args.metadata):
            name = row["filename"]
            if name in metadata:
                raise ValueError("Duplicate metadata filename: " + name)
            metadata[name] = (positive(row["mass_mg"]), positive(row["heating_rate_C_min"]))
    else:
        if args.mass_mg is None or args.heating_rate is None:
            raise ValueError("Provide --metadata or both --mass-mg and --heating-rate")
        metadata = None
    selected = [r for r in load_rows(args.detections, "\t") if r["PeakType"] == "melt_peak"]
    dataset = DSCDataset().load_folder(str(args.curves.resolve()))
    curves = {curve.name: curve for curve in dataset.curves}
    grouped = {}
    for row in selected:
        grouped.setdefault(row["Filename"], []).append(row)
    outputs = []
    for name, rows in grouped.items():
        if name not in curves:
            raise ValueError("Detection workbook not loaded: " + name)
        curve = curves[name]
        check_curve(curve)
        temp, signal = curve.to_numpy()
        boundaries = [(float(row["TempStart"]), float(row["TempEnd"])) for row in rows]
        for start, end in boundaries:
            if not (math.isfinite(start) and math.isfinite(end) and temp[0] <= start < end <= temp[-1]):
                raise ValueError("Invalid or out-of-range peak boundaries: " + name)
        if metadata is not None:
            if name not in metadata:
                raise ValueError("Missing mass/rate metadata for " + name)
            mass, rate = metadata[name]
        else:
            mass, rate = args.mass_mg, args.heating_rate
        baseline, _ = construct_baseline_iterative_poly(temp, signal, boundaries)
        thermal = calculate_enthalpy(temp, signal, baseline, boundaries, mass_mg=mass, heating_rate=rate)
        tangent = calculate_extrapolated_onset_endset(temp, signal, boundaries, baseline)
        for row, heat, temps in zip(rows, thermal, tangent):
            output = dict(filename=name, peak_id=row["PeakID"], mass_mg=mass,
                          heating_rate_C_min=rate, **heat, **temps)
            outputs.append(output)
    fields = ["filename", "peak_id", "mass_mg", "heating_rate_C_min", "t_start", "t_end", "t_peak",
              "hf_peak", "area_mJ", "deltaH_Jg", "t_onset", "t_endset", "onset_slope_mW_per_C",
              "endset_slope_mW_per_C", "onset_status", "endset_status", "smooth_width_C"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(outputs)
    print("Thermal results: {} ({} peaks)".format(args.output, len(outputs)))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("detect", help="Detect with one trained 120/240-point family model")
    p.add_argument("curves", type=Path, help="Folder of downward-endotherm DSC workbooks in mW")
    p.add_argument("--family", required=True, choices=("narrow", "broad"))
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path, help="Detected candidates TSV")
    p.set_defaults(func=detect)
    p = sub.add_parser("quantify", help="Use saved melting-peak boundaries to calculate thermal parameters")
    p.add_argument("curves", type=Path)
    p.add_argument("--detections", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path, help="Thermal results CSV")
    p.add_argument("--metadata", type=Path, help="CSV: filename,mass_mg,heating_rate_C_min")
    p.add_argument("--mass-mg", type=positive, help="Common mass only when all files have this mass")
    p.add_argument("--heating-rate", type=positive, help="Common heating rate in degrees Celsius/minute")
    p.set_defaults(func=quantify)
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT / "src"))
    try:
        args.func(args)
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(2, "Input error: {}\n".format(exc))

if __name__ == "__main__":
    main()
