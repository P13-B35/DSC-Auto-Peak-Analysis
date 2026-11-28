"""Command-line wrappers around the retained DSC detection/thermal functions."""
import argparse
import csv
import hashlib
import json
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


def probability(value):
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return number


def nonnegative(value):
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("must be finite and nonnegative")
    return number


def detect_joint(args):
    from peakbot.dsc_peakbot.data_loader import DSCDataset
    from peakbot.dsc_peakbot.joint_detection import (
        detect_curve_joint, export_joint_results, load_model_ensemble,
    )
    folder = args.curves.resolve()
    if not folder.is_dir():
        raise ValueError("Input curve directory is missing")
    expected = {p.name for p in folder.iterdir()
                if p.suffix.lower() in (".xlsx", ".xls") and not p.name.startswith("~")}
    if not expected:
        raise ValueError("Input directory contains no DSC workbooks")
    metadata = None
    if args.metadata:
        metadata = {}
        for row in load_rows(args.metadata):
            name = row["filename"]
            if name in metadata:
                raise ValueError("Duplicate metadata filename: " + name)
            metadata[name] = positive(row["heating_rate_C_min"])
        missing = expected - metadata.keys()
        if missing:
            raise ValueError("Missing heating rate metadata: " + repr(sorted(missing)))
    dataset = DSCDataset().load_folder(str(folder))
    loaded = {curve.name for curve in dataset.curves}
    if loaded != expected:
        raise ValueError("Some workbooks could not be read: " + repr(sorted(expected - loaded)))
    for curve in dataset.curves:
        check_curve(curve)
    # Do not import the model/TensorFlow layer until inputs pass preflight.
    narrow = load_model_ensemble(args.narrow_models, "narrow")
    broad = load_model_ensemble(args.broad_models, "broad")
    settings = dict(width_threshold_min=args.width_threshold_min,
                    narrow_min_probability=args.narrow_min_probability,
                    broad_min_probability=args.broad_min_probability,
                    min_peak_snr=args.min_peak_snr,
                    narrow_nms_C=args.narrow_nms_C, broad_nms_C=args.broad_nms_C,
                    min_temperature=args.min_temperature,
                    broad_physics_check=not args.disable_broad_physics_check)
    peaks = []
    for curve in dataset.curves:
        rate = metadata[curve.name] if metadata is not None else args.heating_rate
        peaks.extend(detect_curve_joint(curve, narrow, broad, rate, **settings))
    export_joint_results(peaks, args.output)
    def model_identity(path):
        path = path.resolve()
        return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    provenance = dict(
        settings=settings, windows=dict(narrow=dict(points=120, span_C=2.0),
                                        broad=dict(points=240, span_C=4.0)),
        models=dict(narrow=[model_identity(p) for p in args.narrow_models],
                    broad=[model_identity(p) for p in args.broad_models]),
        heating_rate_C_min=(metadata if metadata is not None else args.heating_rate),
        input_directory=str(folder), input_workbooks=sorted(loaded),
        output=str(args.output.resolve()), peaks=len(peaks),
        unresolved_boundaries=sum(row["boundary_status"] != "ok" for row in peaks),
    )
    provenance_path = args.output.with_suffix(args.output.suffix + ".provenance.json")
    provenance_path.write_text(json.dumps(provenance, indent=2, ensure_ascii=False,
                                          allow_nan=False), encoding="utf-8")
    print("Joint detections: {} ({} peaks; {} unresolved boundaries)".format(
        args.output, len(peaks), provenance["unresolved_boundaries"]))

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
        unresolved = [row for row in rows
                      if row.get("BoundaryStatus") and row["BoundaryStatus"] != "ok"]
        if unresolved:
            raise ValueError("Unresolved joint boundary in {} at peak {}: {}".format(
                name, unresolved[0]["PeakID"], unresolved[0]["BoundaryStatus"]))
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
    p = sub.add_parser("detect-joint", help="Apply both scales to the same complete curves")
    p.add_argument("curves", type=Path, help="Folder of endothermic-down workbooks in mW")
    p.add_argument("--narrow-models", required=True, type=Path, nargs="+")
    p.add_argument("--broad-models", required=True, type=Path, nargs="+")
    p.add_argument("--output", required=True, type=Path, help="Joint detections TSV")
    rate = p.add_mutually_exclusive_group(required=True)
    rate.add_argument("--heating-rate", type=positive,
                      help="Common heating rate in C/min, if identical for every curve")
    rate.add_argument("--metadata", type=Path,
                      help="CSV with filename,heating_rate_C_min for every input curve")
    p.add_argument("--width-threshold-min", required=True, type=positive,
                   help="Explicit temporal FWHM boundary-routing threshold in minutes")
    p.add_argument("--narrow-min-probability", type=probability, default=0.5)
    p.add_argument("--broad-min-probability", type=probability, default=0.5)
    p.add_argument("--min-peak-snr", type=nonnegative, default=4.0)
    p.add_argument("--narrow-nms-C", type=nonnegative, default=1.5)
    p.add_argument("--broad-nms-C", type=nonnegative, default=3.0)
    p.add_argument("--min-temperature", type=float, default=None,
                   help="Optional acquisition preparation cutoff; default uses complete curve")
    p.add_argument("--disable-broad-physics-check", action="store_true",
                   help="Disable the two-sided background recovery check (diagnostic use)")
    p.set_defaults(func=detect_joint)
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
