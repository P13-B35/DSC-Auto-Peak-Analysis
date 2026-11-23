"""Portable FS fitting/synthesis wrappers; source calculation functions unchanged."""
import argparse
import json
import math
from pathlib import Path

def fit(args):
    import numpy as np
    import pandas as pd
    from fit_all_fs_peaks import fit_one, auxiliary_refs
    manifest_path = args.manifest.resolve()
    manifest = pd.read_csv(manifest_path)
    required = {"curve_id", "derived_curve", "material", "peak_family", "search_min_C",
                "search_max_C", "diagnostic_center_C", "mass_mg", "heating_rate_C_min"}
    missing = required - set(manifest)
    if missing:
        raise ValueError("Missing manifest columns: " + repr(sorted(missing)))
    if manifest.empty or manifest["curve_id"].duplicated().any():
        raise ValueError("Manifest must contain curves with distinct curve_id values")
    out = args.output.resolve()
    internal = out / "upward_templates"
    downward = out / "downward_training"
    internal.mkdir(parents=True, exist_ok=True)
    downward.mkdir(parents=True, exist_ok=True)
    records, refs, failures = [], [], []
    for index, source in manifest.iterrows():
        source = source.copy()
        name = "curve_{:05d}.xlsx".format(index + 1)
        raw = Path(str(source["derived_curve"]))
        raw = raw if raw.is_absolute() else manifest_path.parent / raw
        try:
            data = pd.read_excel(raw)
            t = pd.to_numeric(data["temperature"], errors="raise").to_numpy(float)
            y = pd.to_numeric(data["heatflow"], errors="raise").to_numpy(float)
            if len(t) < 50 or not np.all(np.isfinite(t)) or not np.all(np.isfinite(y)) or not np.all(np.diff(t) > 0):
                raise ValueError("Need >=50 finite, increasing temperature points")
            if source["peak_family"] not in ("narrow_metal", "crystalline_organic", "broad_polymer"):
                raise ValueError("Unknown peak_family")
            if not float(source["search_min_C"]) < float(source["diagnostic_center_C"]) < float(source["search_max_C"]):
                raise ValueError("Center hint must lie within the search interval")
            upward = -y if args.endotherm == "down" else y
            pd.DataFrame({"temperature": t, "heatflow": upward}).to_excel(internal / name, index=False)
            pd.DataFrame({"temperature": t, "heatflow": -upward}).to_excel(downward / name, index=False)
            source["derived_curve"] = str(internal / name)
            parameters = fit_one(source, float(source["diagnostic_center_C"]))
            accepted = bool(parameters["r2"] >= 0.96 and parameters["nrmse_to_amplitude"] <= 0.20)
            records.append(dict(curve_id=source["curve_id"], filename=name,
                derived_curve=str(internal / name), source_file=str(raw.resolve()),
                material=source["material"], peak_family=source["peak_family"],
                mass_mg=source["mass_mg"], heating_rate_C_min=source["heating_rate_C_min"],
                fit_accepted=accepted, signal_unit="mW", template_endotherm="up", **parameters))
            if accepted:
                peak = dict(filename=name, temp_center=parameters["center_C"],
                    temp_start=parameters["t_start_C"], temp_end=parameters["t_end_C"], type="melt_peak")
                refs.append(peak)
                refs.extend(auxiliary_refs(source, peak))
        except Exception as exc:
            failures.append(dict(curve_id=source["curve_id"], error=str(exc)))
    pd.DataFrame(records).to_csv(out / "fs_fit_params.csv", index=False)
    pd.DataFrame(failures, columns=["curve_id", "error"]).to_csv(out / "fs_fit_failures.csv", index=False)
    pd.DataFrame(refs, columns=["filename", "temp_center", "temp_start", "temp_end", "type"]).to_csv(
        downward / "references.tsv", sep="\t", index=False)
    report = dict(input_curves=len(manifest), fitted=len(records), failures=len(failures),
                  accepted=sum(r["fit_accepted"] for r in records), signal_unit="mW",
                  original_endotherm=args.endotherm, output_endotherm="down")
    (out / "fit_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))
    if failures or not report["accepted"]:
        raise ValueError("Some curves failed or no fits were accepted; inspect fs_fit_failures.csv and fs_fit_params.csv")

def generate(args):
    import pandas as pd
    from generate_all_fs_synthetic import generate_family
    params = pd.read_csv(args.params)
    if "signal_unit" not in params or not params["signal_unit"].eq("mW").all():
        raise ValueError("Use parameters produced by fs_workflow.py fit with mW input")
    if not params["template_endotherm"].eq("up").all():
        raise ValueError("Fitting templates must have upward endotherms")
    params = params[params["fit_accepted"].astype(str).str.lower().isin(["true", "1"])].copy()
    out = args.output.resolve()
    for family in ("narrow", "broad"):
        mask = params["peak_family"].eq("broad_polymer")
        group = params[mask if family == "broad" else ~mask].copy()
        if group.empty:
            continue
        target = out / ("synthetic_" + family)
        if target.exists() and any(target.glob("*.xlsx")):
            raise ValueError("Output already contains curves; use a new directory: " + str(target))
        result = generate_family(group, family, sorted(group["material"].unique()),
            args.n_per_material, args.seed + (1000 if family == "broad" else 0), target)
        for workbook in target.glob("*.xlsx"):
            frame = pd.read_excel(workbook)
            frame["heatflow"] = -frame["heatflow"]
            frame.to_excel(workbook, index=False)
        (target / "SYNTHETIC_DATA.txt").write_text(
            "FS-generated training curves, not new measured DSC experiments.\n"
            "Temperature: degrees Celsius; heatflow: mW; endotherms: down.\n", encoding="utf-8")
        print(json.dumps(result))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("fit", help="Fit your own mW curves using retained FS functions")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--endotherm", choices=("up", "down"), default="down")
    p.set_defaults(func=fit)
    p = sub.add_parser("generate", help="Generate downward-mW synthetic training curves")
    p.add_argument("--params", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--n-per-material", type=int, default=24)
    p.add_argument("--seed", type=int, default=20260713)
    p.set_defaults(func=generate)
    args = parser.parse_args()
    if getattr(args, "n_per_material", 1) < 1:
        parser.error("--n-per-material must be positive")
    try:
        args.func(args)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, "Input error: {}\n".format(exc))

if __name__ == "__main__":
    main()
