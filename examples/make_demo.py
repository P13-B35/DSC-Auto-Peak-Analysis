"""Generate explicitly artificial DSC inputs for software checks only."""
import argparse
import csv
import json
import shutil
from pathlib import Path

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("workspace/demo"))
    args = p.parse_args()
    import numpy as np
    import pandas as pd
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260713)
    manifest = []
    for family in ("narrow", "broad"):
        if family == "narrow":
            t = np.linspace(90., 190., 2001)
            center, width, amplitude = 150., 3., 3.
        else:
            t = np.linspace(150., 290., 2801)
            center, width, amplitude = 225., 17., 2.5
        for split, count in (("real_train", 4), ("real_val", 2)):
            folder = root / "splits" / family / split
            folder.mkdir(parents=True, exist_ok=True)
            rows = []
            for i in range(count):
                c = center + rng.uniform(-0.7, 0.7)
                w = width * rng.uniform(0.93, 1.07)
                a = amplitude * rng.uniform(0.92, 1.08)
                baseline = 0.05 + 0.0003 * (t - center)
                signal = baseline - a * np.exp(-4. * np.log(2.) * ((t - c) / w) ** 2)
                artifact_c = center - (27. if family == "narrow" else 48.)
                signal += 0.10 * np.exp(-4. * np.log(2.) * ((t - artifact_c) / (0.8*w))**2)
                signal += rng.normal(0., 0.003, len(t))
                name = "ARTIFICIAL_{}_{}_{}.xlsx".format(family, split, i + 1)
                pd.DataFrame({"temperature": t, "heatflow": signal}).to_excel(folder / name, index=False)
                def ref(tc, lo, hi, kind):
                    rows.append(dict(filename=name, temp_center=tc, temp_start=lo, temp_end=hi, type=kind))
                ref(c, c - 1.6*w, c + 1.6*w, "melt_peak")
                ref(float(t[100]), float(t[50]), float(t[150]), "baseline")
                ref(float(t[-101]), float(t[-151]), float(t[-51]), "baseline")
                ref(artifact_c, artifact_c-w, artifact_c+w, "artifact")
                if split == "real_train":
                    manifest.append(dict(curve_id=name[:-5], derived_curve=(folder / name).relative_to(root).as_posix(),
                        material="ARTIFICIAL_" + family,
                        peak_family="narrow_metal" if family == "narrow" else "broad_polymer",
                        search_min_C=c-4*w, search_max_C=c+4*w,
                        diagnostic_center_C=c, mass_mg=10., heating_rate_C_min=10.))
            pd.DataFrame(rows).to_csv(folder / "references.tsv", sep="\t", index=False)
        source = root / "splits" / family / "real_val"
        refs = pd.read_csv(source / "references.tsv", sep="\t")
        melts = refs[refs["type"] == "melt_peak"]
        pd.DataFrame([dict(PeakID=j, Filename=r.filename, CenterTemp=r.temp_center,
            CenterHF=0., TempStart=r.temp_start, TempEnd=r.temp_end, PeakType="melt_peak")
            for j,r in enumerate(melts.itertuples(index=False))]).to_csv(
                root / ("ARTIFICIAL_" + family + "_detections.tsv"), sep="\t", index=False)
        pd.DataFrame([dict(filename=name, mass_mg=10., heating_rate_C_min=10.)
            for name in melts["filename"]]).to_csv(root / (family + "_metadata.csv"), index=False)
    pd.DataFrame(manifest).to_csv(root / "ARTIFICIAL_fs_manifest.csv", index=False)
    # Preserve the original demonstration folders for legacy CLI examples,
    # and expose their complete union to BOTH observation scales.
    for split in ("real_train", "real_val"):
        shared = root / "splits" / "shared" / split
        shared.mkdir(parents=True, exist_ok=True)
        refs = []
        for family in ("narrow", "broad"):
            source = root / "splits" / family / split
            for path in source.glob("*.xlsx"):
                shutil.copy2(path, shared / path.name)
            refs.append(pd.read_csv(source / "references.tsv", sep="\t"))
        pd.concat(refs, ignore_index=True).to_csv(
            shared / "references.tsv", sep="\t", index=False)
    (root / "ARTIFICIAL_DATA_README.txt").write_text(
        "All curves and metadata in this directory are artificial software examples.\n"
        "They are not measured data, experimental results, or validation evidence for a trained detector.\n",
        encoding="utf-8")
    print(str(root))

if __name__ == "__main__":
    main()
