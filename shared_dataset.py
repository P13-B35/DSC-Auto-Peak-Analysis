"""One validated curve-level dataset for both observation scales; no TensorFlow."""
import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

REFERENCE_FIELDS = ("filename", "temp_center", "temp_start", "temp_end", "type")
CLASS_NAMES = ("melt_peak", "baseline", "artifact")
SCALES = {"narrow": (120, 2.0), "broad": (240, 4.0)}


def _digest_json(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_folder(folder):
    """Reject missing files, invalid annotations and unreadable numeric curves."""
    import numpy as np
    import pandas as pd
    folder = Path(folder).resolve()
    reference_path = folder / "references.tsv"
    files = sorted(p for p in folder.iterdir()
                   if p.is_file() and p.suffix.lower() == ".xlsx" and not p.name.startswith("~")) if folder.is_dir() else []
    if not files or not reference_path.is_file():
        raise ValueError("Expected .xlsx curves and references.tsv in " + str(folder))
    with reference_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != REFERENCE_FIELDS:
            raise ValueError("references.tsv columns must be " + ", ".join(REFERENCE_FIELDS))
        references = []
        for row in reader:
            if not any(row.values()):
                continue
            if row.get("filename", "").startswith("#"):
                continue
            if not row.get("filename"):
                raise ValueError("Reference row has no filename in " + str(reference_path))
            references.append(dict(row))
    known_names = {p.name for p in files}
    grouped = {name: [] for name in known_names}
    for row in references:
        name = row["filename"]
        if name not in known_names:
            raise ValueError("Annotation names missing curve: " + str(folder / name))
        if row["type"] not in CLASS_NAMES:
            raise ValueError("Unknown reference class: " + str(row["type"]))
        try:
            lo, center, hi = (float(row[key]) for key in
                              ("temp_start", "temp_center", "temp_end"))
        except (TypeError, ValueError) as exc:
            raise ValueError("Non-numeric reference for " + name) from exc
        if not np.all(np.isfinite([lo, center, hi])) or not lo <= center <= hi or lo >= hi:
            raise ValueError("Invalid reference interval for " + name)
        grouped[name].append({"filename": name, "temp_center": center,
                              "temp_start": lo, "temp_end": hi, "type": row["type"]})
    records = []
    for path in files:
        frame = pd.read_excel(path)
        if frame.shape[1] < 2:
            raise ValueError("Curve requires two numeric columns: " + str(path))
        try:
            values = frame.iloc[:, :2].to_numpy(dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("Curve contains non-numeric values: " + str(path)) from exc
        if len(values) < 3 or not np.isfinite(values).all():
            raise ValueError("Curve requires at least 3 finite rows: " + str(path))
        if not np.all(np.diff(values[:, 0]) > 0):
            raise ValueError("Temperatures must be strictly increasing: " + str(path))
        refs = grouped[path.name]
        if not refs:
            raise ValueError("Curve has no annotations: " + str(path))
        if any(ref["temp_start"] < values[0, 0] or
               ref["temp_end"] > values[-1, 0] for ref in refs):
            raise ValueError("Reference interval lies outside curve: " + str(path))
        numeric_hash = hashlib.sha256(values.astype("<f8").tobytes()).hexdigest()
        refs = sorted(refs, key=lambda r: (r["temp_center"], r["type"], r["temp_start"]))
        records.append({"filename": path.name, "path": str(path),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "numeric_sha256": numeric_hash, "n_points": len(values),
                        "references": refs})
    return records


def _merge_folders(folders):
    by_name, by_content = {}, {}
    for folder in folders:
        for record in _read_folder(folder):
            name = record["filename"]
            prior = by_name.get(name)
            if prior is not None:
                if (prior["numeric_sha256"] != record["numeric_sha256"] or
                        prior["references"] != record["references"]):
                    raise ValueError("Conflicting duplicate curve name: " + name)
                continue
            content = record["numeric_sha256"]
            if content in by_content:
                raise ValueError("Same curve appears under different names: " +
                                 name + " and " + by_content[content])
            by_name[name] = record
            by_content[content] = name
    return [by_name[name] for name in sorted(by_name)]


def _partition_folders(root, part):
    canonical = root / "splits" / "shared" / part
    if canonical.is_dir():
        return [canonical]
    legacy = [root / "splits" / family / part for family in SCALES]
    if not all(p.is_dir() for p in legacy):
        raise ValueError("Missing shared split: " + str(canonical))
    return legacy


def _attach_synthetic_lineage(records, train, val):
    """Validate recorded FS parents; absent/incomplete ancestry stays unverified."""
    train_ids = {name for r in train for name in (r["filename"], Path(r["filename"]).stem)}
    val_ids = {name for r in val for name in (r["filename"], Path(r["filename"]).stem)}
    cache = {}
    source_fields = ("template_curve_id", "source_curve_id", "source_curve",
                     "source_file", "source_curve_ids", "parent_curve_ids",
                     "secondary_template_curve_id", "secondary_source_curve_id")
    for record in records:
        folder = Path(record["path"]).parent
        if folder not in cache:
            metadata = folder / "synthetic_metadata.csv"
            if metadata.is_file():
                with metadata.open(encoding="utf-8-sig", newline="") as handle:
                    rows = list(csv.DictReader(handle))
                cache[folder] = {row["filename"]: row for row in rows if row.get("filename")}
            else:
                cache[folder] = {}
        row = cache[folder].get(record["filename"], {})
        parents = []
        for field in source_fields:
            value = str(row.get(field) or "").strip()
            if not value:
                continue
            if value.startswith("["):
                try:
                    values = json.loads(value)
                except ValueError as exc:
                    raise ValueError("Invalid FS parent metadata for " + record["filename"]) from exc
            else:
                values = value.split(";")
            if not isinstance(values, list):
                raise ValueError("FS parents must be a list for " + record["filename"])
            parents.extend(str(value).strip() for value in values if str(value).strip())
        parents = sorted(set(parents))
        aliases = {alias for parent in parents for alias in
                   (parent, Path(parent.replace("\\", "/")).name,
                    Path(parent.replace("\\", "/")).stem)}
        if aliases & val_ids:
            raise ValueError("FS source belongs to validation: " + record["filename"])
        unknown = [parent for parent in parents if not {
            parent, Path(parent.replace("\\", "/")).name,
            Path(parent.replace("\\", "/")).stem} & train_ids]
        secondary_present = any(key.startswith("secondary_") and str(value).strip()
                                for key, value in row.items() if value is not None)
        secondary_identified = bool(row.get("secondary_template_curve_id") or
                                    row.get("secondary_source_curve_id"))
        complete = not secondary_present or secondary_identified
        if "lineage_complete" in row:
            complete = str(row["lineage_complete"]).lower() == "true"
        verified = bool(parents) and not unknown and complete
        record["lineage"] = {"status": "verified" if verified else "unverified",
                              "parents": parents, "unknown_parents": unknown,
                              "complete": bool(complete)}


def build_dataset_plan(data_root, variant="real_only"):
    """Use the same full-curve partitions at both scales, without family filters."""
    if variant not in ("real_only", "real_fs"):
        raise ValueError("Unknown training variant: " + str(variant))
    root = Path(data_root).resolve()
    shared_train = root / "splits" / "shared" / "real_train"
    shared_val = root / "splits" / "shared" / "real_val"
    if shared_train.is_dir() != shared_val.is_dir():
        raise ValueError("Both shared real_train and real_val are required")
    train = _merge_folders(_partition_folders(root, "real_train"))
    val = _merge_folders(_partition_folders(root, "real_val"))
    train_names = {r["filename"] for r in train}
    val_names = {r["filename"] for r in val}
    train_hashes = {r["numeric_sha256"] for r in train}
    val_hashes = {r["numeric_sha256"] for r in val}
    if train_names & val_names or train_hashes & val_hashes:
        raise ValueError("A complete curve occurs in both training and validation")
    synth = []
    if variant == "real_fs":
        canonical = root / "synthetic"
        folders = [canonical] if canonical.is_dir() else [
            root / "synthetic_narrow", root / "synthetic_broad"]
        if not all(p.is_dir() for p in folders):
            raise ValueError("real_fs requires synthetic/ or both legacy synthetic folders")
        synth = _merge_folders(folders)
        if ({r["filename"] for r in synth} & (train_names | val_names) or
                {r["numeric_sha256"] for r in synth} & (train_hashes | val_hashes)):
            raise ValueError("Synthetic curves duplicate measured curve names or contents")
        _attach_synthetic_lineage(synth, train, val)
    classes = {ref["type"] for rec in train + synth for ref in rec["references"]}
    if classes != set(CLASS_NAMES):
        raise ValueError("Training annotations must include all three classes")
    plan = {"format": "shared-curves-v1", "data_root": str(root),
            "variant": variant, "heat_flow_unit": "mW", "endotherm": "down",
            "real_train": train, "real_val": val, "synthetic_train": synth,
            "scales": {k: {"window_size": v[0], "target_span_c": v[1]}
                       for k, v in SCALES.items()}}
    identity = {part: [{"filename": r["filename"],
                        "numeric_sha256": r["numeric_sha256"],
                        "references": r["references"]} for r in plan[part]]
                for part in ("real_train", "real_val", "synthetic_train")}
    plan["fingerprint"] = _digest_json(identity)
    plan["synthetic_lineage_status"] = (
        "not_applicable" if not synth else
        "verified" if all(r["lineage"]["status"] == "verified" for r in synth)
        else "unverified")
    return plan


def load_plan_dataset(records):
    """Load the checked records directly, including pooled legacy directories."""
    from peakbot.dsc_peakbot.data_loader import DSCCurve, DSCDataset
    dataset = DSCDataset()
    for record in records:
        if hashlib.sha256(Path(record["path"]).read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError("Curve changed after dataset validation: " + record["path"])
        curve = DSCCurve(record["path"]).load_from_excel(temp_col=0, hf_col=1)
        dataset.curves.append(curve)
        dataset.references.extend(dict(ref) for ref in record["references"])
    return dataset


def migrate_shared_dataset(data_root, output_root, variant="real_only"):
    """Copy a validated union into a NEW shared-layout directory."""
    plan = build_dataset_plan(data_root, variant)
    target = Path(output_root).resolve()
    if target.exists():
        raise ValueError("Migration output must not already exist: " + str(target))
    destinations = {"real_train": target / "splits" / "shared" / "real_train",
                    "real_val": target / "splits" / "shared" / "real_val",
                    "synthetic_train": target / "synthetic"}
    for part, folder in destinations.items():
        records = plan[part]
        if not records:
            continue
        folder.mkdir(parents=True)
        with (folder / "references.tsv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=REFERENCE_FIELDS, delimiter="\t")
            writer.writeheader()
            for record in records:
                shutil.copy2(record["path"], folder / record["filename"])
                writer.writerows(record["references"])
        if part == "synthetic_train":
            with (folder / "synthetic_metadata.csv").open(
                    "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=(
                    "filename", "source_curve_ids", "lineage_complete"))
                writer.writeheader()
                for record in records:
                    writer.writerow({"filename": record["filename"],
                                     "source_curve_ids": json.dumps(record["lineage"]["parents"]),
                                     "lineage_complete": record["lineage"]["complete"]})
    copied = build_dataset_plan(target, variant)
    if copied["fingerprint"] != plan["fingerprint"]:
        raise RuntimeError("Migrated dataset identity differs from input")
    (target / "dataset_plan.json").write_text(
        json.dumps(copied, ensure_ascii=False, indent=2), encoding="utf-8")
    return copied


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--variant", choices=("real_only", "real_fs"), default="real_only")
    parser.add_argument("--output-root", required=True, type=Path,
                        help="New directory for the shared dataset; never overwrites inputs")
    args = parser.parse_args()
    try:
        plan = migrate_shared_dataset(args.data_root, args.output_root, args.variant)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps({"output_root": str(args.output_root.resolve()),
                      "fingerprint": plan["fingerprint"],
                      "curves": {k: len(plan[k]) for k in
                                 ("real_train", "real_val", "synthetic_train")}}, indent=2))


if __name__ == "__main__":
    main()
