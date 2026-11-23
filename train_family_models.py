"""Prepare leakage-free splits and train narrow/broad DSC PeakBot models.

The locked 10 degC/min heat-flow directory is deliberately not referenced in
this script.  Model selection therefore uses only non-10 degC/min voltage
curves, split at whole-curve level, with FS curves added to training only.
"""

import argparse
import csv
import json
import math
import os
import pickle
import random
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


WORK = Path(__file__).resolve().parent
SRC = WORK / "src"
sys.path.insert(0, str(SRC))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "4")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")

from peakbot.dsc_peakbot.config import Config, PAPER_WINDOWS
from peakbot.dsc_peakbot.data_loader import DSCDataset
from peakbot.dsc_peakbot.model import DSCPeakBot, data_generator, predict_on_curve
from peakbot.dsc_peakbot.preprocessing import (
    extract_scalar_features,
    extract_window,
    find_local_maxima,
    prepare_instances_for_prediction,
    prepare_instances_from_references,
    save_instances,
    temperature_window_size,
)


SEED = 20260713
FAMILY_CONFIG = {
    "narrow": {
        "manifest_families": {"narrow_metal", "crystalline_organic"},
        "window": 128,
        "synthetic": WORK / "data" / "synthetic_narrow",
        "tolerance_C": 5.0,
        "nms_C": 1.5,
    },
    "broad": {
        "manifest_families": {"broad_polymer"},
        "window": 256,
        "synthetic": WORK / "data" / "synthetic_broad",
        "tolerance_C": 10.0,
        "nms_C": 3.0,
    },
}


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_tsv(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = ["filename", "temp_center", "temp_start", "temp_end", "type"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def prepare_splits():
    manifest = read_csv(WORK / "dataset_manifest.csv")
    refs = read_tsv(WORK / "data" / "development_real" / "references.tsv")
    split_rows = []

    for family, cfg in FAMILY_CONFIG.items():
        rows = [r for r in manifest
                if r["role"] == "development"
                and r["qc_status"] == "usable"
                and r["peak_family"] in cfg["manifest_families"]]

        grouped = defaultdict(list)
        for row in rows:
            grouped[row["material"]].append(row)

        rng = random.Random(SEED + (0 if family == "narrow" else 101))
        val_names = set()
        for material, material_rows in sorted(grouped.items()):
            material_rows = sorted(material_rows, key=lambda r: r["curve_id"])
            rng.shuffle(material_rows)
            n_val = max(1, int(round(0.20 * len(material_rows))))
            n_val = min(n_val, max(1, len(material_rows) - 2))
            val_names.update(Path(r["derived_curve"]).name for r in material_rows[:n_val])

        split_root = WORK / "data" / "splits" / family
        if split_root.exists():
            shutil.rmtree(split_root)
        train_dir = split_root / "real_train"
        val_dir = split_root / "real_val"
        train_dir.mkdir(parents=True)
        val_dir.mkdir(parents=True)

        family_names = set()
        for row in rows:
            source = Path(row["derived_curve"])
            split = "val" if source.name in val_names else "train"
            dest = val_dir / source.name if split == "val" else train_dir / source.name
            shutil.copy2(source, dest)
            family_names.add(source.name)
            split_rows.append({
                "family": family,
                "split": split,
                "material": row["material"],
                "curve_id": row["curve_id"],
                "filename": source.name,
                "heating_rate_C_min": row["heating_rate_C_min"],
                "mass_mg": row["mass_mg"],
            })

        train_refs = [r for r in refs if r["filename"] in family_names
                      and r["filename"] not in val_names]
        val_refs = [r for r in refs if r["filename"] in val_names]
        write_tsv(train_dir / "references.tsv", train_refs)
        write_tsv(val_dir / "references.tsv", val_refs)

    out = WORK / "data" / "splits" / "curve_level_split.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8-sig", newline="") as handle:
        fields = ["family", "split", "material", "curve_id", "filename",
                  "heating_rate_C_min", "mass_mg"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(split_rows)

    summary = Counter((r["family"], r["split"], r["material"]) for r in split_rows)
    print("Curve-level split:")
    for key, count in sorted(summary.items()):
        print("  ", key, count)
    return split_rows


def mine_hard_negative_references(dataset, window, per_curve=16,
                                  family=None, target_span_c=None):
    """Label peak-like extrema outside known melt regions as artifacts.

    Sparse point annotations are insufficient for full-curve deployment: a
    curve can contain hundreds of local extrema but only one manually listed
    artifact.  Candidates are ranked by the same dimensionless local SNR used
    by the network.  The strongest well-separated non-target extrema become
    hard negatives.  No locked 10 degC/min curve is used here.
    """
    refs_by_curve = defaultdict(list)
    for ref in dataset.references:
        refs_by_curve[ref["filename"]].append(ref)

    added = []
    for curve in dataset.curves:
        temp, signal = curve.to_numpy()
        known = refs_by_curve.get(curve.name, [])
        melt_centres = [float(r["temp_center"]) for r in known
                        if r["type"] == "melt_peak"]
        occupied = [float(r["temp_center"]) for r in known]
        is_broad = family == "broad" if family is not None else window >= 256
        melt_exclusion = 12.0 if is_broad else 6.0
        min_sep = 1.5 if is_broad else 0.8
        raw_window = (window if target_span_c is None
                      else temperature_window_size(temp, target_span_c))

        extrema = np.where(
            find_local_maxima(signal, mode="min") |
            find_local_maxima(signal, mode="max"))[0]
        scored = []
        for idx in extrema:
            t = float(temp[idx])
            if t < Config.MIN_TEMPERATURE:
                continue
            if melt_centres and min(abs(t - c) for c in melt_centres) < melt_exclusion:
                continue
            if occupied and min(abs(t - c) for c in occupied) < min_sep:
                continue
            _, win = extract_window(temp, signal, int(idx), raw_window)
            score = float(extract_scalar_features(win)[0])
            scored.append((score, t))

        selected = []
        for score, t in sorted(scored, reverse=True):
            if all(abs(t - s) >= min_sep for s in selected):
                selected.append(t)
                added.append({
                    "filename": curve.name,
                    "temp_center": t,
                    "temp_start": t - min_sep,
                    "temp_end": t + min_sep,
                    "type": "artifact",
                })
            if len(selected) >= per_curve:
                break

    dataset.references.extend(added)
    print(f"  | Mined {len(added)} hard-negative extrema from "
          f"{len(dataset.curves)} curves")
    return added


def load_reference_instances(folder, window, augment_count, seed,
                             hard_negatives_per_curve=16, family=None,
                             target_span_c=None):
    np.random.seed(seed)
    random.seed(seed)
    dataset = DSCDataset()
    dataset.load_folder(str(folder), temp_col=0, hf_col=1)
    # Thermocouple-voltage development curves and FS curves use the opposite
    # polarity to the final heat-flow convention. Align them globally so that
    # melting remains a downward (endothermic) feature in both domains.
    if target_span_c is None:
        for curve in dataset.curves:
            curve.heat_flow = -curve.heat_flow
    dataset.load_references(str(folder / "references.tsv"))
    if hard_negatives_per_curve:
        mine_hard_negative_references(
            dataset, window, per_curve=hard_negatives_per_curve,
            family=family, target_span_c=target_span_c)
    instances = prepare_instances_from_references(
        dataset, window_size=window, augment_count=augment_count, verbose=True,
        target_span_c=target_span_c)
    return dataset, instances


def balance_instances(instances, base_target, seed):
    groups = defaultdict(list)
    for inst in instances:
        groups[int(np.argmax(inst["type"]))].append(inst)
    # Peak and baseline receive equal representation; artifact windows are
    # retained at 3x frequency to reflect full-curve candidate prevalence and
    # penalise false positives more strongly.
    targets = {0: base_target, 1: base_target, 2: 3 * base_target}
    rng = np.random.RandomState(seed)
    balanced = []
    for cls_idx in range(Config.NUM_CLASSES):
        group = groups[cls_idx]
        if not group:
            raise RuntimeError(f"No training instances for class {cls_idx}")
        target = targets[cls_idx]
        replace = len(group) < target
        indices = rng.choice(len(group), size=target, replace=replace)
        balanced.extend(group[int(i)] for i in indices)
    rng.shuffle(balanced)
    return balanced


def save_instance_set(instances, folder):
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    save_instances(instances, str(folder))


def load_all_batches(folder):
    batches = []
    idx = 0
    while True:
        path = folder / f"{Config.INSTANCE_PREFIX}{idx}.pickle"
        if not path.exists():
            break
        with path.open("rb") as handle:
            batches.append(pickle.load(handle))
        idx += 1
    if not batches:
        raise RuntimeError(f"No instance batches in {folder}")
    joined = {key: np.concatenate([b[key] for b in batches], axis=0)
              for key in ("area", "features", "type", "center", "box")}
    x = {"area": joined["area"][..., np.newaxis], "features": joined["features"]}
    y = {"peak_type": joined["type"], "center": joined["center"], "box": joined["box"]}
    return joined, x, y


def classification_metrics(y_true, y_pred):
    true_cls = np.argmax(y_true, axis=1)
    pred_cls = np.argmax(y_pred, axis=1)
    tp = int(np.sum((true_cls == 0) & (pred_cls == 0)))
    fp = int(np.sum((true_cls != 0) & (pred_cls == 0)))
    fn = int(np.sum((true_cls == 0) & (pred_cls != 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    confusion = np.zeros((3, 3), dtype=int)
    for t, p in zip(true_cls, pred_cls):
        confusion[t, p] += 1
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "accuracy": float(np.mean(true_cls == pred_cls)),
        "confusion": confusion.tolist(),
    }


def nms_temperature(peaks, min_separation_C):
    melt = [p for p in peaks if p["type_idx"] == 0]
    melt.sort(key=lambda p: (p.get("confidence", 0.0), p.get("full_snr", 0.0)),
              reverse=True)
    kept = []
    for peak in melt:
        if all(abs(peak["center_temp"] - q["center_temp"]) >= min_separation_C
               for q in kept):
            kept.append(peak)
    return sorted(kept, key=lambda p: p["center_temp"])


def curve_detection_metrics(pb, dataset, tolerance_C, nms_C, target_span_c=None):
    refs_by_curve = defaultdict(list)
    for ref in dataset.references:
        if ref["type"] == "melt_peak":
            refs_by_curve[ref["filename"]].append(float(ref["temp_center"]))

    tp = fp = fn = 0
    errors = []
    detail = []
    for curve in dataset.curves:
        areas, feats, temp_win, gd_props = prepare_instances_for_prediction(
            curve, window_size=Config.WINDOW_SIZE,
            min_temperature=Config.MIN_TEMPERATURE, verbose=False,
            target_span_c=target_span_c)
        if areas is None:
            predicted = []
        else:
            predicted = nms_temperature(
                predict_on_curve(pb, areas, feats, temp_win, gd_props, verbose=False),
                nms_C)
        pred_temps = [float(p["center_temp"]) for p in predicted]
        true_temps = list(refs_by_curve.get(curve.name, []))

        pairs = []
        for pi, pred in enumerate(pred_temps):
            for ti, true in enumerate(true_temps):
                pairs.append((abs(pred - true), pi, ti))
        used_p = set()
        used_t = set()
        for err, pi, ti in sorted(pairs):
            if err > tolerance_C or pi in used_p or ti in used_t:
                continue
            used_p.add(pi)
            used_t.add(ti)
            errors.append(err)
        curve_tp = len(used_p)
        curve_fp = len(pred_temps) - curve_tp
        curve_fn = len(true_temps) - curve_tp
        tp += curve_tp
        fp += curve_fp
        fn += curve_fn
        detail.append({
            "filename": curve.name,
            "true_centers_C": true_temps,
            "predicted_centers_C": pred_temps,
            "tp": curve_tp, "fp": curve_fp, "fn": curve_fn,
        })

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "center_MAE_C": float(np.mean(errors)) if errors else None,
        "matched_center_errors_C": errors,
        "curves": detail,
    }


def train_one(family, variant, epochs=50, patience=8, paper_data=None):
    import tensorflow as tf

    cfg = FAMILY_CONFIG[family].copy()
    target_span_c = None
    data_root = WORK / "data"
    if paper_data is not None:
        cfg["window"], target_span_c = PAPER_WINDOWS[family]
        data_root = Path(paper_data).resolve()
        if data_root == (WORK / "data").resolve():
            raise ValueError("Supply prepared mW data rather than the historical voltage dataset")
        cfg["synthetic"] = data_root / f"synthetic_{family}"
    Config.WINDOW_SIZE = cfg["window"]
    Config.SG_BOUNDARY = 151 if family == "broad" else 31
    Config.EPOCHS = epochs
    Config.CONV_LAYER_SIZES = [32, 64, 128, 256]
    Config.DROPOUT = 0.2

    split_root = data_root / "splits" / family
    real_train_dir = split_root / "real_train"
    real_val_dir = split_root / "real_val"
    if not real_train_dir.exists():
        raise RuntimeError("Run --prepare before training")

    window_options = {"family": family, "target_span_c": target_span_c}
    _, real_train = load_reference_instances(
        real_train_dir, cfg["window"], augment_count=2, seed=SEED, **window_options)
    val_dataset, val_instances = load_reference_instances(
        real_val_dir, cfg["window"], augment_count=0, seed=SEED + 1,
        **window_options)

    train_instances = list(real_train)
    if variant == "real_fs":
        _, synth_instances = load_reference_instances(
            cfg["synthetic"], cfg["window"], augment_count=0, seed=SEED + 2,
            hard_negatives_per_curve=8, **window_options)
        train_instances.extend(synth_instances)
    elif variant != "real_only":
        raise ValueError(variant)

    target = 480 if variant == "real_fs" else 320
    train_instances = balance_instances(train_instances, target, SEED + 3)

    if paper_data is None:
        run_root = WORK / "training_runs" / f"{family}_{variant}"
        run_root.mkdir(parents=True, exist_ok=True)
    else:
        run_parent = WORK / "training_runs_120_240"
        run_parent.mkdir(parents=True, exist_ok=True)
        run_root = Path(tempfile.mkdtemp(
            prefix=f"{family}_{variant}_seed{SEED}_", dir=str(run_parent)))
    train_folder = run_root / "train_instances"
    val_folder = run_root / "val_instances"
    save_instance_set(train_instances, train_folder)
    save_instance_set(val_instances, val_folder)

    _, val_x, val_y = load_all_batches(val_folder)
    Config.STEPS_PER_EPOCH = int(math.ceil(len(train_instances) / Config.BATCH_SIZE))

    np.random.seed(SEED)
    random.seed(SEED)
    tf.random.set_seed(SEED)
    try:
        tf.config.threading.set_intra_op_parallelism_threads(4)
        tf.config.threading.set_inter_op_parallelism_threads(2)
    except RuntimeError:
        pass

    pb = DSCPeakBot()
    pb.build_model(verbose=True)
    pb.compile_model()
    csv_log = run_root / "training_history.tsv"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_peak_type_loss", mode="min", patience=patience,
            min_delta=1e-4, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_peak_type_loss", mode="min", patience=max(3, patience // 2),
            factor=0.6, min_lr=3e-7, verbose=1),
        tf.keras.callbacks.CSVLogger(str(csv_log), separator="\t"),
    ]
    history = pb.train(
        data_generator(str(train_folder)), val_data=(val_x, val_y),
        log_dir=str(run_root / "tensorboard"), callbacks=callbacks, verbose=2)

    model_path = run_root / "model.h5"
    pb.save(str(model_path))

    pred_type, _, _ = pb.predict(val_x["area"], features=val_x["features"])
    instance_metrics = classification_metrics(val_y["peak_type"], pred_type)
    curve_metrics = curve_detection_metrics(
        pb, val_dataset, cfg["tolerance_C"], cfg["nms_C"],
        target_span_c=target_span_c)

    result = {
        "family": family,
        "variant": variant,
        "seed": SEED,
        "window_size": cfg["window"],
        "target_span_c": target_span_c,
        "data_root": str(data_root),
        "epochs_requested": epochs,
        "epochs_run": len(history.history.get("loss", [])),
        "steps_per_epoch": Config.STEPS_PER_EPOCH,
        "n_train_instances_balanced": len(train_instances),
        "train_class_counts": dict(Counter(
            int(np.argmax(i["type"])) for i in train_instances)),
        "n_val_instances": len(val_instances),
        "val_class_counts": dict(Counter(
            int(np.argmax(i["type"])) for i in val_instances)),
        "instance_metrics": instance_metrics,
        "curve_detection_metrics": curve_metrics,
        "model_path": str(model_path),
    }
    (run_root / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main():
    global SEED
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--family", choices=["narrow", "broad"])
    parser.add_argument("--variant", choices=["real_only", "real_fs"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--paper-data", type=Path,
                        help="Prepared whole-curve splits in mW (downward melting peaks)")
    args = parser.parse_args()
    SEED = args.seed
    if args.paper_data is not None and (args.prepare or args.family is None):
        parser.error("--paper-data requires an existing dataset and --family; omit --prepare")

    if args.prepare:
        prepare_splits()
    if args.family or args.variant:
        if not (args.family and args.variant):
            parser.error("--family and --variant must be used together")
        train_one(args.family, args.variant, args.epochs, args.patience,
                  paper_data=args.paper_data)
    if not args.prepare and not args.family:
        parser.error("Choose --prepare and/or a model to train")


if __name__ == "__main__":
    main()
