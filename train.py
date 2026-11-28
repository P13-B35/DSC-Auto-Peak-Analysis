"""Train two observation scales on the same complete-curve dataset."""
import argparse
import json
from pathlib import Path

from shared_dataset import build_dataset_plan

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", required=True, type=Path,
                   help="Prepared data in mW, downward endotherms, whole-curve train/validation splits")
    p.add_argument("--family", choices=("both", "narrow", "broad"), default="both",
                   help="Observation scale, not a training-curve family filter")
    p.add_argument("--variant", required=True, choices=("real_only", "real_fs"))
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--seed", type=int, default=20260713)
    p.add_argument("--seeds", type=int, nargs="+",
                   help="Train each requested scale for every listed seed on the fixed split")
    p.add_argument("--output-dir", type=Path, default=Path("runs"))
    p.add_argument("--prepare-only", action="store_true",
                   help="Validate and export the common dataset plan without TensorFlow")
    args = p.parse_args()
    data = args.data_root.resolve()
    if args.epochs < 1 or args.patience < 1:
        p.error("epochs and patience must be positive")
    seeds = args.seeds if args.seeds is not None else [args.seed]
    if len(set(seeds)) != len(seeds):
        p.error("--seeds must not contain duplicates")
    try:
        plan = build_dataset_plan(data, args.variant)
    except (ValueError, OSError) as exc:
        p.error(str(exc))
    from tempfile import mkdtemp
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    session = Path(mkdtemp(prefix="shared_training_", dir=str(output)))
    (session / "dataset_plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Shared dataset fingerprint:", plan["fingerprint"])
    print("Curves: train={}, validation={}, FS={}".format(
        len(plan["real_train"]), len(plan["real_val"]), len(plan["synthetic_train"])))
    print("Session:", session)
    if args.prepare_only:
        return
    import train_family_models as training
    training.WORK = session
    scales = ("narrow", "broad") if args.family == "both" else (args.family,)
    results = []
    for seed in seeds:
        training.SEED = seed
        for scale in scales:
            results.append(training.train_one(
                scale, args.variant, epochs=args.epochs, patience=args.patience,
                paper_data=data, dataset_plan=plan))
    (session / "training_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
