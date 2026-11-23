"""Portable entry point for the existing 120/240-point training pipeline."""
import argparse
from pathlib import Path

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", required=True, type=Path,
                   help="Prepared data in mW, downward endotherms, whole-curve train/validation splits")
    p.add_argument("--family", required=True, choices=("narrow", "broad"))
    p.add_argument("--variant", required=True, choices=("real_only", "real_fs"))
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--seed", type=int, default=20260713)
    p.add_argument("--output-dir", type=Path, default=Path("runs"))
    args = p.parse_args()
    data = args.data_root.resolve()
    for part in ("real_train", "real_val"):
        folder = data / "splits" / args.family / part
        if not folder.is_dir() or not (folder / "references.tsv").is_file():
            p.error("Missing prepared split or references.tsv: " + str(folder))
        if not any(folder.glob("*.xlsx")):
            p.error("No .xlsx curves in " + str(folder))
    if args.variant == "real_fs":
        folder = data / ("synthetic_" + args.family)
        if not (folder / "references.tsv").is_file() or not any(folder.glob("*.xlsx")):
            p.error("Missing synthetic curves or references.tsv: " + str(folder))
    if args.epochs < 1 or args.patience < 1:
        p.error("epochs and patience must be positive")
    # Existing training code remains unchanged. Only the output base is relocated.
    import train_family_models as training
    training.SEED = args.seed
    training.WORK = args.output_dir.resolve()
    training.WORK.mkdir(parents=True, exist_ok=True)
    training.train_one(args.family, args.variant, epochs=args.epochs,
                       patience=args.patience, paper_data=data)

if __name__ == "__main__":
    main()
