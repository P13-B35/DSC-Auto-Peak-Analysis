"""Main entry point for DSC PeakBot — training and peak detection."""

import os
import sys
import argparse

# TF logging control before importing tensorflow
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

from .config import Config, PAPER_WINDOWS
from .core import tic, toc, timeit, writeTSVFile
from .data_loader import DSCDataset
from .preprocessing import (
    prepare_instances_from_references,
    prepare_instances_for_prediction,
    save_instances,
)
from .model import DSCPeakBot, train_model, predict_on_curve


# ---------------------------------------------------------------------------
# Training pipeline
# ---------------------------------------------------------------------------

@timeit
def run_training(curve_folder, reference_file, model_output_path,
                  temp_col=0, hf_col=1,
                  augment_count=5,
                  val_split=0.2,
                  verbose=True):
    """Full training pipeline.

    1. Load DSC curves from *curve_folder*.
    2. Load reference annotations from *reference_file*.
    3. Generate training instances (with augmentation).
    4. Train the 1D CNN model.
    5. Save the model to *model_output_path*.

    Args:
        curve_folder: Folder containing Excel (.xlsx/.xls) files.
        reference_file: TSV file with reference peak annotations.
        model_output_path: Path to save the trained model (.h5).
        temp_col: Temperature column index or name.
        hf_col: Heat flow column index or name.
        augment_count: Number of augmented copies per reference instance.
        val_split: Fraction of instances to use for validation.
        verbose: Print progress.
    """
    if verbose:
        print("=" * 60)
        print("  DSC PeakBot — Training")
        print("=" * 60)
        print(Config.get_as_string())
        print(f"  | Curve folder:     {curve_folder}")
        print(f"  | Reference file:   {reference_file}")
        print(f"  | Model output:     {model_output_path}")
        print(f"  | Augment count:    {augment_count}")
        print(f"  | Validation split: {val_split}")
        print()

    # 1. Load data
    if verbose:
        print("Loading data …")
    dataset = DSCDataset()
    dataset.load_folder(curve_folder, temp_col=temp_col, hf_col=hf_col)
    if verbose:
        print(f"  | Loaded {len(dataset.curves)} curves")

    dataset.load_references(reference_file)
    if verbose:
        print(f"  | Loaded {len(dataset.references)} references")
        print()

    # 2. Generate training instances
    if verbose:
        print("Generating training instances …")
    instances = prepare_instances_from_references(
        dataset, augment_count=augment_count, verbose=verbose)
    print()

    # 3. Train/val split
    n = len(instances)
    n_val = max(1, int(n * val_split))
    indices = list(range(n))
    import random
    random.shuffle(indices)
    val_indices = set(indices[:n_val])
    train_instances = [inst for i, inst in enumerate(instances) if i not in val_indices]
    val_instances   = [inst for i, inst in enumerate(instances) if i in val_indices]

    # 4. Save instances as pickle batches
    train_folder = os.path.join(os.path.dirname(model_output_path),
                                 'train_instances')
    val_folder   = os.path.join(os.path.dirname(model_output_path),
                                 'val_instances')
    save_instances(train_instances, train_folder)
    save_instances(val_instances, val_folder)
    if verbose:
        print(f"  | Saved {len(train_instances)} training, "
              f"{len(val_instances)} validation instances")
        print()

    # 5. Train model
    if verbose:
        print("Training CNN model …")
    pb = train_model(train_folder, model_output_path,
                      val_instances_folder=val_folder,
                      verbose=verbose)
    print()
    if verbose:
        print("Training complete.")
        print(f"  | Model saved to '{model_output_path}'")

    return pb


# ---------------------------------------------------------------------------
# Detection / prediction pipeline
# ---------------------------------------------------------------------------

@timeit
def run_detection(curve_folder, model_path, output_path,
                   temp_col=0, hf_col=1,
                   min_intensity=None,
                   verbose=True, paper_family=None):
    """Detect peaks in DSC curves using a trained model.

    1. Load trained model.
    2. For each DSC curve in *curve_folder*:
       a. Find local maxima.
       b. Extract normalised windows.
       c. Run CNN prediction.
       d. Collect detected melt peaks.
    3. Export results to TSV.

    Args:
        curve_folder: Folder containing Excel files to process.
        model_path: Path to a trained model (.h5).
        paper_family: Use the specified paper window with mW input.
        output_path: Path for the output TSV file.
        temp_col: Temperature column index or name.
        hf_col: Heat flow column index or name.
        min_intensity: Optional minimum heat flow for local maxima.
        verbose: Print progress.
    """
    if verbose:
        print("=" * 60)
        print("  DSC PeakBot — Peak Detection")
        print("=" * 60)
        print(f"  | Curve folder: {curve_folder}")
        print(f"  | Model path:   {model_path}")
        print(f"  | Output:       {output_path}")
        print()

    # 1. Load model
    if verbose:
        print("Loading model …")
    pb = DSCPeakBot()
    pb.load(model_path)
    window_size = None
    target_span_c = None
    if paper_family is not None:
        window_size, target_span_c = PAPER_WINDOWS[paper_family]
    elif pb.model.input_shape[0][1] in {window[0] for window in PAPER_WINDOWS.values()}:
        raise ValueError("Select --paper-family when using a 120/240-point model")
    if paper_family is not None and pb.model.input_shape[0][1] != window_size:
        raise ValueError(f"The selected {paper_family} model must accept {window_size} points")
    if verbose:
        print(f"  | Model '{pb.name}' loaded")
        print()

    # 2. Process each curve
    dataset = DSCDataset()
    dataset.load_folder(curve_folder, temp_col=temp_col, hf_col=hf_col)
    if verbose:
        print(f"Processing {len(dataset.curves)} curves …")

    all_peaks = []
    for curve in dataset.curves:
        if verbose:
            print(f"  | '{curve.name}' ({curve.num_points} points) … ", end='')

        areas, features, temp_win, gd_props = prepare_instances_for_prediction(
            curve, min_intensity=min_intensity, verbose=False,
            window_size=window_size, target_span_c=target_span_c)

        if areas is None:
            if verbose:
                print("no local maxima found")
            continue

        peaks = predict_on_curve(pb, areas, features, temp_win, gd_props,
                                  verbose=False)
        for p in peaks:
            p['filename'] = curve.name
        all_peaks.extend(peaks)

        if verbose:
            n_melt = sum(1 for p in peaks
                         if p['type_idx'] < Config.IS_PEAK_CLASSES)
            print(f"{n_melt} melt peaks detected")

    if verbose:
        print()
        total_melt = sum(1 for p in all_peaks
                          if p['type_idx'] < Config.IS_PEAK_CLASSES)
        print(f"  | Total melt peaks detected: {total_melt}")

    # 3. Export
    export_results(all_peaks, output_path)
    if verbose:
        print(f"  | Results exported to '{output_path}'")

    return all_peaks


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_results(peaks, output_path):
    """Export detected peaks to a TSV file.

    Args:
        peaks: List of peak dicts (from predict_on_curve).
        output_path: Path to output TSV.
    """
    headers = ['PeakID', 'Filename', 'CenterTemp', 'CenterHF',
               'TempStart', 'TempEnd', 'PeakType']
    rows = []
    for i, p in enumerate(peaks):
        rows.append([
            i,
            p.get('filename', ''),
            f"{p['center_temp']:.3f}",
            f"{p['center_hf']:.4f}",
            f"{p['temp_start']:.3f}",
            f"{p['temp_end']:.3f}",
            p['type_name'],
        ])
    writeTSVFile(output_path, headers, rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='DSC PeakBot — ML-based DSC melt-peak detection')

    sub = parser.add_subparsers(dest='command', help='Commands')

    # --- train ---
    p_train = sub.add_parser('train', help='Train a new model')
    p_train.add_argument('curve_folder', help='Folder with DSC Excel files')
    p_train.add_argument('reference_file', help='TSV file with peak annotations')
    p_train.add_argument('model_output', help='Path to save the trained model (.h5)')
    p_train.add_argument('--temp-col', default=0,
                          help='Temperature column (index or name)')
    p_train.add_argument('--hf-col', default=1,
                          help='Heat flow column (index or name)')
    p_train.add_argument('--augment', type=int, default=5,
                          help='Number of augmented copies per reference')
    p_train.add_argument('--val-split', type=float, default=0.2,
                          help='Validation fraction')
    p_train.add_argument('--quiet', action='store_true',
                          help='Suppress verbose output')

    # --- detect ---
    p_detect = sub.add_parser('detect', help='Detect peaks using a trained model')
    p_detect.add_argument('curve_folder', help='Folder with DSC Excel files')
    p_detect.add_argument('model_path', help='Path to the trained model (.h5)')
    p_detect.add_argument('output', help='Path for the output TSV file')
    p_detect.add_argument('--temp-col', default=0,
                           help='Temperature column (index or name)')
    p_detect.add_argument('--hf-col', default=1,
                           help='Heat flow column (index or name)')
    p_detect.add_argument('--min-intensity', type=float, default=None,
                           help='Minimum intensity for local maxima')
    p_detect.add_argument('--paper-family', choices=tuple(PAPER_WINDOWS),
                          help='Use the paper temperature window with heat flow in mW')
    p_detect.add_argument('--quiet', action='store_true',
                           help='Suppress verbose output')

    args = parser.parse_args()
    verbose = not getattr(args, 'quiet', False)

    if args.command == 'train':
        # Handle temp-col / hf-col: try int conversion, fall back to string
        tc = int(args.temp_col) if args.temp_col.isdigit() else args.temp_col
        hc = int(args.hf_col) if args.hf_col.isdigit() else args.hf_col
        run_training(args.curve_folder, args.reference_file,
                      args.model_output,
                      temp_col=tc, hf_col=hc,
                      augment_count=args.augment,
                      val_split=args.val_split,
                      verbose=verbose)
    elif args.command == 'detect':
        tc = int(args.temp_col) if str(args.temp_col).isdigit() else args.temp_col
        hc = int(args.hf_col) if str(args.hf_col).isdigit() else args.hf_col
        run_detection(args.curve_folder, args.model_path,
                       args.output,
                       temp_col=tc, hf_col=hc,
                       min_intensity=args.min_intensity,
                       verbose=verbose, paper_family=args.paper_family)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
