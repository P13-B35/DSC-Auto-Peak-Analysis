# DSC Auto Peak Analysis

DSC Auto Peak Analysis contains a one-dimensional CNN for identifying melting-peak
candidates in differential scanning calorimetry curves, along with preprocessing,
FS fitting, synthetic training-curve generation, baseline construction, and
thermal-parameter calculation.

This repository collects the DSC-specific source code. It contains **no measured
research data and no pretrained model weights**. Train models using your own
annotated curves before applying the detector. The demonstration generator
creates artificial curves for checking software operation, not experimental
results.

## Install

The dependency versions in `requirements.txt` match the local validation
environment: Python 3.8.2, TensorFlow 2.13.0, CPU. Use a separate environment.

```text
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux/macOS:
# source .venv/bin/activate
python -m pip install -r requirements.txt
```

Use `.xlsx` input. Reading old `.xls` workbooks additionally requires
`python -m pip install -r requirements-xls.txt`. GPU installation and execution
on other platforms are not part of the local validation.

## Input

One `.xlsx` workbook represents one curve. Its first sheet must have a header
row and two numeric columns, in this order:

| Column | Meaning | Unit/convention |
|---|---|---|
| `temperature` | Temperature | degrees Celsius, strictly increasing |
| `heatflow` | Heat-flow rate | mW; melting endotherms point downward |

This is heat flow, **not** mass-normalized W/g and not detector voltage.
For enthalpy calculation, provide the actual mass in mg and heating rate in
degrees Celsius per minute. The program does not infer these values from a file
name. With different masses or rates, use the per-file metadata CSV.

Training annotations are tab-separated text with this column order:

```text
filename    temp_center    temp_start    temp_end    type
```

`filename` must match the workbook's name. Temperatures are in degrees Celsius;
`type` is `melt_peak`, `baseline`, or `artifact`. See the header-only templates
in `templates/`. Annotation temperatures must describe the supplied curves.

## Train

Prepare train and validation sets once at whole-curve level. Both observation
scales use every curve and every three-class annotation in the same split.
Keep synthetic descendants of validation curves out of training. The layout is:

```text
my_dataset/
  splits/
    shared/
      real_train/       # .xlsx curves and references.tsv
      real_val/         # .xlsx curves and references.tsv
  synthetic/           # all FS curves, needed only for real_fs
```

The narrow and broad names identify the two observation scales, not training
curve categories. Their target spans are 2 and 4 degrees Celsius, resampled to
120 and 240 points. Both models learn `melt_peak`, `baseline`, and `artifact`
from the same curves. Raw window lengths adapt to the temperature sampling.

```text
python train.py --data-root my_dataset --family both --variant real_only
python train.py --data-root my_dataset --family both --variant real_fs
```

`--epochs`, `--patience`, `--seed`, and `--output-dir` are optional. Models,
training histories, metrics, and dataset fingerprints are saved beneath `runs/`.
Use `--prepare-only` to check and export the shared dataset plan without training.
Use `--seeds 20260713 20260714 20260715 20260716 20260717` for the five-seed
ensemble described in the manuscript. A seed changes
training randomness, never the fixed whole-curve split. Each run records its
actual curve membership. See `shared_dataset.py --help` to pool legacy folders
into a new shared dataset without overwriting the original data.

## FS fitting and synthesis

Use `fs_workflow.py --help`, followed by `fit --help` or `generate --help`.
The wrapper calls the retained `fit_one`, `auxiliary_refs`, and `generate_family`
functions. It prepares upward-pointing internal copies for the existing fitting
routine and exports downward mW curves for the training entry point. Original
input workbooks are not changed.

The manifest template is `templates/fs_manifest.csv`. Paths in `derived_curve`
are relative to that manifest or absolute paths to your own files. The
`diagnostic_center_C` column supplies an initial center estimate;
`search_min_C` and `search_max_C` define the fitting interval. Accepted
`peak_family` values are `narrow_metal`, `crystalline_organic`, and
`broad_polymer`, matching the retained fitting policies. These are configuration
categories; the program does not establish a universal material-to-width rule.

Some retained internal parameter column names end in `_uV`. They are historical
field names in the original fitting scripts. Their numerical amplitude units
follow the input signal. The wrapper records mW explicitly and does not convert
detector voltage into heat flow. Existing fit bounds and synthesis rules remain
in the source; inspect them when using temperatures or signal scales outside
the original operating range.

## Detect peaks and calculate thermal parameters

```text
python dsc_peakbot_cli.py --help
python dsc_peakbot_cli.py detect --help
python dsc_peakbot_cli.py quantify --help
```

The retained `detect` command runs one model. Use `detect-joint` for the new
combined pipeline:

```text
python dsc_peakbot_cli.py detect-joint curves --narrow-models small_seed1.h5 small_seed2.h5 --broad-models large_seed1.h5 large_seed2.h5 --heating-rate 10 --width-threshold-min 0.4602999429067669 --output detections.tsv
python dsc_peakbot_cli.py quantify curves --detections detections.tsv --metadata metadata.csv --output thermal.csv
```

Both scales evaluate the same extrema. Probabilities are averaged over seeds
within each scale before classification and SNR filtering. Retained candidates
are combined once; a candidate retained by one scale uses that boundary rule.
For candidates retained by both, a common quadratic background estimates the
temperature FWHM. Dividing by the heating rate gives minutes; the narrow rule
is selected at or below the supplied threshold, and the broad rule above it.
Unresolved widths/boundaries carry an explicit status and cannot be silently
used as valid thermal results. The CNN regression outputs remain auxiliary;
integration boundaries come from the full curve.

The numerical threshold in the example is a historical development value,
not a calibration of the new shared-training models. The command requires
an explicit threshold. Probability, SNR and candidate-spacing settings are
exposed in `detect-joint --help`. Supply the actual heating rate for each curve
using `--metadata` when rates differ. This revision changes the algorithm;
the paper's existing numerical results are not new evaluations of this code.

## Artificial demonstration

```text
python examples/make_demo.py --output workspace/demo
python train.py --data-root workspace/demo --family both --variant real_only --epochs 1
```

The one-epoch example checks execution at both scales. It does not reproduce the paper's
training results and is not a model for scientific use. `make_demo.py` also
writes detection and metadata files for exercising thermal calculations without
model weights. See `README_中文.md` for step-by-step commands.

See `VALIDATION.md` for checks actually executed on this collection and
`SHA256SUMS.txt` for file checksums.

## Current implementation notes

This update implements shared-curve training and joint two-scale detection.
The default training command runs one seed; supply five seeds explicitly for
five models per scale. Missing or incomplete FS parent metadata is recorded as
unverified and does not block training. Two retained behaviors are unchanged:
quantification excludes baseline samples below 80 degrees Celsius, and boundary
search can return a search limit without confirming a threshold crossing.
See `CHANGELOG.md` for the scope of this update.

## Citation and publication

`CITATION.cff` contains software citation metadata and the repository URL.
Article citation and release metadata can be added when available.
