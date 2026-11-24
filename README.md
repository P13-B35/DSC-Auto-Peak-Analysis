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

Prepare train and validation sets at whole-curve level. Keep a source curve and
synthetic curves derived from it out of the validation set if it is used for
training. The expected directory structure is:

```text
my_dataset/
  splits/
    narrow/
      real_train/       # .xlsx curves and references.tsv
      real_val/         # .xlsx curves and references.tsv
    broad/
      real_train/
      real_val/
  synthetic_narrow/    # needed only for real_fs
  synthetic_broad/     # needed only for real_fs
```

Use the portable entry point below. It selects the existing 120-point narrow
and 240-point broad training path, with target temperature spans of 2 and 4
degrees Celsius respectively. The source code retains these settings in
`PAPER_WINDOWS`; it also retains historical settings used by earlier entry
points. `train.py` explicitly selects the 120/240-point path.

```text
python train.py --data-root my_dataset --family narrow --variant real_only
python train.py --data-root my_dataset --family broad --variant real_fs
```

`--epochs`, `--patience`, `--seed`, and `--output-dir` are optional. Models,
training histories, and metrics are saved beneath `runs/` by default. Use the
printed model path when running detection. The unmodified lower-level
`train_family_models.py` is included as the implementation of this entry point.

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

The detection command accepts one trained model and its family. The
quantification command consumes saved detections and uses their melting-peak
boundaries. It constructs a baseline and calls the existing enthalpy and
extrapolated-temperature functions. The mass and heating rate must be supplied
explicitly, either as common values or as per-workbook metadata. No pretrained
model is silently substituted. Run the narrow and broad models separately if
both are required; this release wrapper does not introduce a new joint-model
selection rule.

## Artificial demonstration

```text
python examples/make_demo.py --output workspace/demo
python train.py --data-root workspace/demo --family narrow --variant real_only --epochs 1
```

The one-epoch example checks execution. It does not reproduce the paper's
training results and is not a model for scientific use. `make_demo.py` also
writes detection and metadata files for exercising thermal calculations without
model weights. See `README_中文.md` for step-by-step commands.

See `VALIDATION.md` for checks actually executed on this collection and
`SHA256SUMS.txt` for file checksums.

## Citation and publication

`CITATION.cff` contains software citation metadata. Add the actual repository
URL, release version, and article citation after they exist.
