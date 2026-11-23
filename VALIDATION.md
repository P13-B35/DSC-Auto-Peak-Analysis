# Local software validation

Date: 2026-11-23.

Environment: Windows, Python 3.8.2, CPU, with the installed versions recorded
in `requirements.txt`.

Completed checks:

- All Python source files parse successfully.
- Command-line help works for training, detection/quantification, FS fitting/
  synthesis, and the artificial example generator.
- The collected DSC package imports from its own `src` directory.
- The artificial example generator creates separate whole-curve training and
  validation inputs for narrow and broad families.
- FS fitting accepts one artificial narrow and one artificial broad curve;
  FS synthesis generates a curve and reference annotations for each family.
- One epoch of narrow real-only training runs with 120-point input.
- One epoch of broad real-plus-FS training runs with 240-point input.
- Both saved example models load and execute detection.
- Thermal calculations run for both artificial families with per-file metadata;
  integral/mass agrees with the exported specific enthalpy within output rounding.
- Selecting a model with the wrong input length is rejected.
- A negative mass argument is rejected.
- Collected original source hashes are unchanged. The two FS source files have
  only their workstation-specific `WORK` path replaced by a script-relative
  path. Portable wrappers are new files and do not replace original projects.
- The release directory contains no measured input workbooks, pretrained
  weights, generated demonstration outputs, PDFs, cache files, or old backups.

These are execution checks using artificial curves. One-epoch models are not
validated scientific detectors and their metrics do not reproduce paper results.
No new clean-environment installation, GPU execution, GitHub upload, Release,
or DOI archive was performed in these checks. After uploading, download the
repository into a separate directory and follow the installation instructions.

The generated test data, model files, and detailed local logs are kept outside
the release directory.
