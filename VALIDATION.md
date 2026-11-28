# Shared-training revision: executed checks

Date: 2026-11-28. Windows, Python 3.8.2, TensorFlow 2.13.0, NumPy 1.24.3,
SciPy 1.10.1, pandas 2.0.3, CPU. The original program and manuscript were not edited.

## Automated checks

`python -B -m unittest discover -s tests -v`: 21 tests passed.

Coverage includes shared whole-curve identity, renamed-curve leakage, invalid
input/annotations, FS-parent validation, migration, preserving all base samples,
common training centers, broad-peak exclusion from hard negatives, dynamic
windows, raw-probability ensemble averaging, probability/SNR gates, within-scale
suppression, common-minimum merging, one-/two-scale routing, temporal FWHM
units, unresolved boundaries, and rejecting invalid quantitative results.

## Executed integration checks

- Generated explicitly artificial curves: 8 training and 4 validation curves.
- Fitted all 8 artificial training curves; all 8 FS fits accepted. Generated
  4 FS curves, with all recorded parents belonging to artificial training.
- Legacy-folder union and canonical shared-directory migration preserved the
  dataset fingerprint.
- Trained both 120- and 240-point networks for one epoch each on the SAME
  8 training, 4 validation and 4 FS curves. Both runs recorded fingerprint
  `f5e41b1698b8faa11da978286f35c07ccea1a93af37e83322320fadcd94c892a`.
- Both networks saved HDF5 weights, training histories and dataset plans.
- Reloaded both networks through detect-joint; exported TSV plus model hashes
  and configuration provenance. The one-epoch models detected zero peaks;
  the empty-output quantify path completed.
- Used controlled probabilities on two clean ARTIFICIAL Gaussian curves to
  exercise nonempty joint output. Widths of 3 and 18 degrees Celsius selected
  narrow and broad boundary rules, respectively. The retained quantitative
  CLI calculated two finite positive enthalpies. This checks routing and file
  compatibility, NOT learned detector accuracy.
- Unresolved/NaN boundaries were rejected by quantify.
- Pure data preparation and routing imports do not load TensorFlow.
- Windows Unicode paths were tested. HDF5 reload uses h5py; CSV history uses
  Python file I/O. TensorBoard is disabled for unsupported non-ASCII log paths.
  Network topology, loss weights and thermal formulas are unchanged.

The check outputs and one-epoch weights are under workspace/, excluded from
version control. workspace/check_summary.json records software-check counts.
CHANGELOG.md describes the source changes. SHA256SUMS.txt lists the current
repository file checksums. The local revision manifest is not distributed.

## Result scope

No measured curves were used for training/evaluation in this revision. No paper
performance claim was recomputed. The one-epoch artificial weights are software
examples, not models for real DSC analysis. Shared-data models need actual-data
training/evaluation before their accuracy can be reported. The historical width
threshold remains explicit configuration, not a newly calibrated result.

These checks preceded this repository update. No release, fresh-environment
installation or GPU validation was performed.
