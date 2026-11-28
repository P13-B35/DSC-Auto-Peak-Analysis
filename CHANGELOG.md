# Changes

## 2026-11-28

- Train both observation scales on one whole-curve training/validation split
  and all corresponding FS training curves. Remove curve-family filtering from
  the new training entry point.
- Keep the 2/4-degree-Celsius windows and 120/240-point network inputs. The
  classes remain melting peak, baseline and artifact.
- Use common training sample positions, preserve all base samples when
  balancing classes, and exclude complete annotated melting intervals from
  hard-negative proposals.
- Add shared-dataset validation, content fingerprints, legacy-folder migration
  and recorded FS parent checks. Record secondary FS parents in new synthetic
  metadata without altering generated signal values.
- Add joint inference: average three-class probabilities within each scale,
  filter and suppress duplicates, merge at common smoothed minima, and select
  boundaries by accepted scales and temporal FWHM.
- Export routing status and model/configuration provenance alongside compatible
  detection TSV files. Retain the existing thermal calculations.
- Handle Windows Unicode paths for training histories and HDF5 model reloads.
- Include source-level tests and document the artificial execution checks.

The update contains source, templates, documentation and test definitions.
Generated data, temporary models and local execution records are not included.
It does not replace the manuscript's results with new measured-data evaluations.

Two retained implementation behaviors were identified and left unchanged:
quantitative baseline fitting excludes samples below 80 degrees Celsius, and
boundary search can stop at a search limit before satisfying the threshold.
Five-seed ensembles must be requested explicitly. Incomplete FS provenance
is recorded as unverified rather than enforced as a training error.
