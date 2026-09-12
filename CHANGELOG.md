# Changelog

## [1.0.0] — 2026-09-12

First public release, accompanying submission of the manuscript to *Journal of Intelligent
Manufacturing*.

### Contents

- Full image dataset: 1056 training patches from 6 specimens and 192 test patches from a seventh,
  with their reference profilometry measurements.
- Reference measurements in the original spreadsheet and as UTF-8 CSV, with outlier flags and
  cross-validation fold assignment made explicit.
- The six leave-one-specimen-out folds materialised as data rather than left implicit in code.
- Training, evaluation, interpretability and figure code for the whole pipeline.
- Published model weights: variant B for Ra and variant C for Rz, four seeds each.
- Metrics, per-epoch histories and per-fold predictions of all 42 architecture-sweep runs, so
  every architecture figure regenerates without retraining.
- Stitched panorama and zone crops needed to regenerate the spatial roughness map.
- Figures as published, at 300 dpi and as vector PDF.

### Adaptations made for publication

The code is the code that produced the results, with these changes and no others:

- `data/fotos_recortes` added to the directory layouts the scripts search, so the repository works
  unmodified after cloning. The original layouts remain first in the search order.
- `comun.py` resolves the data, results and sweep directories against either the published layout
  (`data/`, `results/`) or the original working tree, whichever exists.
- `validar_headless.py` defaults to the published weights of whichever parameter is requested,
  instead of a single hard-coded path.
- Absolute paths to the machine of the first author replaced by `<PROJECT_ROOT>` in provenance
  records and logs. The rest of each record — library versions, platform, seeds, command line —
  is untouched.
- Comments referring to private correspondence and to the peer-review process were rewritten to
  state their technical content directly. No verbatim reviewer text is included, as peer-review
  correspondence is confidential.

None of these changes touch a computation. The numbers in `results/` were regenerated from this
repository and match the published values exactly.
