# Project Log

## 2026-06-13 Row/Col Sinusoidal + Synthetic Orientation

### Changed
- Added `vit_axis_sinusoidal.py` with two explicit variants:
  - `ViTRowSinusoidal`
  - `ViTColSinusoidal`
- Added `synthetic_orientation_data.py` as a controlled horizontal/vertical dataset.
- Registered the new models and dataset routing in `model_registry.py`.
- Extended `train_cifar10_experiment.py` with:
  - `--dataset`
  - synthetic dataset size / noise / stripe controls
- Extended result metadata so config files record the selected dataset and synthetic settings.
- Updated report display names so the new model variants show up cleanly in comparisons.
- Rewrote `README.md` so the current one-runner workflow and new weekly mainline are documented clearly.
- Smoke-verified the unified runner on `synthetic_orientation` for:
  - `vit_baseline`
  - `vit_row_sinusoidal`
  - `vit_col_sinusoidal`

### Learned
- The advisor's current priority is not more seeds, but more directional structure variants under one fixed seed.
- A controlled synthetic dataset is the cleanest first test for the row-wise vs column-wise hypothesis.
- Keeping the new row/col variants as additive positional-embedding branches preserves a clean baseline boundary.
- The new code path writes the expected artifacts correctly:
  - `metrics.csv`
  - `config.json`
  - `summary.json`
  - curves
  - checkpoint

### Problems / Mismatches
- The workspace still contains old deleted/untracked report artifacts under `results/reports/`; they are experiment outputs, not code changes.
- `resnet18_imagenet` remains available, but it is no longer part of the immediate weekly mainline.

### Next
- Run one clean seed on `synthetic_orientation` for:
  - `vit_baseline`
  - `vit_row_sinusoidal`
  - `vit_col_sinusoidal`
- Generate epoch-based comparison plots.
- If the directional signal is visible, start a second-stage real-dataset trial such as CADB.
