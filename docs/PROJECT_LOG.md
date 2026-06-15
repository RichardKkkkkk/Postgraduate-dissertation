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

## 2026-06-14 CADB Orientation Loader

### Changed
- Added `cadb_data.py`.
- Added a new dataset branch: `cadb_orientation`.
- Registered CADB routing in `model_registry.py` for:
  - `vit_*`
  - `resnet18_scratch`
  - `resnet18_imagenet`
- Extended the unified runner with:
  - `--cadb-root`
  - `--cadb-test-ratio`
  - `--cadb-label-mode`
- Added CADB-related metadata into saved config JSON.
- Smoke-tested the CADB path with a local fixture dataset.

### Learned
- CADB does not come with the same ready-made split pattern as CIFAR-10, so the loader needs its own deterministic split logic.
- For this week's question, the cleanest first real-data task is not full CADB, but a binary `horizontal vs vertical` subset.
- Keeping this as a dataset-layer addition avoids touching the main training loop.
- After checking the real archive, the direction labels are better sourced from `composition_elements.json`, and the official split should come from `split.json`.

### Problems / Mismatches
- The CADB loader currently assumes the first real task is `horizontal` vs `vertical`, not the full multi-label composition problem.
- The real CADB archive still needs to be downloaded and unpacked locally by the user; the code path is ready, but the dataset itself is not bundled in the repo.

### Next
- Download/unzip the real CADB dataset under `data/CADB_Dataset/` or pass `--cadb-root`.
- Run one pilot seed for:
  - `vit_baseline`
  - `vit_row_sinusoidal`
  - `vit_col_sinusoidal`
- Generate one CADB epoch-curve comparison report.

## 2026-06-14 CADB Real-Archive Alignment

### Changed
- Updated `cadb_data.py` to align with the real CADB archive.
- Switched orientation label extraction to `composition_elements.json`.
- Switched split handling to prefer the official `split.json`.
- Verified the real CADB path with a smoke run on the user's local archive.

### Learned
- My first CADB assumption was slightly off:
  - `horizontal / vertical` should come from composition elements, not composition attributes
  - the archive already includes an official split file
- The real dataset path now runs end-to-end through the unified experiment runner.

### Problems / Mismatches
- The current binary task still drops images that contain both horizontal and vertical elements.
- That is intentional for this week's clean pilot, but it is a narrower task than full CADB understanding.

### Next
- Run the full 20-epoch pilot for:
  - `vit_baseline`
  - `vit_row_sinusoidal`
  - `vit_col_sinusoidal`
- Compare epoch curves first, then final summary metrics.
