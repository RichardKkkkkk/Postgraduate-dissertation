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

## 2026-06-15 Directory Cleanup

### Changed
- Moved model implementation files into `models/`:
  - `vit.py`
  - `vit_rope.py`
  - `vit_rope_2d.py`
  - `vit_axis_sinusoidal.py`
  - `registry.py`
- Moved dataset loaders into `datasets/`:
  - `cifar10_data.py`
  - `synthetic_orientation_data.py`
  - `cadb_data.py`
- Updated imports in the unified runner and seed-sweep scripts.
- Removed the obsolete `train_resnet18_scratch_cifar10.py` wrapper.
- Updated README, learning notes, and agent instructions for the new layout.
- Added `tmp_*/` to `.gitignore` so smoke artifacts stop cluttering the worktree.

### Learned
- The project has now reached the point where root-level flat files hurt extensibility more than they help.
- Keeping entry scripts in the root but moving implementations into `models/` and `datasets/` is a good balance between cleanliness and ease of use.

### Problems / Mismatches
- There are still many historical experiment artifacts under `results/reports/`; they are not code-structure problems, but they still make `git status` noisy.
- `docs/DEVELOPMENT_MAP.md` currently has local user changes and was intentionally left untouched in this cleanup.

### Next
- Verify the reorganized imports with model smoke tests.
- Continue using `train_cifar10_experiment.py` as the only formal training entry.

## 2026-06-15 Confusion Matrix Annotation Fix

### Changed
- Updated the shared confusion-matrix plotting helper so saved run figures now include the per-cell count labels.
- Kept the colorbar and class-axis labels, and added adaptive text color so large values remain readable.

### Learned
- The report generator already had annotation logic, but the direct training output path did not.
- The issue was a plotting inconsistency, not a metrics or evaluation bug.

### Next
- Regenerate the current confusion-matrix PNG files from the saved CSVs so the existing CADB figures are immediately presentation-ready.

## 2026-06-15 Results Directory Cleanup

### Changed
- Added a shared artifact-path helper so training outputs now save to model-scoped folders:
  - `results/metrics/<model>/...`
  - `results/figures/<model>/...`
  - `checkpoints/<model>/...`
- Updated the unified trainer, seed sweep, comparison report, and per-class report scripts to use the same lookup logic.
- Kept backward compatibility so old flat artifacts can still be loaded by run name.

### Learned
- The real problem was not just "too many files"; it was that saving and reading logic were duplicated across scripts.
- Centralizing path resolution makes future cleanup much safer.

### Next
- Migrate the kept experiment artifacts into the new folder structure.
- Remove outdated early-stopping CADB outputs and temporary compatibility reports.

## 2026-06-15 CADB Balance Mode

### Changed
- Added `--cadb-balance-mode` to the unified training interface.
- Supported:
  - `none`
  - `train_only`
  - `all_splits`
- Wired the new option into both ViT and ResNet CADB dataloaders and saved it in the run config.

### Learned
- This makes it much easier to separate "performance on the original imbalanced data" from "behavior under a class-balanced directional comparison".

### Next
- Re-run `vit_baseline`, `vit_row_sinusoidal`, and `vit_col_sinusoidal` on balanced CADB and check whether the row/col bias becomes clearer.

## 2026-06-17 Synthetic Orientation Clean v2

### Changed
- Added a second synthetic directional dataset branch: `synthetic_orientation_clean`.
- Kept it separate from the original synthetic dataset so controlled ablations can compare:
  - noisier synthetic
  - cleaner synthetic
- Registered the new dataset in the unified experiment runner.

### Learned
- The goal of this branch is not realism; it is to provide a cleaner sanity-check setting for row-wise vs column-wise positional bias.

### Next
- Preview a few generated samples.
- Run `vit_baseline`, `vit_row_sinusoidal`, and `vit_col_sinusoidal` on the clean synthetic split and inspect whether the gap becomes more obvious.

## 2026-06-18 Synthetic Orientation Hard v3

### Changed
- Added `synthetic_orientation_hard` as a third directional synthetic branch.
- The new split keeps the main horizontal / vertical signal, but adds:
  - orthogonal distractor fragments
  - local occlusion
  - stronger span and position variation

### Learned
- `synthetic_orientation_clean` was useful as a sanity check, but it was already too easy because all three models nearly saturated.
- A harder controlled dataset is a better next step than immediately jumping to a noisier real-world dataset.

### Next
- Preview the generated samples.
- Run `baseline / row / col` on the hard synthetic split and check whether the gap becomes more meaningful.

## 2026-06-17 Row Vs Col Sinusoidal PPT

### Changed
- Extended `generate_comparison_report.py` with a dedicated `row_vs_col_sinusoidal` scenario.
- Added scenario-specific title, overview wording, and conclusion wording for `vit_row_sinusoidal` vs `vit_col_sinusoidal`.
- Generated a focused CADB deck from:
  - `row_cadb_seed42_noes`
  - `col_cadb_seed42_noes`

### Learned
- This comparison is easier to present without the baseline run on the same slides.
- The key story is now a controlled structural swap:
  - same ViT backbone
  - fixed sinusoidal positional embedding
  - only the spatial axis used for positional indexing changes
- In the current CADB run, `row-wise` finishes slightly higher on final test accuracy, while `column-wise` reaches better best validation accuracy and stronger selected test macro F1.

### Next
- Reuse the same row-vs-col report framing for later balanced CADB runs.
- If more seeds are added, keep this axis-specific wording but move the headline to mean/std summary reporting.

## 2026-06-18 Synthetic Clean v2 PPT

### Changed
- Extended `generate_comparison_report.py` with an `axis_bias_trio` scenario for:
  - `vit_baseline`
  - `vit_row_sinusoidal`
  - `vit_col_sinusoidal`
- Added scenario-specific wording so clean synthetic v2 results can be presented as:
  - same ViT backbone
  - same task
  - three positional choices
- Generated a new deck from:
  - `baseline_synth_clean_seed42`
  - `row_synth_clean_seed42`
  - `col_synth_clean_seed42`

### Learned
- `synthetic_orientation_clean` is now best treated as a sanity-check dataset, not the main evidence slide, because all three models are already close to saturation.
- Even in that near-saturated regime, `ViT Column-wise` gives the cleanest selected-checkpoint result on test macro F1.

### Next
- Use the v2 clean deck to show the controlled sanity-check stage.
- Use the harder synthetic v3 deck as the next place where model differences should become more meaningful.

## 2026-06-18 Full CADB Scene Branch

### Changed
- Added a new dataset branch: `cadb_scene`.
- Wired `cadb_scene` into the unified registry for:
  - `vit_*`
  - `resnet18_scratch`
  - `resnet18_imagenet`
- Implemented the loader against the original CADB files:
  - `scene_categories.json`
  - `split.json`
- Kept the existing training contract:
  - official `train/test`
  - validation split carved from official `train`
  - same metrics / checkpoint / plotting path as other datasets

### Learned
- "Full/original CADB" is not the same thing as the earlier `cadb_orientation` pilot subset.
- The cleanest first original-CADB task in this repo is scene classification, because it stays single-label and fits the current experiment runner directly.
- Composition elements and composition attributes remain useful future branches, but they would require multi-label or regression handling.

### Next
- Smoke-test `cadb_scene` through the unified runner.
- If it trains cleanly, use it as the new default meaning of "original CADB" in future comparisons.

## 2026-06-20 Synthetic Axis-Code Datasets

### Changed
- Added two new synthetic dataset branches:
  - `synthetic_row_code`
  - `synthetic_col_code`
- Kept them inside the unified experiment runner instead of creating a separate script.
- Designed both tasks so the label depends on absolute row/column activation patterns, not on scene semantics.
- Verified the expected directional split with quick sanity runs:
  - `vit_row_sinusoidal` reaches 100% on `synthetic_row_code`
  - `vit_col_sinusoidal` stays near chance on `synthetic_row_code`
  - `vit_col_sinusoidal` reaches 100% on `synthetic_col_code`
  - `vit_row_sinusoidal` stays near chance on `synthetic_col_code`

### Learned
- The earlier horizontal-vs-vertical tasks were still too entangled with content cues.
- A cleaner way to test positional bias is to keep the visual motif direction fixed inside each task and let the class depend on which rows or columns are active.
- This synthetic setup is much closer to a causal probe of row-wise vs column-wise positional encoding.

### Next
- Re-run the same row/col comparison with full training settings instead of sanity subsets.
- Decide whether this synthetic axis-code evidence should become the main motivation slide before returning to CADB or other real datasets.
