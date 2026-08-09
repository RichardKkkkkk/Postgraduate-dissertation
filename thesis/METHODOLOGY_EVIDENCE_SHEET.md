# Methodology Evidence Sheet

This file records the implementation evidence used to draft Chapter 3 of
`Yikai_Zhao_MSc_Dissertation.docx`. It is an audit aid, not dissertation prose.

## 1. Controlled study design

- Main completed experiment: `results/cifar10_final_vit_models_5seeds/`.
- Evidence coverage: 32 registered ViT configurations, seeds 42--46, giving 160
  selected-checkpoint summaries.
- The final summaries share `split_seed=42` and
  `test_evaluation_protocol=selected_checkpoint_only`.
- The compact ViT backbone and training settings are held constant within the
  final CIFAR-10 comparison. The intended factors are positional encoding,
  patch ordering/assignment, and fusion design.
- Primary code evidence: `models/registry.py`,
  `results/cifar10_final_vit_models_5seeds/**/_config.json`, and
  `generate_thesis_statistics.py`.

## 2. Datasets and preprocessing

### CIFAR-10

- Official 50,000-image training set is divided into 45,000 training images and
  5,000 validation images using a random permutation with `split_seed=42`.
- The official 10,000-image test set remains separate.
- Training augmentation: random 32x32 crop with four-pixel padding, followed by
  random horizontal flipping.
- Evaluation preprocessing: tensor conversion and normalisation only.
- Normalisation: mean `(0.4914, 0.4822, 0.4465)` and standard deviation
  `(0.2470, 0.2435, 0.2616)`.
- Evidence: `datasets/cifar10_data.py` and final run configurations.

### CIFAR-100

- Loader and 100-class model interface are implemented but no formal result
  directory is currently present.
- It uses the same 32x32 input size, 45,000/5,000 split construction and
  10,000-image official test set.
- Training augmentation matches CIFAR-10.
- Normalisation: mean `(0.5071, 0.4867, 0.4408)` and standard deviation
  `(0.2675, 0.2565, 0.2761)`.
- Prespecified comparison: no PE, learnable PE, shifted additive PE and shifted
  multiplicative PE, seeds 42--46.
- Evidence: `datasets/cifar100_data.py`, `models/registry.py`,
  `tests/test_cifar100_registration.py`, and `docs/RESEARCH_PLAN.md`.

### Reduced-data CIFAR-10

- Intended training-set sizes: 1,000, 5,000, 10,000 and full data.
- The 5,000-image validation set and official test set are not reduced.
- `make_subset` samples from the 45,000-image training portion using the
  training seed. All models paired at the same seed therefore receive the same
  subset, while subsets vary across seeds. Reported seed variability combines
  training randomness and subset composition.
- Protocol mismatch to resolve before final prose:
  - `run_low_data_sweep.py` and the research plan specify learned versus
    multiplicative PE with learning rate `3e-4`.
  - Existing 1k/5k four-model runs use no PE, learnable PE, shifted
    multiplicative PE and the hybrid, with learning rate `1e-3`.
  - The final dissertation must select one protocol and must not merge the two
    as if they were a single prespecified experiment.

## 3. Shared compact ViT

- Input: 32x32 RGB image.
- Patch projection: Conv2d, kernel size 4, stride 4.
- Patch grid: 8x8 = 64 patch tokens.
- A learned classification token is prepended, giving 65 tokens.
- Embedding dimension: 128.
- Encoder depth: 4 pre-normalisation Transformer blocks.
- Attention heads: 4; head dimension: 32.
- MLP hidden dimension: 512 with GELU.
- Final LayerNorm and linear classification head.
- All four dropout settings are zero in the formal CIFAR-10 runs.
- Single-branch fixed/no-PE parameter count: 801,034 for CIFAR-10.
- Learnable absolute PE parameter count: 809,354.
- Evidence: `models/vit.py`, `models/vit_baseline.py`, `models/registry.py`,
  final configurations and `selected_test_summary_with_ci.csv`.

## 4. Positional encodings

Let `S(p)` be the D-dimensional sinusoidal vector implemented with sine on even
dimensions and cosine on odd dimensions, using the base 10,000 frequency
schedule.

- No PE: tokens are passed to the shared encoder without a positional vector.
- Learnable absolute PE: trainable tensor of shape `(1, 65, 128)`.
- Row PE: `S(r)` for each row coordinate.
- Column PE: `S(c)` for each column coordinate.
- Additive PE: `S(r) + S(c)`.
- Multiplicative PE: `S(r) elementwise-multiplied by S(c)`.
- Shifted variants: row frequencies use exponents indexed by even embedding
  positions, whereas column frequencies use the corresponding odd-position
  exponents. This is a frequency-schedule shift, not a coordinate translation.
- Squared multiplicative PE: `(S(r) elementwise-multiplied by S(c))^2`.
- Radial PE: `S(sqrt(r^2 + c^2))`, with the top-left patch as `(0,0)`.
- Fixed encodings are registered as non-persistent buffers and add no trainable
  parameters. Their classification-token positional vector is zero.
- Evidence: `models/vit_axis_sinusoidal.py`.
- Word synchronisation: Section 3.4 now contains native Word Equations (7)--(12)
  for the common sinusoidal function, axis-wise, additive, multiplicative,
  shifted-frequency, squared and radial constructions, plus a compact PE-family
  table. The shifted definition follows the implemented even/odd frequency
  exponent schedules and is explicitly distinguished from coordinate shifting.

## 5. Patch ordering and assignment

- Physical patches are first produced in row-major order by the convolution and
  flatten operations.
- `index_select` then applies one of four deterministic orders:
  `normal_row`, `normal_col`, `proper_row` (row-wise serpentine), or
  `proper_col` (column-wise serpentine).
- Fixed PE buffers remain indexed by row-major sequence slot. Therefore a
  physical patch moved to slot `s` receives the fixed PE coordinate associated
  with slot `s`.
- The analysis must distinguish physical patch coordinate, sequence slot and
  assigned PE coordinate.
- Evidence: `models/unfolding.py`, `tests/test_unfolding_mapping.py`, and
  `generate_patch_mapping_report.py`.
- Word synchronisation: Equation (13) distinguishes the physical patch selected
  by `pi_u(s)` from the row-major PE coordinate assigned to sequence slot `s`.

## 6. Hybrid and fusion extensions

- Hybrid: learnable absolute PE plus a fixed multiplicative PE scaled by one
  learned scalar `fixed_pos_scale`, initialised to zero; parameter count
  809,355.
- Latent concat fusion: independent row and column encoders, concatenated class
  latents, MLP fusion, shared head; 1,798,538 parameters.
- Mean fusion: elementwise mean of row and column class latents; 1,600,778
  parameters.
- Mean-MLP fusion: mean followed by an MLP; 1,732,746 parameters.
- Bidirectional cross-attention: row tokens query column tokens and vice versa;
  fused class tokens feed either a linear or MLP head; 1,999,114 or 2,031,242
  parameters.
- These models have a capacity confound relative to single-branch models.
- Evidence: `models/vit_axis_sinusoidal.py` and the thesis statistics summary.
- Word synchronisation: Equations (14)--(16) define the learnable--fixed hybrid,
  latent fusion and query/context cross-attention used by the implementation.

## 7. Training and evaluation

- Loss: cross-entropy for CIFAR single-label classification.
- Optimiser: AdamW, learning rate `3e-4`, weight decay `0.05` for the completed
  formal CIFAR-10 experiment.
- Batch size: 128; maximum epochs: 100; data-loader workers: 2.
- Scheduler: ReduceLROnPlateau on validation accuracy, factor 0.5, patience 5,
  minimum learning rate `1e-6`.
- Early stopping: validation accuracy, patience 10, minimum delta 0.001.
- Training seeds: 42, 43, 44, 45 and 46. Python, PyTorch and CUDA RNGs are
  seeded; deterministic-algorithm enforcement is not enabled.
- The best validation state is restored, re-evaluated on validation data and
  evaluated once on the test set.
- Evidence: `train_cifar10_experiment.py`, `experiment_utils.py`, and all final
  summary files.

## 8. Metrics and uncertainty

- Accuracy and cross-entropy loss are primary reported metrics.
- Macro-precision, macro-recall and macro-F1 are computed from the confusion
  matrix; macro-F1 is especially useful for the 100-class extension.
- Parameter count is reported for capacity comparisons.
- Across five seeds, summaries report the mean, sample standard deviation and a
  two-sided 95% Student-t interval with four degrees of freedom:
  `mean +/- 2.776445 * sample_SD / sqrt(5)`.
- Paired differences use matching seeds and are defined as comparison minus
  reference.
- Five seeds are not treated as five independent datasets. The analysis is
  descriptive and avoids unsupported significance or equivalence claims.
- Evidence: `experiment_utils.py` and `generate_thesis_statistics.py`.

## 9. Computing environment

- Primary workstation GPU: NVIDIA GeForce RTX 5070 Ti.
- CPU: AMD Ryzen 7 9800X3D.
- System memory: 32 GB RAM.
- Verified project software versions: Python 3.11.15, PyTorch 2.12.0+cu130
  and Torchvision 0.27.0+cu130.
- The exact Windows edition/build and the machine used for every headline run
  still require confirmation before the drafting note is removed.

## 10. Remaining evidence gates

- Add the original CIFAR technical report to Zotero before finalising dataset
  citations; it is not currently present in the local library.
- Add/verify the AdamW citation in Zotero.
- Freeze the low-data model set and learning rate before presenting it as a
  prespecified experiment.
- Record the Git commit used for every final table.
- Confirm that all headline runs used the stated RTX 5070 Ti, Ryzen 7 9800X3D
  and 32 GB workstation before removing the computing-environment drafting note.
- No methodology figure is required for the present text-only draft. A later
  pipeline schematic should be derived from this evidence sheet.
