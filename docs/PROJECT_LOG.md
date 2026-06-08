# Project Log

这个文件是两台电脑协作开发时的交接日志。它记录“现在做到哪一步、为什么这么做、下一步是什么”。每次 Codex 开始工作时应该先读这里。

## 2026-05-29 Snapshot

### Current Stage

项目已经从最小 ViT 复现推进到一套较完整的 CIFAR-10 研究基础设施：

- 原始 ViT baseline 已实现，文件是 `vit.py`。
- 基础 RoPE 版本已拆到 `vit_rope.py`，避免污染原始 baseline。
- CIFAR-10 训练已经支持 train / validation / test 流程。
- 已加入 early stopping 和基于 validation 的 checkpoint 选择。
- 训练会保存 metrics、config、summary、confusion matrix、macro precision / recall / F1 和 best checkpoint。
- 已加入 ResNet18 CNN baseline，用于对照 ViT。
- 已加入比较报告生成脚本 `generate_comparison_report.py`，可以输出汇总 CSV、overview、图表和 PPT。
- 当前已有一组报告在 `results/reports/weekly_cnn_vs_vit_baseline/`。

### Latest Observed Result

已有报告显示：

- CNN Baseline final test accuracy: 94.92%
- ViT Dropout 0.1 final test accuracy: 73.40%
- CNN 当前明显强于小 ViT，差距约 21.52 percentage points。

这个结果符合直觉：CNN 有更强图像归纳偏置，而且如果使用预训练或更成熟的设置，小数据集上通常更稳。

### Important Sync Note

README 和 `docs/DEVELOPMENT_MAP.md` 中提到了统一入口 `train_cifar10_experiment.py`，但当前工作区文件列表里没有这个文件。

下一步继续开发前，需要先确认：

- 这个文件是否在另一台电脑上存在但还没 push。
- 这个文件是否曾经存在但被删除。
- 当前统一入口是否其实由 `train_cifar10.py` + `train_cnn_cifar10.py` 承担。

在解决这个不一致之前，不建议继续基于 `train_cifar10_experiment.py` 扩展新模型。

### Recommended Next Step

优先做一个“同步和入口清理”小任务：

1. 检查 GitHub / 另一台电脑上是否有 `train_cifar10_experiment.py`。
2. 如果有，把它 pull 回来并 smoke test。
3. 如果没有，修正文档，把当前真实入口写清楚。
4. 然后再进入下一步研究实验：比较 `vit_baseline` vs `vit_rope`，判断基础 RoPE 是否值得扩展成 2D image-aware RoPE。

### Next Research Direction

短期研究路径建议：

1. 先确认 baseline 训练入口和报告入口完全一致。
2. 跑干净的 ViT baseline 和 ViT + RoPE 对比。
3. 观察 test accuracy、macro F1、confusion matrix 和训练曲线。
4. 如果基础 RoPE 有正向信号，再设计 2D RoPE。
5. 如果基础 RoPE 没有提升，也仍然有论文价值：可以说明一维序列式 RoPE 不一定适合图像 patch 网格，从而引出二维位置建模。

## Log Format For Future Entries

以后每次有明显进展，按这个格式追加：

```text
## YYYY-MM-DD Short Title

### Changed
- ...

### Learned
- ...

### Problems / Mismatches
- ...

### Next
- ...
```

## 2026-06-02 Explicit ResNet18 Scratch Entry

### Changed
- Added `train_resnet18_scratch_cifar10.py` as an explicit ResNet18 scratch training entry.
- The new script reuses `train_cnn_cifar10.py` but forces `--weights none`.
- Updated README, learning notes, and development map.
- Smoke tested the scratch entry with a tiny CIFAR-10 subset.

### Learned
- The existing CNN script already supported `--weights none`, but a dedicated scratch entry makes the experiment intent clearer and reduces command mistakes.
- The new wrapper correctly records `weights: none` and `image_size: 32` in the config JSON.
- Passing `--weights imagenet` to the scratch wrapper is blocked on purpose.

### Problems / Mismatches
- The repository still documents `train_cifar10_experiment.py`, but the file is not present in the current working tree.

### Next
- Before adding more model variants, resolve whether `train_cifar10_experiment.py` should be restored or removed from docs.

## 2026-06-06 Unified Runner Cleanup

### Changed
- Restored `train_cifar10_experiment.py` in the working tree.
- Refactored the unified runner into a registry-style structure.
- Kept one shared training loop and moved model-specific differences into explicit registration blocks.
- Updated README, learning notes, and development map to explain the new extension pattern.

### Learned
- The unified runner is now real rather than only documented.
- Adding a new model should now mean:
  - add the model file
  - register it in `train_cifar10_experiment.py`
- This is cleaner than continuing to create one training `main` per model variant.

### Problems / Mismatches
- `results/reports/unified_model_compare/` is currently untracked because reports are still allowed through `.gitignore`.
- Some older historical references in README still mention older entry-point ideas; a later docs pass can simplify those if needed.

### Next
- Use the unified runner for the next clean baseline experiments.
- Run `vit_baseline` vs `vit_rope` as the first structure-focused comparison.
- If RoPE shows useful signal, extend the model design rather than creating another training script.

## 2026-06-06 Single Training Entry Refactor

### Changed
- Removed the old dedicated training scripts.
- Extracted shared training logic into `experiment_utils.py`.
- Extracted CIFAR-10 dataloader logic into `cifar10_data.py`.
- Extracted model registration and defaults into `model_registry.py`.
- Kept `train_cifar10_experiment.py` as the only formal training entrypoint.
- Rewrote README so it documents the new one-runner structure clearly.

### Learned
- A single training script is much cleaner for later ablations.
- New models now fit the project structure as:
  - new model file
  - new registration entry
- This separates model design from training protocol more cleanly.

### Problems / Mismatches
- `results/reports/unified_model_compare/` is still untracked and should stay out of normal code commits unless the user explicitly wants to version report artifacts.

### Next
- Run a clean `vit_baseline` vs `vit_rope` comparison with the new single-entry structure.
- Decide whether the next research step is `2D RoPE` or another lightweight image bias based on that result.

## 2026-06-07 2D RoPE Registration

### Changed
- Added a new model file: `vit_rope_2d.py`.
- Implemented a lightweight `2D-aware RoPE` variant on top of the current small ViT.
- Registered `vit_rope_2d` in `model_registry.py` so it can be trained from the unified runner.
- Updated README, learning notes, and development map so the current main experiment line is documented clearly.
- Smoke tested `vit.py`, `vit_rope_2d.py`, and the unified runner with `--model vit_rope_2d`.

### Learned
- The first clean 2D version does not need a full Swin-style rewrite.
- A simple and explainable design is enough for the next ablation step:
  - keep the same patch grid
  - keep `cls token` outside rotation
  - split each attention head into row-rotated and col-rotated halves
- This keeps the baseline boundary clean and makes the next comparison easy to explain.

### Problems / Mismatches
- `results/reports/unified_model_compare/` is still untracked and should stay out of normal code commits.
- `resnet18_imagenet` still exists in the runner, but it should currently be treated as optional reference only, not a main control line.

### Next
- Run the first clean baseline trio:
  - `vit_baseline`
  - `vit_rope`
  - `resnet18_scratch`
- Then run the second-round structure comparison:
  - `vit_baseline`
  - `vit_rope`
  - `vit_rope_2d`
- After that, decide whether to continue toward locality bias or small-sample evaluation.

## 2026-06-07 Multi-Seed Sweep Helper

### Changed
- Added `run_seed_sweep.py` to automate multi-seed experiments.
- The new script loops through seeds and model names, then calls the unified training runner.
- After each seed finishes, it automatically calls `generate_comparison_report.py`.
- Updated the report display mapping so `vit_rope_2d` shows up cleanly in slides and summaries.
- Updated README, learning notes, and development map for the new multi-seed workflow.

### Learned
- The next useful methodological step is no longer “add more models first”.
- It is better to test whether the current `vit_baseline -> vit_rope -> vit_rope_2d` ranking is stable across multiple random seeds.
- Per-seed reports are a good intermediate layer before later computing mean/std summaries.

### Problems / Mismatches
- Multi-seed runs will generate more ignored result artifacts under `results/` and `checkpoints/`; these should stay out of git.

### Next
- Run several seeds for `vit_baseline`, `vit_rope`, and `vit_rope_2d`.
- Inspect whether `vit_rope_2d` stays ahead consistently.
- Only after that start parameter tuning.

## 2026-06-07 Seed Summary Layer

### Changed
- Added `summarize_seed_sweep.py`.
- The new script reads multiple run summaries and aggregates them into `mean/std/min/max`.
- It also writes per-seed CSV, delta-vs-reference CSV, overview markdown, and cross-seed plots.
- Updated README, learning notes, and development map for the new summary workflow.

### Learned
- Per-seed comparison reports are useful, but they are still one level too raw for later thesis writing.
- A separate summary layer makes it much easier to answer:
  - is `vit_rope_2d` better on average
  - how large is the variance
  - is the gain stable enough to justify parameter tuning

### Problems / Mismatches
- The new summary artifacts also live under `results/reports/` and should remain outside normal code commits.

### Next
- Run `summarize_seed_sweep.py` on the current `cifar10_main` results.
- Use the mean/std table to decide whether tuning should focus on `vit_rope_2d`.

## 2026-06-08 Per-Class Comparison Layer

### Changed
- Added `analyze_per_class_report.py`.
- The new script reads saved run summaries and compares per-class accuracy / per-class F1.
- It exports per-class CSV tables, grouped comparison plots, a delta-vs-reference plot, and an overview markdown file.
- Updated README, learning notes, and development map for the new meeting-analysis workflow.

### Learned
- Multi-seed averages are useful for deciding whether a direction is stable.
- Per-class analysis is useful for deciding what kind of improvement the structure is actually giving.
- This creates a better bridge from raw experiment results to meeting slides and thesis narrative.

### Problems / Mismatches
- The new report artifacts also live under `results/reports/` and should remain outside normal code commits.

### Next
- Run the per-class report on `vit_baseline`, `vit_rope`, and `vit_rope_2d`.
- Use the resulting class-level gains and weak-class analysis in the next group meeting.

## 2026-06-07 Seed Summary PPT Export

### Changed
- Extended `summarize_seed_sweep.py` so the seed-summary layer can now export a presentation deck directly.
- Added a PPT structure that matches weekly lab reporting better:
  - title
  - summary highlights
  - aggregate table
  - delta vs baseline table
  - metric pages with aggregate bars and per-seed lines
  - conclusion
- Generated a real deck from `cifar10_main_seed_summary` instead of reusing a single-seed comparison deck.

### Learned
- `generate_comparison_report.py` and `summarize_seed_sweep.py` serve different reporting layers:
  - `generate_comparison_report.py` is for comparing concrete runs
  - `summarize_seed_sweep.py` is for showing mean/std evidence across multiple seeds
- For group meetings, the 3-seed aggregate deck is usually the better default because it tells the stability story instead of over-emphasizing one run.

### Problems / Mismatches
- It was easy to confuse the per-run report script with the multi-seed summary script because only the former used to export PPT.
- Generated PPT and exported slide images live under `results/reports/` and should still stay out of normal code commits.

### Next
- Keep using `cifar10_main_seed_summary` as the main weekly meeting deck for the current three-seed ViT comparison.
- Add future model families into the same summary workflow rather than creating separate ad hoc slide scripts.

## 2026-06-08 Report Entry Unification

### Changed
- Rolled back the direct PPT export that had been added to `summarize_seed_sweep.py`.
- Added a new seed-summary input mode to `generate_comparison_report.py`.
- The report script can now read an existing `summary_manifest.json` and render the weekly meeting PPT from that aggregate summary.
- Updated README and learning notes so the script responsibilities are clear again.

### Learned
- The cleaner design is:
  - `summarize_seed_sweep.py` = produce summary artifacts
  - `generate_comparison_report.py` = render meeting materials
- This keeps a single PPT entry point while still preserving a separate analysis layer.

### Problems / Mismatches
- The first implementation worked, but it blurred the boundary between “compute evidence” and “render report”.
- That made it harder to remember which script should be used to generate the final deck.

### Next
- Use `generate_comparison_report.py --summary-report <report_name>` for aggregate multi-seed decks.
- Keep `summarize_seed_sweep.py` focused on reproducible summary outputs only.
