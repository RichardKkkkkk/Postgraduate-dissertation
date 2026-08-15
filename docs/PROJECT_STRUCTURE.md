# 项目结构

最近对齐：2026-07-29

这个文档只回答一件事：

当前仓库里，每个目录和关键文件分别负责什么。

## 根目录

- `train_cifar10_experiment.py`
  统一训练入口。
- `experiment_utils.py`
  训练、验证、测试、early stopping、指标计算、画图、保存结果。
- `generate_comparison_report.py`
  读取多个 run 的结果，生成对比图、汇总表和 PPT。
- `analyze_per_class_report.py`
  读取 per-class 指标，生成按类别的对比分析。
- `run_seed_sweep.py`
  多个 seed 的批量运行入口。
- `summarize_seed_sweep.py`
  多 seed 结果聚合与汇总。
- `result_paths.py`
  统一管理训练结果、图、报告、checkpoint 的路径规则；对比报告未指定 experiment 时，
  可以按唯一 run name 跨 experiment 查找结果。
- `refresh_single_run_figures.py`
  从已有 metrics / summary 重新生成单次实验图。
- `paper_plotting.py`
  单模型、模型对比和多 seed 图共享的论文绘图样式、颜色、尺度和 PNG/PDF 保存逻辑。
- `requirements.txt`
  Python 依赖的锁定版本。
- `environment.yml`
  `vit_research` Conda 环境入口，Python 固定为 3.11。

一句英文解释：

`result_paths.py is the artifact router of the project.`

## `models/`

- `registry.py`
  模型注册表，也是统一入口背后的总开关。
- `vit.py`
  原始 baseline ViT，带 learnable positional embedding。
- `vit_baseline.py`
  不加位置编码的 ViT baseline。
- `vit_axis_sinusoidal.py`
  row-wise / column-wise / radial / additive / multiplicative / squared multiplicative、
  hybrid PE、row/column latent fusion 等 sinusoidal 变体。
- `unfolding.py`
  patch flatten / unfolding 顺序工具，支持 normal row、normal column、proper row、proper column。
- `vit_rope.py`
  1D RoPE 版本。
- `vit_rope_2d.py`
  2D RoPE 版本。

## `datasets/`

- `cifar10_data.py`
  CIFAR-10 数据读取和类别名。
- `cadb_data.py`
  CADB 数据读取、标签解析、split、dataloader。
- `synthetic_orientation_data.py`
  synthetic 数据集生成与加载。

## `docs/`

- `PROJECT_STRUCTURE.md`
  项目结构说明。
- `RESEARCH_PLAN.md`
  当前研究主线、实验协议、下一步计划。
- `PROJECT_LOG.md`
  项目日志，记录最近做了什么、学到了什么、接下来做什么。
- `LEARNING_NOTES.md`
  代码理解、PyTorch 语法、实现逻辑笔记。
- `FIGURE_STANDARD.md`
  论文实验图的统一规范，包括 test holdout、单模型图、模型对比图和多 seed mean ± std 图。

## 运行产物目录

- `data/`
  本地数据集，不提交。
- `results/`
  实验结果、图和报告。
- `checkpoints/`
  保存最佳模型参数。

当前仓库中的正式实验目录：

- `results/cadb_elements_positional_100e/`
  CADB Elements 八模型、seed 42。
- `results/cifar10_positional_8models/`
  CIFAR-10 八模型、seed 42。
- `results/cifar10_positional_8models_5seeds/`
  CIFAR-10 八模型、seed 42-46，以及 mean ± std 汇总。

当前本地还有若干未整理进最终论文协议的 exploratory 结果目录：

- `results/cifar10_positional_squared_seed42/`
- `results/cifar10_positional_radial_seed42/`
- `results/cifar10_unfolding_15_seed42/`
- `results/cifar10_hybrid_seed42/`
- `results/cifar10_latent_fusion_seed42/`
- `results/cifar10_fusion_variants_seed42/`
- `results/cifar10_low_data_seed42/`
- `results/smoke_*`

这些目录用于筛选候选模型和检查图形规范。正式论文统计结论仍需要在修正 test
holdout 流程后重新跑 multi-seed。

历史上已有 56 个 checkpoint 和上述结果被 Git 追踪。不要因为 `checkpoints/` 出现在 `.gitignore` 中就假设这些历史文件未被追踪；清理前必须先制定跨设备保留方案。

## 当前统一的实验目录结构

现在默认按 `experiment_name` 归档，风格和之前的 `cadb_elements_positional_100e` 一致：

```text
results/
└── <experiment_name>/
    ├── metrics/
    │   └── <model>/
    │       ├── <run_name>_metrics.csv
    │       ├── <run_name>_config.json
    │       ├── <run_name>_summary.json
    │       └── <run_name>_test_confusion_matrix.csv
    ├── figures/
    │   └── <model>/
    │       ├── <run_name>_loss.png
    │       ├── <run_name>_loss.pdf
    │       ├── <run_name>_accuracy.png
    │       ├── <run_name>_accuracy.pdf
    │       └── <run_name>_test_confusion_matrix.png
    └── reports/
        └── <report_name>/
            ├── figures/
            │   ├── val_loss_comparison.png
            │   ├── val_loss_comparison.pdf
            │   ├── val_acc_comparison.png
            │   ├── val_acc_comparison.pdf
            │   ├── train_loss_comparison.png
            │   ├── train_loss_comparison.pdf
            │   ├── train_acc_comparison.png
            │   ├── train_acc_comparison.pdf
            │   ├── paper_selected_test_accuracy.png
            │   └── paper_selected_test_accuracy.pdf
            ├── comparison_summary.csv
            ├── publication_selected_checkpoints.csv
            ├── figure_captions.md
            ├── overview.md
            ├── presentation_summary.json
            └── report_manifest.json

checkpoints/
└── <experiment_name>/
    └── <model>/
        └── <run_name>_best.pt
```

如果不显式传 `--experiment-name`，默认会用 `dataset` 名字作为实验目录名。

下一组老师方法扩展统一使用：

```text
results/cifar10_teacher_extensions/
checkpoints/cifar10_teacher_extensions/
```

## 推荐理解顺序

如果你想熟悉整个项目，建议按这个顺序看：

1. `README.md`
2. `train_cifar10_experiment.py`
3. `models/registry.py`
4. `result_paths.py`
5. `experiment_utils.py`
6. `models/vit.py`
7. `models/vit_baseline.py`
8. `models/vit_axis_sinusoidal.py`
9. `datasets/cadb_data.py`
10. `docs/RESEARCH_PLAN.md` 中的当前证据、协议问题和下一步
## Dissertation and generalisation additions (2026-08-07)

- `datasets/cifar100_data.py`
  CIFAR-100 train/validation/test loaders, 100 class names and dataset-specific normalisation.
- `generate_thesis_statistics.py`
  Validates and aggregates the 160 final CIFAR-10 summaries for thesis tables.
- `generate_patch_mapping_report.py`
  Exports the deterministic patch-to-position audit table.
- `run_low_data_sweep.py`
  Retains the earlier two-model reduced-data proposal; it is not the reproduction
  command for the completed four-model `lr=1e-3` result set.
- `generate_robustness_figures.py`
  Validates and plots the completed 1k/5k/10k CIFAR-10 and four-model CIFAR-100
  suites, exporting PNG/PDF figures, source CSV files, captions and a manifest.
  The default publication-facing output is
  `results/reports/thesis_robustness_figures_v2/`.
- `generate_thesis_figures.py`
  Generates the main CIFAR-10 comparison package under
  `results/cifar10_final_vit_models_5seeds/reports/thesis_comparison_figures_v2/`.
  Epoch curves use colour plus line style without point markers; selected-test
  summaries use seed circles and mean diamonds with 95% t intervals.
- `generate_final_test_figures.py`
  Generates the frozen teacher-requested selected-test table and seven-figure
  package under `results/reports/thesis_selected_test_figures_v1/`. It reads the
  three CIFAR-10 low-data `_lr3e4` experiments, the CIFAR-100 `_lr3e4`
  experiment and the existing full-data CIFAR-10 suite. It fails if the
  full-data/low-data configuration alignment gate does not pass.
- `generate_final_thesis_evidence.py`
  Generates the final paper-facing package under
  `results/reports/thesis_final_evidence_figures_v1/`: six five-seed validation
  trajectory figures with pointwise 95% t intervals, selected-checkpoint test
  tables, four test-only auxiliary figures, source CSVs, captions, a
  configuration audit and a manifest.
- `audit_coordinate_aligned_unfolding.py`
  Exports the 8×8 physical-patch/sequence-slot/original-PE/assigned-PE mapping,
  audits permutation-sensitive operations and records same-weight forward
  equivalence for five fixed PE families and four unfoldings.
- `run_coordinate_aligned_unfolding_sweep.py`
  Validates reuse of the 25 protocol-matched normal-row runs and safely runs the
  75 required non-normal-row coordinate-aligned CIFAR-10 conditions. It invokes
  `analyze_coordinate_aligned_unfolding.py` after successful completion.
- `analyze_coordinate_aligned_unfolding.py`
  Validates the 100-cell coordinate-aligned matrix, checks run artifacts and
  protocol fields, then writes per-seed and mean ± 95% CI tables. Historical
  sequence-slot results are exported separately and are never pooled.
- `tests/test_unfolding_mapping.py`
  Verifies sequence and coordinate mappings with labelled patches.
- `tests/test_cifar100_registration.py`
  Verifies dataset metadata and the 100-logit model interface.
- `thesis/tools/build_dissertation.py`
  Builds the one fixed no-figure MSc dissertation Word file from verified result tables.
