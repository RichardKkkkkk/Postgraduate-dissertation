# 项目结构

最近对齐：2026-07-20

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
  统一管理训练结果、图、报告、checkpoint 的路径规则。
- `refresh_single_run_figures.py`
  从已有 metrics / summary 重新生成单次实验图。
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

`vit_squared_multiplicative_sinusoidal`、shifted 版本和 `vit_radial_sinusoidal` 目前只有实现，没有对应的正式结果目录。

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
    │       ├── <run_name>_accuracy.png
    │       └── <run_name>_test_confusion_matrix.png
    └── reports/
        └── <report_name>/
            ├── figures/
            ├── comparison_summary.csv
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
