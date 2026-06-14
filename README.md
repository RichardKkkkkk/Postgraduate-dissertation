# ViT Research Playground

这个仓库现在是一个面向论文实验的轻量研究平台，目标不是只“跑通一个 ViT”，而是持续比较不同的图像位置编码和结构偏置。

当前主线已经收敛成：

- 一个统一训练入口：`train_cifar10_experiment.py`
- 一个统一模型注册表：`model_registry.py`
- 一个统一结果输出格式：`metrics / config / summary / curves / checkpoint`
- 两类数据集：
  - `cifar10`
  - `synthetic_orientation`

一句英文可以这样记：

`Use one runner, register many model variants.`

## Current Mainline

本周主线不是继续扩展 `RoPE`，而是转向老师要求的方向性实验：

- `vit_baseline`
- `vit_row_sinusoidal`
- `vit_col_sinusoidal`

研究问题是：

- row-wise positional prior 是否更适合 horizontal structure
- column-wise positional prior 是否更适合 vertical structure

因此接下来主要看：

- `epoch-based loss curves`
- `epoch-based accuracy curves`
- synthetic horizontal vs vertical 数据上的对比

## Environment

统一使用 conda 环境：

```bash
conda activate vit_research
```

安装依赖：

```bash
pip install -r requirements.txt
```

## Single Training Entry

项目现在只保留一个正式训练入口：

```bash
python train_cifar10_experiment.py --model vit_baseline
```

也就是说：

- 训练逻辑只维护一份
- early stopping 只维护一份
- metrics / summary / checkpoint 保存逻辑只维护一份
- 新模型不要再新建一个 `train_xxx.py`

后续新增模型的推荐方式：

1. 新建模型文件
2. 在 [model_registry.py](/D:/UCL/UCL-dissertation/Postgraduate-dissertation/model_registry.py:1) 注册
3. 复用统一训练入口

## Supported Models

- `vit_baseline`
  Vanilla ViT with learned absolute positional embedding.
- `vit_rope`
  Basic 1D RoPE ViT.
- `vit_rope_2d`
  Lightweight 2D-aware RoPE ViT.
- `vit_row_sinusoidal`
  Additive row-wise sinusoidal positional embedding.
- `vit_col_sinusoidal`
  Additive column-wise sinusoidal positional embedding.
- `resnet18_scratch`
  CNN baseline without pretrained weights.
- `resnet18_imagenet`
  Optional reference model with ImageNet pretrained weights.

## Supported Datasets

### `cifar10`

默认自然图像分类基线，支持：

- `train / validation / test`
- validation-based early stopping
- confusion matrix
- macro precision / recall / F1

### `synthetic_orientation`

这是本周新增的方向性数据集，先用来快速验证老师提出的假设。

类别是：

- `horizontal`
- `vertical`

数据特征：

- `horizontal` 类图像包含横向条纹
- `vertical` 类图像包含纵向条纹
- 每张图会加少量随机噪声
- 训练 / 验证 / 测试 split 由固定 seed 生成

一句英文可以这样理解：

`The synthetic dataset is a controlled testbed for directional positional bias.`

## Most Useful Commands

### 1. CIFAR-10 baseline

```bash
python train_cifar10_experiment.py --model vit_baseline --dataset cifar10 --epochs 20 --run-name vit_baseline_cifar10
```

### 2. Synthetic row-wise experiment

```bash
python train_cifar10_experiment.py --model vit_row_sinusoidal --dataset synthetic_orientation --epochs 20 --seed 42 --early-stopping-patience 5 --early-stopping-metric val_acc --early-stopping-min-delta 0.001 --run-name row_synth_seed42
```

### 3. Synthetic column-wise experiment

```bash
python train_cifar10_experiment.py --model vit_col_sinusoidal --dataset synthetic_orientation --epochs 20 --seed 42 --early-stopping-patience 5 --early-stopping-metric val_acc --early-stopping-min-delta 0.001 --run-name col_synth_seed42
```

### 4. Synthetic baseline experiment

```bash
python train_cifar10_experiment.py --model vit_baseline --dataset synthetic_orientation --epochs 20 --seed 42 --early-stopping-patience 5 --early-stopping-metric val_acc --early-stopping-min-delta 0.001 --run-name baseline_synth_seed42
```

### 5. Fast smoke test

```bash
python train_cifar10_experiment.py --model vit_row_sinusoidal --dataset synthetic_orientation --epochs 1 --train-subset 64 --val-subset 32 --test-subset 32 --num-workers 0 --run-name smoke_row
```

### 6. ResNet18 scratch baseline

```bash
python train_cifar10_experiment.py --model resnet18_scratch --dataset cifar10 --epochs 20 --run-name resnet18_scratch_cifar10
```

## Comparison / Report Commands

单次 run 对比：

```bash
python generate_comparison_report.py --run baseline_synth_seed42 --run row_synth_seed42 --run col_synth_seed42 --report-name synthetic_orientation_compare
```

如果你想把重点放在 epoch 曲线上，这个脚本会复用每个 run 的 `metrics.csv`，按 epoch 画出 loss / accuracy 对比图。

## Saved Outputs

每次训练完成后会保存：

- `results/metrics/<run_name>_metrics.csv`
- `results/metrics/<run_name>_config.json`
- `results/metrics/<run_name>_summary.json`
- `results/metrics/<run_name>_test_confusion_matrix.csv`
- `results/figures/<run_name>_loss.png`
- `results/figures/<run_name>_accuracy.png`
- `results/figures/<run_name>_test_confusion_matrix.png`
- `checkpoints/<run_name>_best.pt`

其中最重要的原则是：

- `validation` 用于选择模型
- `test` 只用于最终报告

英文一句话：

`Validation is for selection, test is for final reporting.`

## CLI Parameters

### Common parameters

- `--model`
- `--dataset`
- `--data-dir`
- `--results-dir`
- `--checkpoint-dir`
- `--run-name`
- `--epochs`
- `--batch-size`
- `--lr`
- `--weight-decay`
- `--train-subset`
- `--val-subset`
- `--test-subset`
- `--val-ratio`
- `--num-workers`
- `--seed`
- `--early-stopping-patience`
- `--early-stopping-min-delta`
- `--early-stopping-metric`

### ViT-specific parameters

- `--embedding-dropout`
- `--attention-dropout`
- `--projection-dropout`
- `--mlp-dropout`

### RoPE-specific parameter

- `--rope-base`

### Synthetic dataset parameters

- `--synthetic-train-size`
- `--synthetic-val-size`
- `--synthetic-test-size`
- `--synthetic-line-width`
- `--synthetic-noise-std`
- `--synthetic-max-stripes`

### ResNet-specific parameter

- `--image-size`

## Default Values

### Shared defaults

- `epochs = 5`
- `val_ratio = 0.1`
- `num_workers = 2`
- `seed = 42`
- `early_stopping_metric = val_acc`
- `early_stopping_min_delta = 0.0`
- `early_stopping_patience = disabled by default`

### ViT defaults

- `batch_size = 128`
- `lr = 3e-4`
- `weight_decay = 0.05`
- all dropout values default to `0.0`

### ResNet18 defaults

- `batch_size = 64`
- `lr = 1e-4`
- `weight_decay = 0.01`

### Synthetic dataset defaults

- `synthetic_train_size = 2400`
- `synthetic_val_size = 600`
- `synthetic_test_size = 600`
- `synthetic_line_width = 3`
- `synthetic_noise_std = 0.08`
- `synthetic_max_stripes = 4`

## Project Structure

- [train_cifar10_experiment.py](/D:/UCL/UCL-dissertation/Postgraduate-dissertation/train_cifar10_experiment.py:1)
  Unified training entry.
- [model_registry.py](/D:/UCL/UCL-dissertation/Postgraduate-dissertation/model_registry.py:1)
  Model registration and dataset routing.
- [experiment_utils.py](/D:/UCL/UCL-dissertation/Postgraduate-dissertation/experiment_utils.py:1)
  Shared training / evaluation / saving utilities.
- [cifar10_data.py](/D:/UCL/UCL-dissertation/Postgraduate-dissertation/cifar10_data.py:1)
  CIFAR-10 loaders.
- [synthetic_orientation_data.py](/D:/UCL/UCL-dissertation/Postgraduate-dissertation/synthetic_orientation_data.py:1)
  Synthetic horizontal / vertical dataset.
- [vit.py](/D:/UCL/UCL-dissertation/Postgraduate-dissertation/vit.py:1)
  Baseline ViT.
- [vit_rope.py](/D:/UCL/UCL-dissertation/Postgraduate-dissertation/vit_rope.py:1)
  1D RoPE ViT.
- [vit_rope_2d.py](/D:/UCL/UCL-dissertation/Postgraduate-dissertation/vit_rope_2d.py:1)
  2D-aware RoPE ViT.
- [vit_axis_sinusoidal.py](/D:/UCL/UCL-dissertation/Postgraduate-dissertation/vit_axis_sinusoidal.py:1)
  Row-wise / column-wise sinusoidal ViT.
- [generate_comparison_report.py](/D:/UCL/UCL-dissertation/Postgraduate-dissertation/generate_comparison_report.py:1)
  Comparison plots and PPT/report generation.
- [docs/LEARNING_NOTES.md](/D:/UCL/UCL-dissertation/Postgraduate-dissertation/docs/LEARNING_NOTES.md:1)
  Implementation explanations.
- [docs/DEVELOPMENT_MAP.md](/D:/UCL/UCL-dissertation/Postgraduate-dissertation/docs/DEVELOPMENT_MAP.md:1)
  Research roadmap.
- [docs/PROJECT_LOG.md](/D:/UCL/UCL-dissertation/Postgraduate-dissertation/docs/PROJECT_LOG.md:1)
  Cross-device progress log.

## Git Reminder

这些目录通常不要提交：

```text
data/
checkpoints/
results/metrics/
results/figures/
results/reports/
```

切机器前建议顺序：

1. `git status`
2. 提交本次有价值的代码和文档
3. `git push`
4. 到另一台机器 `git pull`

如果只是跑实验，不要把临时结果目录一起 `git add`。
