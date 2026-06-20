# ViT Research Playground

这个仓库现在是一个面向论文实验的轻量研究平台，目标不是只“跑通一个 ViT”，而是持续比较不同的图像位置编码和结构偏置。

当前主线已经收敛成：

- 一个统一训练入口：`train_cifar10_experiment.py`
- 一个统一模型注册表：`models/registry.py`
- 一个统一结果输出格式：`metrics / config / summary / curves / checkpoint`
- 两类数据集：
  - `cifar10`
- `synthetic_orientation`
- `synthetic_orientation_clean`
- `synthetic_orientation_hard`
- `cadb_orientation`

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
2. 在 [models/registry.py](/D:/UCL/UCL-dissertation/Postgraduate-dissertation/models/registry.py:1) 注册
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

### `synthetic_orientation_clean`

这是一个更干净的 synthetic v2 版本：
- 仍然是 `horizontal / vertical` 二分类
- 条纹更规整，噪声更低
- 更适合先看 `row-wise / col-wise` positional encoding 能不能在 controlled setting 下拉开差异

### `synthetic_orientation_hard`

这是一个更难的 synthetic v3 版本：
- 主方向条纹仍然决定类别
- 会加入短的正交干扰条纹
- 会加入局部遮挡和更强的位置变化
- 目标是避免 baseline / row / col 太快同时饱和

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

### `cadb_orientation`

这是面向老师本周任务新增的真实数据集分支，基于 CADB 做最小可用的方向性二分类实验。

当前默认假设：

- 使用 CADB 中的 `horizontal` 和 `vertical` composition classes
- 默认只保留“只属于 horizontal”或“只属于 vertical”的样本
- 同时属于两者，或两者都不属于的样本会先跳过
- 训练入口会自己做 deterministic `train / val / test` split

默认路径：

```text
data/CADB_Dataset/
```

期望结构：

```text
CADB_Dataset/
├── composition_elements.json
├── split.json
├── composition_attributes.json
└── images/
```

一句英文解释：

`cadb_orientation is a binary horizontal-vs-vertical subset built from CADB annotations.`

当前实现细节：

- 标签优先来自 `composition_elements.json`
- `horizontal` 和 `vertical` 看的是对应 element annotation 是否非空
- 如果存在 `split.json`，优先使用 CADB 官方 `train/test`
- 再从官方 `train` 里切出 `validation`

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

### 5. Synthetic clean baseline experiment

```bash
python train_cifar10_experiment.py --model vit_baseline --dataset synthetic_orientation_clean --epochs 20 --seed 42 --run-name baseline_synth_clean_seed42
```

### 6. Synthetic hard baseline experiment

```bash
python train_cifar10_experiment.py --model vit_baseline --dataset synthetic_orientation_hard --epochs 20 --seed 42 --run-name baseline_synth_hard_seed42
```

### 5. Fast smoke test

```bash
python train_cifar10_experiment.py --model vit_row_sinusoidal --dataset synthetic_orientation --epochs 1 --train-subset 64 --val-subset 32 --test-subset 32 --num-workers 0 --run-name smoke_row
```

### 6. ResNet18 scratch baseline

```bash
python train_cifar10_experiment.py --model resnet18_scratch --dataset cifar10 --epochs 20 --run-name resnet18_scratch_cifar10
```

### 7. CADB row-wise pilot run

```bash
python train_cifar10_experiment.py --model vit_row_sinusoidal --dataset cadb_orientation --cadb-root data/CADB_Dataset --image-size 96 --epochs 20 --seed 42 --early-stopping-patience 5 --early-stopping-metric val_acc --early-stopping-min-delta 0.001 --run-name row_cadb_seed42
```

### 8. CADB column-wise pilot run

```bash
python train_cifar10_experiment.py --model vit_col_sinusoidal --dataset cadb_orientation --cadb-root data/CADB_Dataset --image-size 96 --epochs 20 --seed 42 --early-stopping-patience 5 --early-stopping-metric val_acc --early-stopping-min-delta 0.001 --run-name col_cadb_seed42
```

### 9. CADB baseline pilot run

```bash
python train_cifar10_experiment.py --model vit_baseline --dataset cadb_orientation --cadb-root data/CADB_Dataset --image-size 96 --epochs 20 --seed 42 --early-stopping-patience 5 --early-stopping-metric val_acc --early-stopping-min-delta 0.001 --run-name baseline_cadb_seed42
```

## Comparison / Report Commands

单次 run 对比：

```bash
python generate_comparison_report.py --run baseline_synth_seed42 --run row_synth_seed42 --run col_synth_seed42 --report-name synthetic_orientation_compare
```

如果你想把重点放在 epoch 曲线上，这个脚本会复用每个 run 的 `metrics.csv`，按 epoch 画出 loss / accuracy 对比图。

CADB 对比同理：

```bash
python generate_comparison_report.py --run baseline_cadb_seed42 --run row_cadb_seed42 --run col_cadb_seed42 --report-name cadb_orientation_compare
```

如果你只想聚焦 `row-wise` 和 `column-wise` 的结构差异，可以直接导出一版更窄的组会 PPT：

```bash
python generate_comparison_report.py --run row_cadb_seed42_noes="ViT Row-wise" --run col_cadb_seed42_noes="ViT Column-wise" --report-name cadb_row_vs_col_noes --title "CADB Orientation: Row-wise vs Column-wise Sinusoidal ViT"
```

`synthetic_orientation_clean` 的 v2 三模型对比也可以直接复用同一个入口：

```bash
python generate_comparison_report.py --run baseline_synth_clean_seed42="ViT Baseline" --run row_synth_clean_seed42="ViT Row-wise" --run col_synth_clean_seed42="ViT Column-wise" --report-name synth_clean_v2_compare --title "Synthetic Orientation Clean v2: Baseline vs Row-wise vs Column-wise"
```

## Saved Outputs

每次训练完成后会保存：

- `results/metrics/<model>/<run_name>_metrics.csv`
- `results/metrics/<model>/<run_name>_config.json`
- `results/metrics/<model>/<run_name>_summary.json`
- `results/metrics/<model>/<run_name>_test_confusion_matrix.csv`
- `results/figures/<model>/<run_name>_loss.png`
- `results/figures/<model>/<run_name>_accuracy.png`
- `results/figures/<model>/<run_name>_test_confusion_matrix.png`
- `checkpoints/<model>/<run_name>_best.pt`

这样做的好处是：

- 同一个模型的 run 会自动放进同一层目录
- `results/metrics/` 和 `results/figures/` 不会因为 run 太多变得难找
- 报告脚本仍然兼容旧的平铺结果

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

### CADB dataset parameters

- `--cadb-root`
- `--cadb-test-ratio`
- `--cadb-label-mode`
- `--cadb-balance-mode`

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

### CADB balance modes

- `--cadb-balance-mode none`
  Keep the original horizontal / vertical class distribution.
- `--cadb-balance-mode train_only`
  Balance only the training split while keeping validation and test on the original distribution.
- `--cadb-balance-mode all_splits`
  Balance train / validation / test separately for the cleanest directional comparison.

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

### CADB dataset defaults

- `cadb_root = data/CADB_Dataset`
- `cadb_test_ratio = 0.2`
- `cadb_label_mode = exclusive`
- default dataset image size = `96`

## Project Structure

当前建议按职责理解目录：

- 根目录：
  只保留训练入口、报告脚本、环境文件和项目说明
- `models/`：
  放所有模型实现和模型注册表
- `datasets/`：
  放所有 dataset loader 和 split 逻辑
- `docs/`：
  放研究路线、学习笔记、跨设备日志
- `results/` / `checkpoints/` / `data/`：
  放实验产物和数据，不作为代码层结构

- [train_cifar10_experiment.py](/D:/UCL/UCL-dissertation/Postgraduate-dissertation/train_cifar10_experiment.py:1)
  Unified training entry.
- [experiment_utils.py](/D:/UCL/UCL-dissertation/Postgraduate-dissertation/experiment_utils.py:1)
  Shared training / evaluation / saving utilities.
- [models/registry.py](/D:/UCL/UCL-dissertation/Postgraduate-dissertation/models/registry.py:1)
  Model registration and dataset routing.
- [models/vit.py](/D:/UCL/UCL-dissertation/Postgraduate-dissertation/models/vit.py:1)
  Baseline ViT.
- [models/vit_rope.py](/D:/UCL/UCL-dissertation/Postgraduate-dissertation/models/vit_rope.py:1)
  1D RoPE ViT.
- [models/vit_rope_2d.py](/D:/UCL/UCL-dissertation/Postgraduate-dissertation/models/vit_rope_2d.py:1)
  2D-aware RoPE ViT.
- [models/vit_axis_sinusoidal.py](/D:/UCL/UCL-dissertation/Postgraduate-dissertation/models/vit_axis_sinusoidal.py:1)
  Row-wise / column-wise sinusoidal ViT.
- [datasets/cifar10_data.py](/D:/UCL/UCL-dissertation/Postgraduate-dissertation/datasets/cifar10_data.py:1)
  CIFAR-10 loaders.
- [datasets/synthetic_orientation_data.py](/D:/UCL/UCL-dissertation/Postgraduate-dissertation/datasets/synthetic_orientation_data.py:1)
  Synthetic horizontal / vertical dataset.
- [datasets/cadb_data.py](/D:/UCL/UCL-dissertation/Postgraduate-dissertation/datasets/cadb_data.py:1)
  CADB orientation subset loader and deterministic split logic.
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
checkpoints/<model>/
results/metrics/<model>/
results/figures/<model>/
results/reports/
```

切机器前建议顺序：

1. `git status`
2. 提交本次有价值的代码和文档
3. `git push`
4. 到另一台机器 `git pull`

如果只是跑实验，不要把临时结果目录一起 `git add`。
## CADB Scene Note

`cadb_orientation` is only the earlier horizontal-vs-vertical pilot subset.

If you want the full original CADB single-label branch, use:

```bash
python train_cifar10_experiment.py --model vit_baseline --dataset cadb_scene --cadb-root data/CADB_Dataset --image-size 96 --epochs 20 --run-name vit_cadb_scene_seed42
```

涓枃璇存槑锛?
- `cadb_scene` 瀵瑰簲 CADB 鍘熷鐨?`scene_categories.json`
- 杩欐槸 10 绫诲崟鏍囩鍒嗙被锛屼笉鏄箣鍓嶇殑 `horizontal / vertical` 浜屽垎瀛愰泦
- 榛樿浼樺厛浣跨敤 `split.json` 鐨?`train / test`锛屽啀浠?`train` 鍒囧嚭 `validation`

## Synthetic Axis-Code Note

These two datasets are designed specifically to test whether row-wise and column-wise positional encodings really produce different inductive biases.

- `synthetic_row_code`
  Every sample is built from horizontal row bands. The class label depends on which rows are active, not on whether the image is horizontal or vertical.
- `synthetic_col_code`
  Every sample is built from vertical column bands. The class label depends on which columns are active.

涓枃璇存槑锛?
- 杩欎袱涓?dataset 涓嶆槸鏂扮殑鈥渉orizontal vs vertical鈥濅簩鍒嗕换鍔°
- `synthetic_row_code` 娴嬭瘯鐨勬槸锛氭ā鍨嬫槸鍚﹁兘鏇村ソ鍦板埄鐢?row position
- `synthetic_col_code` 娴嬭瘯鐨勬槸锛氭ā鍨嬫槸鍚﹁兘鏇村ソ鍦板埄鐢?column position
- 杩欐牱鍙互鏇撮挍瀵瑰湴楠岃瘉 row-wise / column-wise positional bias锛岃€屼笉鏄鍥惧儚璇箟鎴栫汗鐞嗙洊杩?

Recommended quick commands:

```bash
python train_cifar10_experiment.py --model vit_row_sinusoidal --dataset synthetic_row_code --epochs 20 --seed 42 --run-name row_model_on_row_code
python train_cifar10_experiment.py --model vit_col_sinusoidal --dataset synthetic_row_code --epochs 20 --seed 42 --run-name col_model_on_row_code
python train_cifar10_experiment.py --model vit_row_sinusoidal --dataset synthetic_col_code --epochs 20 --seed 42 --run-name row_model_on_col_code
python train_cifar10_experiment.py --model vit_col_sinusoidal --dataset synthetic_col_code --epochs 20 --seed 42 --run-name col_model_on_col_code
```
