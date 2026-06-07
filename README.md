# ViT Learning

这个项目用于逐步复现和研究 Vision Transformer，并围绕“如何让 Transformer 更适合图像”搭建一个可持续扩展的实验平台。

当前项目已经具备：

- 原始 `ViT baseline`
- 基础 `ViT + RoPE`
- `ResNet18 scratch`
- `ResNet18 ImageNet pretrained`
- `train / validation / test`
- validation-based early stopping
- confusion matrix / macro precision / macro recall / macro F1
- 自动保存 metrics / config / summary / checkpoint
- 对比报告和 PPT 生成脚本

## 环境

统一使用 conda 环境：

```bash
conda activate vit_research
```

新机器安装依赖：

```bash
pip install -r requirements.txt
```

## 唯一训练入口

现在项目只保留一个正式训练入口：

```bash
python train_cifar10_experiment.py --model vit_baseline
```

也就是说：

- 所有正式实验都从 `train_cifar10_experiment.py` 进入
- 不再维护多个独立训练脚本
- 新模型后续都通过“注册到统一入口”的方式接入

## 当前支持的模型

- `vit_baseline`：原始 ViT，使用 learned absolute positional embedding
- `vit_rope`：基础 1D sequence-style RoPE 版本
- `vit_rope_2d`：轻量 2D-aware RoPE 版本，按 `(row, col)` 分别对 `Q/K` 旋转
- `resnet18_scratch`：不加载预训练权重的 ResNet18
- `resnet18_imagenet`：加载 ImageNet 预训练权重的 ResNet18（当前保留为可选参考，不是近期主实验线）

## 当前主实验线

近期默认只保留这 3 条线做正式比较：

- `vit_baseline`
- `vit_rope`
- `resnet18_scratch`

下一步结构实验再加入：

- `vit_rope_2d`

也就是说，当前 `resnet18_imagenet` 不删除，但暂时不放进主实验表，避免把 `ImageNet pretraining` 变成干扰变量。

## 后续新增模型怎么接入

后续如果要加新模型，推荐流程是：

1. 先新建模型文件  
   例如：
   - `vit_rope_2d.py`
   - `vit_local_bias.py`
   - `vit_rope_local.py`

2. 再把它注册进统一入口  
   当前相关模块是：
   - [model_registry.py](/D:/UCL/UCL-dissertation/Postgraduate-dissertation/model_registry.py:1)
   - [train_cifar10_experiment.py](/D:/UCL/UCL-dissertation/Postgraduate-dissertation/train_cifar10_experiment.py:1)

一句话记法：

```text
新模型 = 新模型文件 + 在统一入口注册
```

英文可以记一句：

`Use one experiment runner, register multiple model variants.`

## 训练参数

### 通用参数

所有模型都支持：

- `--model`
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

### ViT 系列参数

- `--embedding-dropout`
- `--attention-dropout`
- `--projection-dropout`
- `--mlp-dropout`

### RoPE 专属参数

- `--rope-base`

说明：
- `--rope-base` 同时适用于 `vit_rope` 和 `vit_rope_2d`

### ResNet18 参数

- `--image-size`

## 默认值

### 通用默认值

- `--epochs 5`
- `--val-ratio 0.1`
- `--num-workers 2`
- `--seed 42`
- `--early-stopping-metric val_acc`
- `--early-stopping-min-delta 0.0`
- `--early-stopping-patience` 默认不开启，需要手动指定

### ViT 默认值

- `--batch-size 128`
- `--lr 3e-4`
- `--weight-decay 0.05`
- `--embedding-dropout 0.0`
- `--attention-dropout 0.0`
- `--projection-dropout 0.0`
- `--mlp-dropout 0.0`

### RoPE 默认值

- `--rope-base 10000.0`

### ResNet18 默认值

- `resnet18_scratch`：
  - `--batch-size 64`
  - `--lr 1e-4`
  - `--weight-decay 0.01`
  - `--image-size 32`

- `resnet18_imagenet`：
  - `--batch-size 64`
  - `--lr 1e-4`
  - `--weight-decay 0.01`
  - `--image-size 224`

## 常用命令模板

运行 ViT baseline：

```bash
python train_cifar10_experiment.py --model vit_baseline --epochs 20 --run-name vit_baseline
```

运行 ViT + RoPE：

```bash
python train_cifar10_experiment.py --model vit_rope --epochs 20 --run-name vit_rope
```

运行 ViT + 2D RoPE：

```bash
python train_cifar10_experiment.py --model vit_rope_2d --epochs 20 --run-name vit_rope_2d
```

运行 ResNet18 scratch：

```bash
python train_cifar10_experiment.py --model resnet18_scratch --epochs 20 --run-name cnn_scratch
```

运行 ResNet18 ImageNet pretrained：

```bash
python train_cifar10_experiment.py --model resnet18_imagenet --epochs 20 --run-name cnn_imagenet
```

说明：
- 当前正式 baseline 对比优先跑 `vit_baseline`、`vit_rope`、`resnet18_scratch`
- `resnet18_imagenet` 先作为参考，不参与近期主结论

运行一个快速 smoke test：

```bash
python train_cifar10_experiment.py --model vit_rope --epochs 1 --train-subset 128 --val-subset 64 --test-subset 64 --num-workers 0 --run-name smoke_vit_rope
```

运行带 early stopping 的实验：

```bash
python train_cifar10_experiment.py --model vit_baseline --epochs 30 --early-stopping-patience 5 --early-stopping-metric val_acc --early-stopping-min-delta 0.001 --run-name vit_baseline_es
```

## 输出内容

训练完成后会保存：

- `results/metrics/<run_name>_metrics.csv`
- `results/metrics/<run_name>_config.json`
- `results/metrics/<run_name>_summary.json`
- `results/metrics/<run_name>_test_confusion_matrix.csv`
- `results/figures/<run_name>_loss.png`
- `results/figures/<run_name>_accuracy.png`
- `results/figures/<run_name>_test_confusion_matrix.png`
- `checkpoints/<run_name>_best.pt`

其中：

- best checkpoint 由 validation 指标选择
- `test` 只用于最终报告，不用于模型选择

英文可以记一句：

`Validation is for selection, test is for final reporting.`

## 对比报告

使用：

```bash
python generate_comparison_report.py --run vit_baseline --run vit_rope --report-name vit_baseline_vs_rope
```

第二轮结构对比可以用：

```bash
python generate_comparison_report.py --run vit_baseline --run vit_rope --run vit_rope_2d --report-name vit_rope_family_compare
```

当前报告层会优先利用这些结构化字段：

- `model_name`
- `model_family`
- `model_variant`
- `position_encoding`
- pretraining / initialization status

适合展示的对比包括：

- `ViT Baseline vs ViT RoPE`
- `ViT Baseline vs ViT RoPE vs ViT RoPE 2D`
- `ViT vs CNN`
- `ResNet18 scratch vs ResNet18 ImageNet pretrained`

输出目录：

```text
results/reports/<report_name>/
```

主要文件：

- `comparison_summary.csv`
- `config_comparison.csv`
- `overview.md`
- `presentation_summary.json`
- `<report_name>.pptx`

## 文件结构

- [train_cifar10_experiment.py](/D:/UCL/UCL-dissertation/Postgraduate-dissertation/train_cifar10_experiment.py:1)：唯一训练入口
- [model_registry.py](/D:/UCL/UCL-dissertation/Postgraduate-dissertation/model_registry.py:1)：模型注册表，定义每个模型怎么接入统一训练入口
- [experiment_utils.py](/D:/UCL/UCL-dissertation/Postgraduate-dissertation/experiment_utils.py:1)：共享训练、评估、保存结果工具
- [cifar10_data.py](/D:/UCL/UCL-dissertation/Postgraduate-dissertation/cifar10_data.py:1)：CIFAR-10 dataloader 构建逻辑
- [vit.py](/D:/UCL/UCL-dissertation/Postgraduate-dissertation/vit.py:1)：原始 ViT baseline
- [vit_rope.py](/D:/UCL/UCL-dissertation/Postgraduate-dissertation/vit_rope.py:1)：基础 RoPE 版本
- [vit_rope_2d.py](/D:/UCL/UCL-dissertation/Postgraduate-dissertation/vit_rope_2d.py:1)：轻量 2D-aware RoPE 版本
- [generate_comparison_report.py](/D:/UCL/UCL-dissertation/Postgraduate-dissertation/generate_comparison_report.py:1)：对比报告与 PPT 生成
- [docs/LEARNING_NOTES.md](/D:/UCL/UCL-dissertation/Postgraduate-dissertation/docs/LEARNING_NOTES.md:1)：学习笔记
- [docs/DEVELOPMENT_MAP.md](/D:/UCL/UCL-dissertation/Postgraduate-dissertation/docs/DEVELOPMENT_MAP.md:1)：研究路线图
- [docs/PROJECT_LOG.md](/D:/UCL/UCL-dissertation/Postgraduate-dissertation/docs/PROJECT_LOG.md:1)：跨设备协作日志

## Git 提醒

不要提交这些目录：

```text
data/
checkpoints/
results/metrics/
results/figures/
```

如果你在两台电脑间切换，推荐顺序是：

1. `git status`
2. 提交当前有价值的代码和文档改动
3. `git push`
4. 到另一台机器 `git pull`
