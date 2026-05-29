# ViT Learning

这个项目用于逐步复现 Vision Transformer，并为后续研究“如何让 ViT 更好地保留二维位置信息”打基础。

当前 MVP 使用 CIFAR-10 跑通完整训练流程：数据集、DataLoader、模型前向传播、loss、反向传播、参数更新和测试集评估。

## 运行

本项目开发统一使用已有 conda 环境 `vit_research`。先激活环境：

```bash
conda activate vit_research
```

如果是在新电脑上配置环境，可以在激活环境后安装依赖：

```bash
pip install -r requirements.txt
```

默认会使用完整 CIFAR-10 训练集和测试集：

```bash
python train_cifar10.py
```

默认训练 `5` 个 epoch。想跑更多轮，比如 20 个 epoch：

```bash
python train_cifar10.py --epochs 20
```

如果只是想快速确认代码能跑，可以手动指定小子集：

```bash
python train_cifar10.py --epochs 1 --train-subset 2000 --test-subset 500
```

训练 CNN baseline：

```bash
python train_cnn_cifar10.py
```

默认使用 ImageNet 预训练的 ResNet18。第一次运行如果本地没有权重，`torchvision` 会下载预训练权重。快速 smoke test 可以不用预训练：

```bash
python train_cnn_cifar10.py --weights none --epochs 1 --train-subset 2000 --test-subset 500
```

训练结束后会保存：

- 指标 CSV：`results/metrics/`
- 实验配置 JSON：`results/metrics/`
- 结果摘要 JSON：`results/metrics/`
- loss 和 accuracy 曲线：`results/figures/`

## 文件结构

- `vit.py`：最小 ViT 模型实现。
- `train_cifar10.py`：CIFAR-10 训练和评估脚本。
- `train_cnn_cifar10.py`：CIFAR-10 ResNet18 CNN baseline 训练和评估脚本。
- `docs/LEARNING_NOTES.md`：学习笔记，记录代码解释、PyTorch 语法和 tensor shape。
- `docs/DEVELOPMENT_MAP.md`：项目开发思维导图，记录每一步为什么这样做。
- `docs/PROJECT_LOG.md`：跨设备开发交接日志，记录当前进度、已知问题和下一步。

添加新的可运行脚本时，需要同步更新这个 README。
## Early Stopping

Both training scripts now support early stopping:

```bash
python train_cifar10.py --epochs 30 --early-stopping-patience 5
python train_cnn_cifar10.py --epochs 30 --early-stopping-patience 5
```

Optional controls:

```bash
--early-stopping-metric val_acc
--early-stopping-metric val_loss
--early-stopping-min-delta 0.001
```

## Experiment Comparison Reports

Use `generate_comparison_report.py` to compare multiple runs and generate:

- overlay plots for every shared numeric metric in `results/metrics/*_metrics.csv`
- a CSV summary of final and best values
- a config-difference CSV built from `*_config.json`
- an `overview.md` summary with meeting-ready takeaway text
- a `presentation_summary.json` file that records the report headline insights
- model-aware comparison tables that prioritize:
  - `model_name`
  - `model_family`
  - `model_variant`
  - `position_encoding`
  - initialization / pretraining status
- a meeting-ready `.pptx` deck with:
  - title page
  - experiment setup / run context
  - model comparison page
  - results overview
  - main metrics comparison
  - macro metrics page
  - curve pages
  - confusion matrix / per-class analysis pages when those artifacts exist
  - a short conclusion page

Recommended usage:

- Put the baseline or reference run first in the command so all deltas and meeting conclusions read naturally.
- Use `run_name=Display Label` so the slide text stays readable.
- Re-run the same command each week and only update the `--run` list / `--title`.

Example for a weekly meeting deck:

```bash
python generate_comparison_report.py \
  --run unified_vit_baseline_smoke \
  --run unified_vit_rope_smoke \
  --report-name vit_baseline_vs_rope \
  --title "Weekly Comparison: ViT Baseline vs ViT RoPE"
```

ViT vs CNN example:

```bash
python generate_comparison_report.py \
  --run unified_vit_baseline_smoke \
  --run unified_resnet_smoke \
  --report-name vit_vs_cnn \
  --title "Weekly Comparison: ViT vs CNN"
```

Scratch vs pretrained example:

```bash
python generate_comparison_report.py \
  --run resnet18_scratch_run="ResNet18 Scratch" \
  --run resnet18_imagenet_run="ResNet18 ImageNet" \
  --report-name scratch_vs_pretrained \
  --title "Weekly Comparison: ResNet18 Scratch vs ImageNet Pretrained"
```

Outputs are written to `results/reports/<report_name>/`.

Key output files:

- `comparison_summary.csv`: final / best metric table
- `config_comparison.csv`: differing config values across runs
- `overview.md`: short written summary for notes or email updates
- `presentation_summary.json`: structured summary for downstream tooling
- `<report_name>.pptx`: the weekly meeting deck

## Validation Split

Training now uses a `train / validation / test` workflow.

- `train`: used for parameter updates
- `validation`: used for early stopping and model selection
- `test`: only used for final reporting each epoch and for the selected checkpoint summary

Default behavior:

```bash
python train_cifar10.py --val-ratio 0.1
python train_cnn_cifar10.py --val-ratio 0.1
```

Useful options:

```bash
--val-subset 500
--early-stopping-metric val_acc
--early-stopping-metric val_loss
```

Both scripts now restore the best validation checkpoint before writing the final
summary JSON.

## Extra Evaluation Outputs

Both training scripts now also save selected-checkpoint evaluation artifacts:

- best checkpoint under `checkpoints/<run_name>_best.pt`
- test confusion matrix CSV under `results/metrics/`
- test confusion matrix figure under `results/figures/`
- macro precision / recall / F1 inside `summary.json`

The selected checkpoint is the model chosen by the validation monitoring rule,
not simply the final epoch.

## RoPE Baseline

The original ViT baseline and the RoPE variant are now split into separate
model files so the baseline implementation stays clean:

- `vit.py`: original ViT with learned absolute positional embedding
- `vit_rope.py`: basic RoPE variant

Run them from the same training script:

```bash
python train_cifar10.py --model-variant baseline
python train_cifar10.py --model-variant rope
```

The current RoPE implementation is intentionally minimal:

- it rotates `Q` and `K` inside self-attention
- it leaves the `cls token` unrotated
- it uses a simple 1D sequence-style setup, not a 2D image-aware RoPE yet

## Unified Experiment Runner

项目现在额外提供了一个统一的 CIFAR-10 实验入口：

```bash
python train_cifar10_experiment.py --model vit_baseline
python train_cifar10_experiment.py --model vit_rope
python train_cifar10_experiment.py --model resnet18_scratch
python train_cifar10_experiment.py --model resnet18_imagenet
```

推荐用法：

- 默认优先使用 `train_cifar10_experiment.py`
- 保留 `train_cifar10.py` 和 `train_cnn_cifar10.py` 作为模型专用入口
- 后续新增模型时，优先注册到统一入口，而不是每次再新建一个训练 `main`

示例：

```bash
python train_cifar10_experiment.py --model vit_rope --epochs 20 --run-name vit_rope_clean
python train_cifar10_experiment.py --model resnet18_imagenet --epochs 20 --run-name cnn_ref
```

### 当前支持的模型

- `vit_baseline`：原始 ViT baseline，使用 learned absolute positional embedding
- `vit_rope`：当前最基础的 ViT + RoPE 版本
- `resnet18_scratch`：不加载预训练权重，从随机初始化开始训练的 ResNet18
- `resnet18_imagenet`：加载 ImageNet 预训练权重的 ResNet18

### 通用 CLI 参数

下面这些参数所有模型都支持：

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

### 模型专属参数

ViT 系列：

- `--embedding-dropout`
- `--attention-dropout`
- `--projection-dropout`
- `--mlp-dropout`

RoPE 专属：

- `--rope-base`

ResNet18 系列：

- `--image-size`

### 默认值

通用默认值：

- `--epochs 5`
- `--val-ratio 0.1`
- `--num-workers 2`
- `--seed 42`
- `--early-stopping-metric val_acc`
- `--early-stopping-min-delta 0.0`
- `--early-stopping-patience` 默认不开启，需要你手动指定

ViT 默认值：

- `--batch-size 128`
- `--lr 3e-4`
- `--weight-decay 0.05`
- `--embedding-dropout 0.0`
- `--attention-dropout 0.0`
- `--projection-dropout 0.0`
- `--mlp-dropout 0.0`

RoPE 额外默认值：

- `--rope-base 10000.0`

ResNet18 默认值：

- `--batch-size 64`
- `--lr 1e-4`
- `--weight-decay 0.01`
- `resnet18_scratch` 默认 `--image-size 32`
- `resnet18_imagenet` 默认 `--image-size 224`

### 常用命令模板

运行 ViT baseline：

```bash
python train_cifar10_experiment.py --model vit_baseline --epochs 20 --run-name vit_baseline
```

运行 ViT + RoPE：

```bash
python train_cifar10_experiment.py --model vit_rope --epochs 20 --run-name vit_rope
```

运行 ResNet18 from scratch：

```bash
python train_cifar10_experiment.py --model resnet18_scratch --epochs 20 --run-name cnn_scratch
```

运行带 ImageNet 预训练的 ResNet18：

```bash
python train_cifar10_experiment.py --model resnet18_imagenet --epochs 20 --run-name cnn_imagenet
```

运行一个快速 smoke test：

```bash
python train_cifar10_experiment.py --model vit_rope --epochs 1 --train-subset 128 --val-subset 64 --test-subset 64 --num-workers 0 --run-name smoke_vit_rope
```

运行带 early stopping 的实验：

```bash
python train_cifar10_experiment.py --model vit_baseline --epochs 30 --early-stopping-patience 5 --early-stopping-metric val_acc --early-stopping-min-delta 0.001 --run-name vit_baseline_es
```
