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

添加新的可运行脚本时，需要同步更新这个 README。
## Early Stopping

Both training scripts now support early stopping:

```bash
python train_cifar10.py --epochs 30 --early-stopping-patience 5
python train_cnn_cifar10.py --epochs 30 --early-stopping-patience 5
```

Optional controls:

```bash
--early-stopping-metric test_acc
--early-stopping-metric test_loss
--early-stopping-min-delta 0.001
```

Current behavior keeps the project simple by monitoring the existing evaluation
split. For more rigorous experiments, the next step should be switching early
stopping from the test set to a dedicated validation split.

## Experiment Comparison Reports

Use `generate_comparison_report.py` to compare multiple runs and generate:

- overlay plots for every shared numeric metric in `results/metrics/*_metrics.csv`
- a CSV summary of final and best values
- a config-difference CSV built from `*_config.json`
- an `overview.md` summary with meeting-ready takeaway text
- a `presentation_summary.json` file that records the report headline insights
- a meeting-ready `.pptx` deck with:
  - title page
  - experiment setup / run context
  - results overview
  - main metrics comparison
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
  --run cnn_resnet18_baseline="CNN Baseline" \
  --run vit_dropout_01="ViT Dropout 0.1" \
  --report-name cnn_vs_vit_dropout \
  --title "Weekly Comparison: CNN Baseline vs ViT Dropout 0.1"
```

Example that also includes confusion matrix and per-class pages from existing
selected-checkpoint evaluation outputs:

```bash
python generate_comparison_report.py \
  --run cnn_eval_smoke="CNN Eval Smoke" \
  --run vit_eval_smoke="ViT Eval Smoke" \
  --report-name weekly_eval_smoke \
  --title "Weekly Comparison: CNN vs ViT (Eval Smoke)"
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
