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
