# 论文实验绘图规范

最近更新：2026-07-29

这个文档规定单模型、模型对比和多 seed 汇总图的统一格式。后续正式重跑实验时直接复用这套规则，不为每组实验临时改图。

## 0. 统一样式源

所有论文/报告图统一从 `paper_plotting.py` 读取样式。不要在单个脚本里临时写新的颜色、dpi
或字号。

当前样式版本：

```text
PAPER_STYLE_VERSION = 2026-07-29-single-metric-v2
```

统一规则：

- PNG: 300 dpi
- PDF: 同步输出，保留矢量线条
- 主曲线图尺寸：`7.2 x 4.5 inches`
- bar plot 尺寸：`7.2 x 4.2 inches`
- heatmap 尺寸：`6.2 x 5.4 inches`
- 使用色盲友好的高对比调色板
- 同一个模型在不同报告中保持同一颜色
- marker 也尽量按模型固定，避免只依赖颜色区分
- legend 默认放在图下方，减少遮挡曲线

## 1. 单模型训练曲线

每个 run 默认生成两张核心图：

- `<run_name>_loss.png` 和 `<run_name>_loss.pdf`
  - Train loss
  - Validation loss
- `<run_name>_accuracy.png` 和 `<run_name>_accuracy.pdf`
  - Train accuracy
  - Validation accuracy

统一规则：

- 横轴始终是 `Epoch`
- accuracy 统一转换成 `0-100%`
- accuracy 纵轴固定为 `0-100%`
- loss 纵轴从 `0` 开始
- validation 选中的 checkpoint epoch 用灰色竖虚线标出
- 100 epoch 曲线只稀疏显示约 10 组 marker，避免每个点都堆在一起
- PNG 使用 300 dpi，PDF 保留矢量线条，方便插入论文

## 2. 为什么不画 test epoch 曲线

论文协议是：

```text
train each epoch
-> evaluate validation
-> select one checkpoint from validation
-> evaluate test once
```

因此：

- train / validation 可以画随 epoch 变化的曲线
- test 只报告 selected checkpoint 的最终指标
- 不把 test accuracy 或 test loss 画成逐 epoch 曲线
- 不根据 `best_test_epoch` 选择模型

旧实验 CSV 可能仍然包含逐 epoch test 指标，但刷新图时会忽略这些列。当前训练循环已经停止逐 epoch test evaluation；正式论文结果应按新协议重新跑。

## 3. 单 seed 模型对比图

同一张图只比较一个指标：

- `val_loss`
- `val_acc`
- `train_loss`
- `train_acc`

主要结论优先看：

1. Validation loss
2. Validation accuracy
3. Train / validation gap

统一规则：

- 同一个模型在所有报告中使用同一个颜色
- 同时使用 marker 区分曲线，不能只依赖颜色
- 一张图最多放约 8 个模型；更多模型拆成主题组
- 同组模型必须使用相同 dataset、split、epoch 上限、optimizer 和 early-stopping 规则
- 单 seed 图必须在标题、图注或正文中明确标记为 exploratory result

推荐按研究问题拆图：

- PE comparison
- Fusion comparison
- Unfolding comparison
- Low-data comparison

### Publication-style report package

`generate_comparison_report.py` 除了生成单指标曲线，也会为每个 comparison report 生成
一组更接近正式论文的材料：

```text
results/<experiment_name>/reports/<report_name>/figures/val_loss_comparison.png
results/<experiment_name>/reports/<report_name>/figures/val_loss_comparison.pdf
results/<experiment_name>/reports/<report_name>/figures/val_acc_comparison.png
results/<experiment_name>/reports/<report_name>/figures/val_acc_comparison.pdf
results/<experiment_name>/reports/<report_name>/figures/train_loss_comparison.png
results/<experiment_name>/reports/<report_name>/figures/train_loss_comparison.pdf
results/<experiment_name>/reports/<report_name>/figures/train_acc_comparison.png
results/<experiment_name>/reports/<report_name>/figures/train_acc_comparison.pdf
results/<experiment_name>/reports/<report_name>/figures/paper_selected_test_accuracy.png
results/<experiment_name>/reports/<report_name>/figures/paper_selected_test_accuracy.pdf
results/<experiment_name>/reports/<report_name>/publication_selected_checkpoints.csv
results/<experiment_name>/reports/<report_name>/figure_captions.md
```

当前默认保持单张单指标图，不自动拼接 2x2 panel。原因是探索阶段更需要清楚回答：

- validation loss 是否更低
- validation accuracy 是否更高
- train/validation gap 是否变大

如果把多个指标提前拼在一张图里，开会时反而容易让注意力分散。后续写论文排版时，
可以再手动决定是否把几张最终图拼成一个 figure。

`paper_selected_test_accuracy` 只显示 validation-selected checkpoint 的 test accuracy。
这张图回答的是“最后哪一个模型在 held-out test set 上更好”，不回答“哪一个 epoch
test 最高”。

`figure_captions.md` 是图注草稿。正式写论文时需要补充：

- seed 数量
- 数据集 split
- checkpoint selection metric
- 是否为 single-seed exploratory result 或 multi-seed final result

## 4. 多 seed 论文图

多 seed 曲线使用：

```text
solid line = mean
shaded band = mean +/- standard deviation
```

如果不同 seed 因 early stopping 在不同 epoch 结束，某个模型的 mean 曲线只画到所有 seed
都仍有数据的最后一个 epoch。不能在后半段用越来越少的 seed 继续计算 mean。

论文正文或图注必须写清楚：

- seed 列表
- seed 数量
- 阴影代表 standard deviation
- checkpoint selection metric

最终 test accuracy 不画成 epoch 曲线。它应该使用：

- mean ± std 表格，或
- 带 error bar 的点图 / 柱状图

## 5. 文件与目录

单模型图：

```text
results/<experiment_name>/figures/<model>/<run_name>_loss.png
results/<experiment_name>/figures/<model>/<run_name>_loss.pdf
results/<experiment_name>/figures/<model>/<run_name>_accuracy.png
results/<experiment_name>/figures/<model>/<run_name>_accuracy.pdf
```

模型对比图：

```text
results/<experiment_name>/reports/<report_name>/figures/
```

跨 experiment 的临时对比图：

```text
results/reports/<report_name>/figures/
```

跨 experiment 查找依赖全项目唯一的 `run_name`。如果同名 run 出现在多个 experiment，
报告脚本会停止并要求传 `--experiment-name`，避免静默读取错误结果。

## 6. 重画已有结果

刷新某个 experiment 里全部单模型图：

```bash
conda activate vit_research
python refresh_single_run_figures.py --experiment-name <experiment_name>
```

生成单 seed 对比图：

```bash
python generate_comparison_report.py \
  --run <run_name_1>="Model A" \
  --run <run_name_2>="Model B" \
  --metrics val_loss val_acc train_loss train_acc \
  --report-name <report_name> \
  --skip-ppt
```

生成多 seed mean ± std 曲线：

```bash
python summarize_seed_sweep.py \
  --models <model_1> <model_2> \
  --seeds 42 43 44 45 46 \
  --experiment-name <experiment_name> \
  --run-prefix <run_prefix> \
  --report-name <report_name>
```

## 7. 正式重跑前检查

在开始最终 multi-seed 实验前逐项确认：

- 确认训练流程：test 只在训练完成后对 selected checkpoint 评估一次
- summary JSON 记录 `test_evaluation_protocol = selected_checkpoint_only`
- 比较模型使用相同数据 split 和训练超参数
- run name 包含 model、dataset 和 seed
- 单模型 PNG / PDF 可以正常生成
- 对比图的 legend、标题和轴标签没有遮挡
- 多 seed 图的 accuracy 已正确显示为百分比
- 正式图和临时 exploratory 图放在不同 report 目录
