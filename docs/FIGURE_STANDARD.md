# 论文实验绘图规范

最近更新：2026-08-12

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
- 多模型 epoch 曲线使用固定颜色和线型共同区分，不使用无额外含义的 point marker
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
- 单 run 诊断图可以保留完整量程；论文对比图允许使用有边距的动态纵轴以显示方法差异
- validation 选中的 checkpoint epoch 用灰色竖虚线标出
- 单 run 图中的稀疏 marker 只用于区分 train/validation，不表示显著性或 checkpoint
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
- 同时使用固定颜色和线型区分曲线；论文 epoch 图不使用无语义 marker
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

## 8. 论文主结果图的分组规则

最终主结果图由 `generate_thesis_figures.py` 统一生成。每张图只回答一个预先定义的问题：

1. 基础 PE：reference、axis-specific 和 two-axis constructions。
2. Shift：只画 additive 和 multiplicative 的 seed-level paired difference。
3. Patch assignment：以 row-major 为 reference，报告同 seed 下的 percentage-point difference。
4. Fusion：accuracy 与 trainable parameter count 同时报告，明确 dual-encoder capacity confound。

正式图必须：

- 使用 `selected_model.test_acc`，不能使用逐 epoch test peak。
- selected-test 汇总图同时显示五个原始 seed 点和 mean ± 95% t confidence interval。
- 按研究问题固定模型顺序，不能按 test accuracy 自动挑选或重新排序主图。
- 图注写明 training seeds、split seed、checkpoint selection metric 和 final holdout protocol。
- 每张图同步保存 PNG、真正的 Matplotlib PDF、source CSV 和 manifest。

完整 32 模型排名、per-class 图、confusion matrix、全部训练曲线以及 radial、squared、hybrid 等探索性材料默认放 Appendix，不与主验证图混排。

## 9. Epoch-based comparison figures

论文中需要展示 optimisation dynamics 时，统一使用：

- 横坐标：`Epoch`
- 纵坐标：`Validation accuracy (%)` 或 `Validation loss`
- 曲线：五个 training seeds 的逐 epoch mean
- 阴影：逐 epoch 的 95% t confidence interval

由于不同 seed 可能在不同 epoch early stop，主图只画到所有五个 seeds 都仍有记录的
最后一个共同 epoch，不能让曲线后段在没有提示的情况下变成更小的样本数。

逐 epoch 曲线只使用 train/validation metrics。test split 仍只在 validation-selected
checkpoint 上评估一次，并用独立 seed 点、paired difference、表格或 error-bar summary
报告。模型或 assignment 数量过多时使用 small multiples，避免在一个坐标轴中堆叠大量曲线。

### Marker 与不确定性语义

- Epoch 曲线：没有三角形、方形或菱形 marker；颜色和线型表示模型或 assignment。
- Epoch 阴影：五个 seeds 在该 epoch 的 95% t confidence interval。
- Selected-test 半透明圆点：一个 seed 的最终 test 结果。
- Selected-test 菱形：五个 seeds 的均值。
- Selected-test 竖向 error bar：基于 `df = 4` 的 95% t confidence interval。
- 这些形状不表示 statistical significance、checkpoint choice 或 model complexity。

Selected-test 图不能为了形式统一而使用 `Epoch` 横轴，因为 holdout test 只在
validation-selected checkpoint 加载后评估一次。此时横轴应为模型类别，纵轴为数值指标。

## 10. Historical selected-test figure set

`generate_final_test_figures.py` 保留老师最初要求的 selected-test 图集。它生成：

1. core PE test accuracy/loss；
2. patch-assignment paired-delta heatmap；
3. low-data test accuracy/loss；
4. CIFAR-100 test accuracy/loss；
5. fusion accuracy-parameter trade-off；
6. shifted-variant paired effects；
7. per-class paired recall differences。

所有普通汇总使用 `mean ± 2.776445 × sample_SD / sqrt(5)`。配对图先在每个 seed
内计算差值，再对五个差值计算 interval，不能用两个模型各自的 interval 相减。
`test_per_class_accuracy` 在当前 single-label 实现中数值上等于 recall，但论文图必须写
`recall`。普通 confusion matrix、CIFAR-100 100×100 confusion matrix、单独 Macro-F1、
attention map、t-SNE/UMAP 和 calibration 图不属于当前冻结图集。

## 11. Final thesis evidence set

论文最终图表入口为 `generate_final_thesis_evidence.py`。证据分工固定为：Epoch 曲线
只画 validation accuracy 或 validation loss，用于描述训练过程、收敛和 repeatability；
最终模型比较只使用 validation-selected checkpoint 的 test accuracy 和 test loss 表格；
不生成 test metric 随 epoch 变化的图。

逐 epoch 阴影使用：

```text
mean_e +/- 2.776445 * sample_SD_e / sqrt(5)
```

如果 early stopping 导致 seed 长度不同，每个 condition 只画五个 seeds 都存在的共同
epoch 范围。六张主图覆盖 core PE、shifted PE、patch assignment、low-data、CIFAR-100
和 fusion。四张辅助 test 图覆盖 paired shift、patch heatmap、fusion capacity 和
per-class recall；per-class 图只显示 mean 和 paired 95% CI。
