# 研究计划

最近对齐：2026-07-20

这个文档回答三个问题：

1. 当前论文真正已经获得了什么证据
2. 哪些模型只是完成实现、还没有正式结果
3. 下一组实验应该按什么协议推进

一句英文概括：

`Keep the training interface stable, compare positional designs under one protocol, and separate implemented ideas from validated evidence.`

## 当前研究问题

- ViT 在图像分类中是否需要 positional encoding
- 不同 positional encoding 是否带来不同的二维空间归纳偏置
- 固定的 row / column / additive / multiplicative 设计能否接近或超过 learned absolute positional embedding
- 更强的 row-column 耦合或 radial distance 是否能进一步改善固定位置编码

## 当前证据主线

当前最可靠的主线已经从早期 CADB 四模型探索，推进到 CIFAR-10 八模型五 seed 对比。

已完成正式五 seed 对比的模型：

- `vit_baseline`：无位置编码对照组
- `vit_learnable_position`：标准 learned absolute positional embedding
- `vit_row_sinusoidal`
- `vit_col_sinusoidal`
- `vit_additive_sinusoidal`
- `vit_additive_sinusoidal_shifted`
- `vit_multiplicative_sinusoidal`
- `vit_multiplicative_sinusoidal_shifted`

已实现但尚未产生正式结果的老师方法扩展：

- `vit_squared_multiplicative_sinusoidal`
- `vit_squared_multiplicative_sinusoidal_shifted`
- `vit_radial_sinusoidal`

其中 radial positional encoding 使用 patch grid 左上角为原点：

```text
r = sqrt(row^2 + col^2)
```

以下模型保留为可选扩展，目前不属于下一组主线实验：

- `vit_rope`
- `vit_rope_2d`
- `resnet18_scratch`
- `resnet18_imagenet`

## 当前数据集定位

### CIFAR-10

CIFAR-10 是当前主要定量证据来源，原因是：

- 数据划分稳定
- 单标签指标清楚
- 已完成八模型、seed 42-46 的同协议比较
- 更适合判断不同位置编码的平均收益和 seed 稳定性

### CADB Elements

CADB Elements 是已经完成的探索性 multi-label 对比，不再作为当前最强结论来源。

标签集合：

- `horizontal`
- `vertical`
- `diagonal`
- `triangle`
- `symmetric`
- `pattern`

已知限制：

- `pattern` 没有有效正样本
- `triangle` 和 `symmetric` 较稀疏
- 普通 `acc` 是逐标签位准确率，会被大量负样本抬高
- row-wise 与 column-wise 在 CADB 上没有清楚支持最初的方向偏置假设

因此 CADB 只用于补充讨论。解释时优先使用 `macro_f1`、`per_class_f1` 和 `subset_accuracy`，不能把逐标签位 `acc` 作为论文主结论。

## 已完成的 CIFAR-10 五 seed 协议

- `dataset = cifar10`
- `seeds = 42, 43, 44, 45, 46`
- `epochs = 100`
- `batch_size = 128`
- `lr = 3e-4`
- `weight_decay = 0.05`
- `val_ratio = 0.1`
- `early_stopping_metric = val_acc`
- `early_stopping_patience = 10`
- `early_stopping_min_delta = 0.001`
- `lr_plateau_patience = 5`
- `lr_plateau_factor = 0.5`
- 所有 dropout 为 `0.0`

实验目录：

```text
results/cifar10_positional_8models_5seeds/
```

## 当前主要结果

五 seed 平均 test accuracy：

- `vit_learnable_position`: `78.854% ± 0.409 pp`
- `vit_multiplicative_sinusoidal_shifted`: `77.638% ± 0.388 pp`
- `vit_multiplicative_sinusoidal`: `77.458% ± 0.485 pp`
- `vit_additive_sinusoidal_shifted`: `76.824% ± 0.345 pp`
- `vit_additive_sinusoidal`: `76.552% ± 0.465 pp`
- `vit_row_sinusoidal`: `74.918% ± 0.185 pp`
- `vit_col_sinusoidal`: `74.574% ± 0.617 pp`
- `vit_baseline`: `71.390% ± 0.567 pp`

当前可支持的判断：

1. CIFAR-10 上位置编码相对于 no-position baseline 有明显收益。
2. learned absolute position 在 5/5 个 seed 上取得最高 test accuracy，仍是当前最佳方案。
3. fixed positional encoding 中，multiplicative shifted 最接近 learned position，平均差距约 `1.216 pp`。
4. multiplicative 系列整体优于单独 row / column 和 additive 系列，支持继续研究 row-column 耦合。
5. 当前没有证据表明任何自定义固定位置编码已经超过 learned position。

## 命名约定

- `vit_baseline` 永远表示无位置编码模型，不再使用 `no_pos` 作为另一个模型名。
- `vit_learnable_position` 表示标准 learned absolute position baseline。
- run name 必须包含模型名和 seed。
- 同一组可比较实验必须共享同一个 `experiment_name`。

下一组实验统一使用：

```text
experiment_name = cifar10_teacher_extensions
run_name = cifar10ext_<model_name>_seed42
```

## 已知实验协议问题

文档原则仍然是：

`Validation is for model selection, test is for final reporting.`

当前训练代码虽然只用 validation 指标做 early stopping 和 checkpoint selection，但仍会在每个 epoch 计算 test，并在 summary 中记录 best test epoch。正式论文实验不应根据这些中间 test 指标选择或描述模型。

在下一轮正式实验前，优先把协议收紧为：

1. 每个 epoch 只计算 train 和 validation 指标。
2. 用 validation 选择唯一 checkpoint。
3. 训练结束后只对该 checkpoint 计算一次 test 指标。
4. 报告不再突出 `best_test_epoch` 或按 epoch 的 test 曲线。

旧结果仍可作为开发与方向判断依据，但论文最终表格应清楚说明旧协议，或在新协议下重跑核心对比。

## 下一步

1. 先修正 test holdout 流程，并做小 subset smoke test。
2. 在 CIFAR-10 seed 42 上正式运行：
   - `vit_squared_multiplicative_sinusoidal`
   - `vit_squared_multiplicative_sinusoidal_shifted`
   - `vit_radial_sinusoidal`
3. 与现有 `vit_learnable_position`、`vit_multiplicative_sinusoidal` 和 shifted 版本比较。
4. 只有接近或超过 `vit_multiplicative_sinusoidal_shifted` 的新模型才扩展到 seed 43-46。
5. 最终确定论文核心模型后，再决定是否按新 holdout 协议重跑完整八模型基线。

## 当前不优先

- 大改 ViT backbone
- 堆叠大量 training tricks
- 直接转向复杂医疗数据集
- 在没有先解决 holdout 和 CADB 指标解释问题前继续扩展大量模型
- 未制定保留策略前删除历史结果或 checkpoint
