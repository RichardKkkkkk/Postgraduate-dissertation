# 研究计划

最近对齐：2026-07-29

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

老师新一轮建议中的 unfolding 实验也已接入。当前将 patch flatten 顺序作为独立变量：

- `normal_row`
- `normal_col`
- `proper_row`
- `proper_col`

第一轮 unfolding 实验只覆盖 5 个模型族：

- baseline
- learnable position
- row sinusoidal
- column sinusoidal
- multiplicative sinusoidal

这样可以先判断 flatten 顺序本身是否值得继续扩展，再决定是否加入 additive、squared multiplicative、radial 等更多 PE。

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

证据完整性说明：

- 当前仓库可直接审计的完整主线是上述八模型五 seed 结果。
- `PROJECT_LOG.md` 记录的 unfolding、hybrid、fusion 和 low-data 实验中，部分原始 metrics/checkpoint 未被 Git 追踪。
- 缺少原始产物的日志结果只能作为方向性记录，不能与完整主线使用同等级证据措辞。

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

## 实验协议状态

文档原则仍然是：

`Validation is for model selection, test is for final reporting.`

2026-07-29 已将训练代码收紧为：

1. 每个 epoch 只计算 train 和 validation 指标。
2. 用 validation 选择唯一 checkpoint。
3. 训练结束后只对该 checkpoint 计算一次 test 指标。
4. 新 metrics CSV 不再包含逐 epoch test 曲线，summary 不再写 `best_test_epoch`。
5. summary 记录 `evaluation_protocol`，config 记录 `test_evaluation`。

旧版 CIFAR-10 八模型五 seed 结果仍包含逐 epoch test 指标。它们只能使用 validation 选定 checkpoint 后记录在 `selected_model` 中的 test 指标，不能按 `best_test_epoch` 选模型。论文最终表格应清楚说明旧协议；若新增方法接近主线结果，再决定是否用新协议重跑核心八模型。

## 下一步

1. 对新 holdout 流程做小 subset smoke test。
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

## 当前结果解释原则

对于 `cadb_elements`，优先看：

- `val_macro_f1`
- `test_macro_f1`
- `per_class_f1`
- `subset_accuracy`

不要只看：

- `acc`

原因是当前 `acc` 在 multi-label 任务里更接近逐标签位准确率，容易因为负样本太多而偏高。

## Hybrid PE 小实验

当前 CIFAR-10 结果显示：

- `vit_learnable_position` 很稳定
- fixed PE 里相对最强的是 `normal_col + multiplicative`
- 但纯 fixed PE 仍然没有稳定超过 learnable PE

因此下一步不直接假设 fixed PE 会超过 learnable PE，而是做一个低成本 hybrid sanity check：

```text
learnable_pos + alpha * fixed_multiplicative_pos
```

其中：

- `learnable_pos` 仍然是标准 ViT 的可学习 positional embedding
- `fixed_multiplicative_pos` 使用之前表现较好的 multiplicative PE
- unfolding 使用之前 fixed PE 里表现最好的 `normal_col`
- `alpha` 是一个可学习标量，并初始化为 0

这样设计的含义是：

- 初始状态等价于普通 learnable PE
- 如果 fixed 2D prior 有帮助，模型可以把 `alpha` 学到非零
- 如果 fixed 2D prior 没有帮助，模型可以接近退回普通 learnable PE

当前新增模型：

- `vit_normal_col_learnable_multiplicative_sinusoidal`

先只跑 CIFAR-10 seed42。若没有超过或接近 `vit_learnable_position`，暂时不扩展 multi-seed。

## Row/Column Latent Fusion

老师下一步建议是把 row-wise 和 column-wise 信息放到 latent representation 层面融合，而不是继续只在
positional encoding 层面手写 `row + col` 或 `row * col`。

第一版模型：

- `vit_row_col_latent_fusion`
- `vit_row_col_mean_fusion`
- `vit_row_col_mean_mlp_fusion`

结构：

```text
image -> row-wise encoder -> row latent
image -> column-wise encoder -> column latent
concat(row latent, column latent) -> fusion MLP -> fused latent -> prediction head
```

Mean fusion baseline:

```text
image -> row-wise encoder -> row latent
image -> column-wise encoder -> column latent
mean(row latent, column latent) -> prediction head
```

Mean + NN fusion baseline:

```text
image -> row-wise encoder -> row latent
image -> column-wise encoder -> column latent
mean(row latent, column latent) -> fusion MLP -> fused latent -> prediction head
```

当前 CIFAR-10 默认维度：

```text
row latent:     (B, 128)
column latent:  (B, 128)
concat latent:  (B, 256)
fusion output:  (B, 128)
logits:         (B, 10)
```

Mean fusion 默认维度：

```text
row latent:     (B, 128)
column latent:  (B, 128)
mean latent:    (B, 128)
logits:         (B, 10)
```

Mean + NN fusion 默认维度：

```text
row latent:     (B, 128)
column latent:  (B, 128)
mean latent:    (B, 128)
fusion output:  (B, 128)
logits:         (B, 10)
```

训练方式：

- 两个 encoder 都看完整同一张图片
- 两个 encoder、fusion module、final head 端到端同时训练
- 每个 batch 只从最终 prediction 计算一个 loss

第一轮只跑 CIFAR-10 seed42。主要比较对象：

- `vit_row_sinusoidal`
- `vit_col_sinusoidal`
- `vit_additive_sinusoidal`
- `vit_multiplicative_sinusoidal`
- `vit_learnable_position`

需要注意：该模型包含两个 ViT encoder，参数量明显大于单 encoder ViT。若结果提升，后续需要考虑参数量
fairness 对照，例如 larger single-encoder ViT。
