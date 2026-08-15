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

## 最终实验协议

文档原则仍然是：

`Validation is for model selection, test is for final reporting.`

当前训练代码已经收紧为 final holdout protocol：

1. 每个 epoch 只计算 train 和 validation 指标。
2. 用 validation 选择唯一 checkpoint。
3. 训练结束后加载 selected checkpoint，只对 test split 评估一次。
4. summary 中记录 `test_evaluation_protocol = selected_checkpoint_only`。
5. 报告不再突出 `best_test_epoch` 或按 epoch 的 test 曲线。

旧结果仍可作为开发与方向判断依据，但论文最终表格应在新协议下重跑。

## 下一步

1. 先用 tiny subset smoke test 确认 final holdout 输出结构。
2. 在完整 CIFAR-10 上按统一协议重跑所有已注册模型：
   - seeds: `42 43 44 45 46`
   - fixed train/validation split seed: `42`
   - epochs: `100`
   - batch size: `128`
   - learning rate: `3e-4`
   - weight decay: `0.05`
   - validation split: `val_ratio = 0.1`
   - early stopping: `val_acc`, patience `10`, min delta `0.001`
   - LR scheduler: ReduceLROnPlateau, patience `5`, factor `0.5`, min lr `1e-6`
3. 用统一图形标准输出每个模型的单模型图、每个 seed 的 comparison 图，以及多 seed mean +/- std。
4. 根据 full-data multi-seed 结果收束 thesis 主线模型，而不是继续无限扩展 exploratory variants。

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
- `vit_row_col_cross_attention_fusion`
- `vit_row_col_cross_attention_mlp_head_fusion`

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

Bidirectional cross-attention fusion:

```text
image -> row-wise encoder -> row token sequence
image -> column-wise encoder -> column token sequence

row-to-column cross attention:
Q = row tokens
K = column tokens
V = column tokens

column-to-row cross attention:
Q = column tokens
K = row tokens
V = row tokens

concat(updated row cls, updated column cls) -> prediction head
```

Cross-attention smoother-head refinement:

```text
same bidirectional cross-attention body
concat(updated row cls, updated column cls) -> LayerNorm -> projection MLP -> prediction
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

Bidirectional cross-attention 默认维度：

```text
row tokens:             (B, 65, 128)
column tokens:          (B, 65, 128)
row-to-column output:   (B, 65, 128)
column-to-row output:   (B, 65, 128)
row cls:                (B, 128)
column cls:             (B, 128)
concat latent:          (B, 256)
logits:                 (B, 10)
```

Smoother-head refinement 默认维度：

```text
concat latent:          (B, 256)
projection head:        (B, 256) -> (B, 128) -> (B, 10)
```

训练方式：

- 两个 encoder 都看完整同一张图片
- 两个 encoder、fusion module、final head 端到端同时训练
- 每个 batch 只从最终 prediction 计算一个 loss
- bidirectional cross-attention 版本保留完整 token sequence，而不是只融合 cls latent

第一轮只跑 CIFAR-10 seed42。主要比较对象：

- `vit_row_sinusoidal`
- `vit_col_sinusoidal`
- `vit_additive_sinusoidal`
- `vit_multiplicative_sinusoidal`
- `vit_learnable_position`

需要注意：该模型包含两个 ViT encoder，参数量明显大于单 encoder ViT。若结果提升，后续需要考虑参数量
fairness 对照，例如 larger single-encoder ViT。

## 论文图与最终重跑的收束规则

后续不再为每组实验临时设计不同的 loss / accuracy 图，统一遵守
`docs/FIGURE_STANDARD.md`。

正式论文的核心曲线：

- 单模型：
  - train loss vs validation loss
  - train accuracy vs validation accuracy
- 单 seed 模型对比：
  - validation loss
  - validation accuracy
  - train loss
  - train accuracy
- 多 seed：
  - epoch mean +/- standard deviation
  - selected checkpoint test metric mean +/- standard deviation

test 不画逐 epoch 曲线。正式重跑前的 gate：

1. 确认训练循环每个 epoch 只评估 validation。
2. 由 validation 选择唯一 checkpoint。
3. 加载 selected checkpoint 后只评估一次 test。
4. 确认 PNG 和 PDF 图都能从同一 metrics 文件重建。
5. 再启动最终模型的 multi-seed 实验。

当前已有 seed42 结果只用于检查图形规范和筛选候选模型，不直接作为最终论文统计结论。

## 论文图的最终分组计划

论文结果不使用 32 模型的单一总排名作为主叙事。主文按研究问题拆为四组：

1. **Basic PE**：No PE、learnable、row-only、column-only、additive、multiplicative。
2. **Shift effect**：additive shifted-minus-unshifted 与 multiplicative shifted-minus-unshifted。
3. **Patch-to-position assignment**：No PE、learnable、row、column 和 multiplicative 在四种 assignment 下的配对变化。
4. **Fusion**：五种 row/column fusion，并加入 row-only、column-only、learnable references 与参数量。

图形证据使用：

- epoch-based validation accuracy / loss curves（mean ± pointwise 95% t CI）
- individual seed points
- mean ± 95% t confidence interval for final five-seed summaries
- paired percentage-point differences when the same seeds are available
- selected-checkpoint-only test accuracy

主文计划保留：

- 一张 experimental design overview
- 一张 PE construction / coordinate convention 图
- 一张 basic PE 结果图
- 一张 shift paired-effect 图
- 一张 patch assignment schematic 与一张 interaction result 图
- 一张 bidirectional fusion architecture 图
- 一张 capacity-aware fusion result 图

主文的 basic PE、shift、patch assignment 和 fusion 对比优先使用 `Epoch`—validation
accuracy/loss 曲线展示训练动态，并用 selected-checkpoint-only test 点图、配对差值或表格
报告最终泛化表现。Patch assignment 的 loss 曲线可在正文空间不足时移至 Appendix。

完整排名、per-class metrics、confusion matrices、非核心模型的全部训练曲线、radial、squared 和 hybrid 默认放 Appendix。后续第二数据集和 low-data 结果只有在协议完整后再增加主文图位。
## 2026-08-07 MSc dissertation execution plan

### Frozen story line

The thesis is a controlled empirical evaluation, not a proposal for a universally new Transformer architecture. Its evidence chain is: PE necessity → row/column construction → patch-to-position correctness → data-regime and dataset generalisation → hybrid/fusion complexity boundaries.

### Experiment priority

1. **Completed evidence consolidation:** 160 CIFAR-10 summaries, five-seed t intervals, paired contrasts, parameter counts and hybrid scale.
2. **Completed implementation check:** deterministic mapping test for `normal_row`, `normal_col`, `proper_row` and `proper_col`.
3. **Prepared, not running on 6 August:** learned versus multiplicative PE at 1k, 5k and 10k training examples, seeds 42–46. Full-data results will be reused from the final experiment.
4. **Prepared:** CIFAR-100 loader and 100-class model interface. Run no PE, learned, shifted additive and shifted multiplicative with seeds 42–46 after the dataset gate.
5. **No additional runs before the draft deadline:** fusion. Interpret it with parameter count and the capacity confound.

### Writing order

Methodology → Experiments and Results → Analysis and Discussion → Literature Review → Introduction → Conclusion → Abstract.

The fixed Word draft follows this structure:

1. Title Page
2. Abstract
3. Introduction
4. Literature Review
5. Methodology
6. Experiments and Results
7. Analysis and Discussion
8. Conclusion
9. References (unnumbered)

No figures or figure placeholders are included in the current draft. Incomplete low-data and cross-dataset suites are described with explicit evidence-status prose and no partial result claims.

## 2026-08-12 Frozen selected-test evidence set

The low-data CIFAR-10 matrix was rerun for four models at 1k, 5k and 10k
training examples using seeds 42--46 and `lr=3e-4` (60 runs). The four-model
CIFAR-100 suite was also rerun with seeds 42--46 and `lr=3e-4` (20 runs). These
80 summaries use the selected-checkpoint-only test protocol.

The final paper-facing result set is now selected-test-first: a nine-model core
table plus core PE, patch assignment, low-data, CIFAR-100, fusion capacity,
paired shift and per-class recall figures. Validation epoch curves may describe
optimisation behaviour but must not determine relative performance claims.

An automated gate confirms that the new 1k/5k/10k experiments and existing
full-data CIFAR-10 references share the learning rate, scheduler, augmentation,
normalisation, split seed, batch size, weight decay, early stopping, registered
model identifiers and test protocol. Full-data may therefore appear as the
fourth point in the new data-size comparison. Across low-data seeds, both the
sampled subset and stochastic training vary; within each seed, all four models
share the same subset.

The earlier `lr=1e-3` low-data and CIFAR-100 results remain exploratory and are
not sources for the frozen thesis table or headline figures. The current Word
draft contains values inserted before the `lr3e4` rerun and must be updated in a
separate, visually verified document pass after the new figures are approved.

## 2026-08-12 Final figure evidence split

The final presentation separates training behaviour from final performance.
Six main figures show five-seed validation trajectories against epoch for core
PE, shifted PE, patch assignment, low-data, CIFAR-100 and fusion. Pointwise
bands are 95% t intervals and each condition stops at its five-seed common epoch.

Selected-checkpoint test accuracy and loss tables remain the source for relative
performance conclusions. Paired shift effects, patch-assignment test deltas,
fusion capacity and per-class recall remain auxiliary test analyses. This avoids
test-over-epoch plots and prevents validation trajectories from being used as
the ultimate ranking evidence.

## 2026-08-13 Coordinate-preserving unfolding correction

The historical unfolding experiment is retained and explicitly described as
sequence-slot assignment: changing token order also changed which fixed PE
coordinate was assigned to a physical patch. A new coordinate-aligned branch
permutes physical patch tokens and their fixed PE vectors together while leaving
the CLS token untouched.

The controlled matrix covers row, column, additive, multiplicative and radial
fixed PE under row-major, column-major and two serpentine orders. Protocol-matched
normal-row runs can be reused because coordinate-aligned and sequence-slot
assignment are identical for the identity permutation. The remaining 75 runs
must be written to `cifar10_coordinate_aligned_unfolding_5seeds` and reported
separately from the legacy assignment statistics.

If trained models differ materially across coordinate-aligned unfoldings, that
is not evidence that the forward function represents different spatial inputs:
the same-weight audit establishes near-equivalence. Such differences would
instead reflect finite optimisation stochasticity. Conversely, forward
equivalence does not imply that independently trained runs must have identical
selected-test metrics.
