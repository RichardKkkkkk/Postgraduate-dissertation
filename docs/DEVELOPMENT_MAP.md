# Development Map

## 2026-06-13 New Weekly Direction

老师这周给的要求，已经把近期路线收得更明确了：

- 暂时不把重点放在 multi-seed
- 暂时不继续扩展 `RoPE`
- 先用 `1 seed` 做更多结构分支
- 结果展示必须以 `epoch` 为 x-axis
- 数据上优先准备能区分 horizontal / vertical relationships 的实验

所以当前项目主线已经从：

- `baseline -> RoPE -> 2D RoPE`

切到：

- `baseline -> row-wise sinusoidal -> column-wise sinusoidal`

## Current Research Question

当前更具体的研究问题是：

- 如果模型只拿到 row-wise positional prior，它会不会更适合 horizontal-feature data
- 如果模型只拿到 column-wise positional prior，它会不会更适合 vertical-feature data

这条线比“继续泛化调参”更适合做 clean story。

一句英文概括：

`The immediate question is no longer whether RoPE is better, but whether directional positional priors align with directional image structure.`

## Current Implementation Contract

这一轮实现边界已经锁定：

- 不改 baseline `vit.py`
- 新增独立模型文件接入
- 仍然使用 additive positional embedding
- 仍然保留 `cls token`
- 不引入新的 training tricks
- 不引入 window attention
- 不引入完整 Swin-style hierarchy

也就是说，这一阶段还是 lightweight positional-design study，不是大改 backbone。

## Data Strategy

当前默认两阶段：

### Stage 1

先用 `synthetic_orientation` 跑通方向性假设。

原因：

- 控制变量最干净
- 最容易看出 row / col 是否真的有偏好
- 最适合用 epoch 曲线检查实现是否合理

### Stage 2

如果 synthetic 上有信号，再继续：

- 调研 CADB
- 或找更明显带有 horizontal / vertical structure 的真实图像数据

这个顺序很重要，因为真实数据集往往会同时混入很多别的变化来源。

## Reporting Direction

这周起，主要结果展示方式不再是“只报最后一个 accuracy 数字”，而是：

- `train_loss vs epoch`
- `val_loss vs epoch`
- `train_acc vs epoch`
- `val_acc vs epoch`

必要时再补：

- `test_acc`
- `confusion matrix`
- `macro F1`

因为老师明确强调要从曲线里看：

- 收敛速度
- 是否过拟合
- 是否有异常 spike
- 是否存在可能的实现错误

## Practical Next Steps

当前建议执行顺序已经很明确：

1. 跑 `vit_baseline` on `synthetic_orientation`
2. 跑 `vit_row_sinusoidal` on `synthetic_orientation`
3. 跑 `vit_col_sinusoidal` on `synthetic_orientation`
4. 用统一报告脚本画 epoch curves
5. 看 row / col 谁在对应方向数据上更快收敛或更高准确率
6. 如果有清晰信号，再进入真实数据集阶段

## 2026-06-14 CADB Pilot Layer

现在这条“真实数据集阶段”不再只是想法，已经有了可执行的最小实现：

- `cadb_orientation`

这一步的研究定位不是“完整使用 CADB 的全部标注任务”，而是：

- 先抽出 `horizontal`
- 再抽出 `vertical`
- 做一个 clean binary pilot

这样更贴近老师这周要验证的方向性问题。

### Why this is a good next step

- synthetic 结果已经说明代码逻辑通了
- synthetic 太容易时，很难看出模型差异
- CADB 作为真实图像数据，更适合观察方向性位置先验是否仍然有效

### Boundary of this stage

这一阶段暂时不做：

- composition score regression
- 全 13 类 composition pattern classification
- 多标签复杂任务建模

先做最小问题：

`horizontal vs vertical`

如果这一步有信号，后面再决定是否把 CADB 扩到更多 composition classes。

## What Is Temporarily De-prioritized

当前暂时不是重点的事情：

- multi-seed sweep
- 继续扩展 `2D RoPE`
- 复杂调参
- 大量 augmentation tricks
- 完整 Swin 结构改写

这些东西不是不能做，而是现在做会分散论文主线。
## 2026-06-15 Codebase Structure Cleanup

随着模型分支和数据集分支都开始增加，工程结构本身也变成了一个需要主动控制的变量。

这次整理的原则是：

- 根目录保留“入口”
- `models/` 保留“模型实现”
- `datasets/` 保留“数据读取与 split 逻辑”

这样做的目的不是重构出很复杂的框架，而是让后续新增内容有固定落点。

例如后面如果继续扩展：

- 新 ViT 变体
- 新 positional encoding
- 新真实数据集

就不需要继续把实现文件直接堆在根目录。

一句英文可以这样记：

`Keep entry scripts flat, but group implementations by responsibility.`
