# 项目日志

这个文档记录三件事：

- 最近改了什么
- 当前知道了什么
- 下一步该做什么

## 2026-06-30 项目收束

### 已完成

- 训练入口统一为 `train_cifar10_experiment.py`
- 模型和数据集统一通过 `models/registry.py` 管理
- 当前主线模型包括：
  - `vit_baseline`
  - `vit_learnable_position`
  - `vit_row_sinusoidal`
  - `vit_col_sinusoidal`
  - `vit_additive_sinusoidal`
  - `vit_additive_sinusoidal_shifted`
  - `vit_multiplicative_sinusoidal`
  - `vit_multiplicative_sinusoidal_shifted`
  - `vit_rope`
  - `vit_rope_2d`
- `cadb_elements` 已接成 multi-label 任务

### 当前认识

- CADB 上 row-wise / column-wise 的差异没有像最初预期那样明显
- `pattern` 标签本身不可靠，不能作为强证据
- 当前更需要 clean protocol 和可解释实验，而不是继续堆很多 trick

## 2026-07-01 位置编码扩展

### 已完成

- 接入 additive sinusoidal variants
- 接入 multiplicative sinusoidal variants
- 这些模型都已经注册到统一训练入口里
- 已支持统一生成 comparison report 和 PPT

### 当前认识

- additive / multiplicative 系列已经具备工程可运行性
- 后续重点应放在更干净的数据集设计和更清楚的对照实验上

## 2026-07-02 结果目录对齐

### 已完成

- 保留 `--experiment-name`，让新实验可以按实验名归档
- 结果结构对齐回 CADB 风格：
  - `results/<experiment_name>/metrics/<model>/...`
  - `results/<experiment_name>/figures/<model>/...`
  - `results/<experiment_name>/reports/<report_name>/...`
  - `checkpoints/<experiment_name>/<model>/<run_name>_best.pt`
- 保留对旧版 `results/metrics` / `results/figures` 的读取兼容
- 保留对误建的深层 `runs/...` 目录的读取兼容，避免之前的结果直接失效

### 当前认识

- 你真正想要的是“每次实验一个总文件夹”，不是再往里面加一层很深的 run 目录
- 这样和之前 `cadb_elements_positional_100e` 的使用习惯一致，也更容易人工查看

## 2026-07-02 当时待办（后续状态已更新）

- [x] 用新的 `experiment_name` 口径重新跑一组 CIFAR-10 对比
- [x] 检查 comparison report 是否完整落在同一个 experiment 文件夹下
- [ ] 决定哪些旧结果需要保留，哪些可以清理
- [x] 接入老师要求的 squared multiplicative 和 radial positional encoding
- [ ] 为 squared multiplicative 和 radial positional encoding 生成正式结果

## 2026-07-04 Multi-Seed Summary Plot Update

### 已完成

- `summarize_seed_sweep.py` 现在除了最终 mean/std 表格外，还会输出按 `epoch` 聚合的 mean ± std 曲线
- 新图会对每个模型画均值曲线，并用阴影显示标准差
- 这些曲线直接从每个 run 的 `metrics.csv` 聚合得到
- 后续 multi-seed 汇总不再保留旧版柱状图和按 seed 折线图，主图改为 epoch-wise mean ± std

### 当前认识

- 对老师汇报来说，`epoch-wise mean ± std` 比只看最终柱状图更有解释力
- 这样能直接看到：
  - 收敛速度
  - 波动大小
  - 不同模型是否稳定

## 2026-07-08 Teacher Method Expansion: Squared Multiplicative PE

### 已完成

- 根据老师的新实验建议，新增两个 CIFAR-10 positional encoding 候选模型：
  - `vit_squared_multiplicative_sinusoidal`
  - `vit_squared_multiplicative_sinusoidal_shifted`
- 两个模型都通过 `models/registry.py` 接入统一训练入口
- 修复 `experiment_utils.evaluate()` 在 very small subset smoke test 上的 confusion matrix 类别数推断问题
- squared multiplicative 的核心定义是对 multiplicative PE 再逐元素平方：
  - normal: `(row_pe * col_pe) ** 2`
  - shifted: `(shifted_row_pe * shifted_col_pe) ** 2`

### 当前认识

- 这是对之前 “multiplicative 优于 additive” 结果的直接外推实验
- 平方会去掉乘积的正负号，因此它测试的是更强的 row/column 耦合强度，而不是相位符号
- 两个新模型已经通过 1-epoch CIFAR-10 subset smoke test
- 下一步可以在 CIFAR-10 上跑正式对比

## 2026-07-10 Teacher Method Expansion: Radial PE

### 已完成

- 新增 `vit_radial_sinusoidal`
- 按老师邮件原文实现 radial distance：
  - `r = sqrt(row^2 + col^2)`
  - 原点是 patch grid 左上角 `(0, 0)`
- 模型已接入 `models/registry.py`，可通过统一训练入口运行

### 下一步

- 先跑 CIFAR-10 seed42 单实验
- 如果结果接近或超过 multiplicative / squared multiplicative，再考虑 multi-seed

## 2026-07-20 Windows 工作站环境与文档对齐

### 新工作站环境

- 已创建 Conda 环境 `vit_research`
- Python `3.11.15`
- PyTorch `2.12.0+cu130`
- Torchvision `0.27.0+cu130`
- GPU 为 NVIDIA GeForce RTX 5070 Ti
- 已验证 `torch.cuda.is_available() == True`
- 已在 CUDA 上完成基础 ViT 前向测试，输入 `(8, 3, 32, 32)`，输出 `(8, 10)`
- 15 个注册模型均可正常导入
- 当前 Windows 工作站尚未放置 `data/` 数据目录

### 文档与实际状态的不一致

- README 和研究计划仍把 CADB 四模型写成当前主线，但实际上已经完成 CIFAR-10 八模型、五 seed 对比
- squared multiplicative 和 radial 已实现，但还没有正式结果
- 研究计划中的 `no_pos` / `baseline` 命名与代码现状混用
- README 包含旧电脑绝对路径和一行乱码
- 历史 checkpoint 与 results 实际已被 Git 追踪，与“checkpoint 不提交”的后续规则不完全一致
- 训练代码每个 epoch 都计算 test；虽然 checkpoint 只按 validation 选择，但这不符合最严格的最终 holdout 叙事

### 本次选择的对齐方案

- 当前主要证据改为 CIFAR-10 八模型五 seed 结果
- CADB 改为探索性补充证据，并明确不能用逐标签位 accuracy 作为主结论
- `vit_baseline` 统一表示无位置编码，`vit_learnable_position` 表示标准 learned position baseline
- 下一组实验统一归档到 `cifar10_teacher_extensions`
- 在跑下一组正式实验前，优先修正 test 只在 selected checkpoint 上评估一次的流程
- 未制定 Git LFS 或外部存储方案前，不清理历史结果与 checkpoint

### 当前五 seed 结论

- `vit_learnable_position` 平均 test accuracy 为 `78.854% ± 0.409 pp`，并赢得 5/5 个 seed
- `vit_multiplicative_sinusoidal_shifted` 为 `77.638% ± 0.388 pp`，是当前最接近 learned position 的固定位置编码
- `vit_baseline` 为 `71.390% ± 0.567 pp`
- 当前证据支持位置编码有效，也支持继续研究 multiplicative coupling，但还不支持“自定义固定 PE 超过 learned PE”

### 下一步

1. 修正 test holdout 流程并运行 smoke test
2. 正式运行 squared multiplicative、shifted squared multiplicative 和 radial 的 CIFAR-10 seed 42
3. 只把有竞争力的新模型扩展到 seed 43-46

## 2026-07-12 Teacher Method Expansion: Unfolding Variants

### 已完成

- 新增 `models/unfolding.py`
- 支持 4 种 patch unfolding / flatten 顺序：
  - `normal_row`
  - `normal_col`
  - `proper_row`
  - `proper_col`
- `proper_row` 和 `proper_col` 按老师图片实现为 snake / serpentine 顺序
- 将 unfolding 接入 5 个模型族：
  - baseline
  - learnable position
  - row sinusoidal
  - column sinusoidal
  - multiplicative sinusoidal
- 新增 15 个非默认 unfolding 模型名，原始 5 个模型继续表示 `normal_row`

### 当前认识

- `normal_row` 是当前默认 ViT patch flatten
- baseline 无 positional encoding，理论上对 token 顺序不敏感，可作为 sanity check
- 有 PE 的模型才是 unfolding 实验的主要观察对象

## 2026-07-13 CIFAR-10 Unfolding Seed42 Result Summary

### 已完成

- `cifar10_unfolding_15_seed42` 的 15 个新增 unfolding 实验已完成
- 生成对比报告：
  - `results/cifar10_unfolding_15_seed42/reports/unfolding_seed42_comparison/overview.md`
  - `results/cifar10_unfolding_15_seed42/reports/unfolding_seed42_comparison/comparison_summary.csv`
  - `results/cifar10_unfolding_15_seed42/reports/unfolding_seed42_comparison/figures/`

### 当前认识

- `normal_col` 在这组 seed42 实验里整体最好，特别是 multiplicative PE：
  - normal row multiplicative: 76.61%
  - normal col multiplicative: 77.91%
- proper / snake unfolding 没有稳定提升：
  - proper row multiplicative: 76.17%
  - proper col multiplicative: 75.74%
- baseline 和 learnable position 对 unfolding 顺序基本不敏感，符合 Transformer 对 token permutation 较强不敏感性的预期
- fixed sinusoidal PE 对 unfolding 更敏感，说明 unfolding 主要影响“patch token 和固定 PE 的对应关系”

### 下一步

- 若要继续 unfolding，优先考虑 `normal_col_multiplicative_sinusoidal` 做 multi-seed
- proper row / proper col 暂时不作为优先 multi-seed 候选

## 2026-07-16 Hybrid PE Seed42 Candidate

### 已完成

- 新增 `vit_normal_col_learnable_multiplicative_sinusoidal`
- 该模型组合：
  - `normal_col` patch unfolding
  - learnable absolute positional embedding
  - fixed multiplicative sinusoidal positional embedding
  - 可学习标量 `fixed_pos_scale`
- `fixed_pos_scale` 初始化为 0，因此模型初始状态接近普通 `vit_learnable_position`

### 当前认识

- 这个实验不是假设 fixed PE 一定强于 learnable PE
- 它测试的是：表现最好的 fixed spatial prior 能否作为辅助项，帮助 learnable PE
- 如果 seed42 结果没有接近或超过 learnable baseline，则暂时不扩大到 multi-seed

### 下一步

- 先在 CIFAR-10 上跑 seed42：
  - `vit_normal_col_learnable_multiplicative_sinusoidal`
  - 100 epochs
  - 与已有 `vit_learnable_position` 和 `vit_normal_col_multiplicative_sinusoidal` 对比

## 2026-07-20 Teacher Method: Row/Column Latent Fusion

### 已完成

- 新增 `vit_row_col_latent_fusion`
- 给 `ViTAxisSinusoidal` 增加 `forward_features()`，用于返回 prediction head 之前的 cls latent
- 已通过 `python -m models.vit_axis_sinusoidal` 和 1-epoch CIFAR-10 tiny smoke test
- 新模型包含：
  - row-wise sinusoidal encoder
  - column-wise sinusoidal encoder
  - concat fusion MLP
  - 单个最终 prediction head

### 当前认识

- 两个 encoder 都输入完整图片，并端到端同时训练
- row branch 和 column branch 不分别预测；它们只提供 latent representation
- 当前默认 `embed_dim=128`，因此 concat 后是 `256`，fusion MLP 再压回 `128`
- 该模型参数量大于单 encoder ViT：
  - single row/column sinusoidal ViT: about 0.80M parameters
  - learnable ViT: about 0.81M parameters
  - row/column latent fusion ViT: about 1.80M parameters
- 若结果提升，需要后续做参数量公平性讨论

### 下一步

- 跑 CIFAR-10 seed42：
  - `vit_row_col_latent_fusion`
  - 100 epochs
  - 与 row-wise、column-wise、multiplicative、learnable position 对比

## 2026-07-21 CIFAR-10 Low-Data Seed42 Result Summary

### 已完成

- 完成 `cifar10_low_data_seed42`
- 生成 low-data vs full-data 对比图：
  - `results/cifar10_low_data_seed42/reports/low_data_vs_full_comparison/figures/low_data_vs_full_selected_test_accuracy.png`
- 训练集大小：
  - 1000
  - 5000
  - 10000
- 对比模型：
  - `vit_learnable_position`
  - `vit_normal_col_multiplicative_sinusoidal`
  - `vit_row_col_latent_fusion`

### Selected test accuracy

| train subset | learnable | normal_col multiplicative | row/column latent fusion |
| --- | ---: | ---: | ---: |
| 1000 | 36.10% | 40.62% | 40.92% |
| 5000 | 54.16% | 56.55% | 54.69% |
| 10000 | 63.06% | 62.74% | 63.01% |

### 当前认识

- 低数据量下 fixed structural PE 确实更有竞争力：
  - 1000 samples 时，normal_col multiplicative 和 latent fusion 都明显超过 learnable
  - 5000 samples 时，normal_col multiplicative 仍明显超过 learnable
  - 10000 samples 时三者基本打平
- 这支持一个更清楚的研究问题：
  - learnable PE 在 full-data setting 很强
  - structured 2D PE 可能在 low-data setting 提供更有用的 inductive bias
- 目前只有 seed42，不能直接作为最终结论，需要 multi-seed 验证

### 下一步

- 优先对 low-data CIFAR-10 做 multi-seed，而不是继续新增模型
- 建议先扩大：
  - `vit_learnable_position`
  - `vit_normal_col_multiplicative_sinusoidal`
  - 可选 `vit_row_col_latent_fusion`
- 若算力有限，优先 multi-seed 1000 和 5000 两档，因为这两档最能体现 structured PE 的潜在优势

## 2026-07-23 Teacher Fusion Variant: Mean + Prediction

### 已完成

- 新增 `vit_row_col_mean_fusion`
- 该模型复用 row-wise encoder 和 column-wise encoder
- 两个 encoder 都输出 prediction head 之前的 cls latent
- fusion 方式从 concat/MLP 改为逐元素平均：
  - `fused_latent = (row_latent + col_latent) / 2`
- 平均后的 latent 直接进入 shared prediction head

### 当前认识

- 当前默认 `embed_dim=128`
- row latent 和 column latent 都是 `(B, 128)`
- mean fusion 后仍是 `(B, 128)`，不会像 concat fusion 一样变成 `(B, 256)`
- 这个模型参数量会小于 `vit_row_col_latent_fusion`，适合作为 fusion 方法的简单 baseline

### 下一步

- 先跑 CIFAR-10 seed42：
  - `vit_row_col_mean_fusion`
  - 100 epochs
  - 与 no PE、row、column、learnable、concat fusion 放在同一张 validation loss 图里比较
- 后续再实现：
  - mean + NN + prediction
  - bidirectional cross-attention + prediction

### CIFAR-10 seed42 result

- Experiment: `cifar10_fusion_variants_seed42`
- Run: `row_col_mean_fusion_seed42`
- Early stopped at epoch 50
- Selected checkpoint epoch: 40
- Selected validation accuracy: 77.04%
- Selected test accuracy: 76.14%
- Best observed test accuracy: 76.68%

Initial interpretation:

- Mean fusion improves over the previous concat + MLP latent fusion result:
  - concat + MLP selected test acc: 75.48%
  - mean fusion selected test acc: 76.14%
- It also improves over individual row-wise and column-wise PE models:
  - row-wise selected test acc: 74.79%
  - column-wise selected test acc: 74.33%
- It is still below learnable PE:
  - learnable selected test acc: 78.88%
- This suggests row/column fusion is useful, but simple mean fusion is not enough to replace learnable PE on full CIFAR-10.

## 2026-07-23 Teacher Fusion Variant: Mean + NN + Prediction

### 已完成

- 新增 `vit_row_col_mean_mlp_fusion`
- 该模型先复用 mean fusion：
  - `fused_latent = (row_latent + col_latent) / 2`
- 然后把平均后的 latent 送入一个 fusion MLP：
  - `LayerNorm(128)`
  - `Linear(128, 512)`
  - `GELU`
  - `Linear(512, 128)`
- 最后用 shared prediction head 输出最终预测

### 当前认识

- 这是老师提出的第二个 fusion variant
- 它测试的问题是：直接平均 row/column latent 后，再用一个小 NN 做非线性重整，是否比 pure mean fusion 更好
- 与 concat + MLP fusion 相比，它不会把 fusion 输入扩大到 256 维，因此参数更少，也更公平

### 下一步

- 跑 CIFAR-10 seed42：
  - `vit_row_col_mean_mlp_fusion`
  - 100 epochs
  - 与 `vit_row_col_mean_fusion` 和 `vit_row_col_latent_fusion` 比较 validation loss、training loss、training accuracy、validation accuracy

### CIFAR-10 seed42 result

- Experiment: `cifar10_fusion_variants_seed42`
- Run: `row_col_mean_mlp_fusion_seed42`
- Early stopped at epoch 52
- Selected checkpoint epoch: 42
- Selected validation accuracy: 76.90%
- Selected test accuracy: 76.40%
- Best observed test accuracy: 76.56%

Initial interpretation:

- Mean + NN fusion is close to pure mean fusion, but does not clearly improve it:
  - mean fusion selected test acc: 76.14%
  - mean + NN selected test acc: 76.40%
  - mean fusion best test acc: 76.68%
  - mean + NN best test acc: 76.56%
- Mean + NN has higher selected validation loss than pure mean fusion:
  - mean fusion selected val loss: 0.7541
  - mean + NN selected val loss: 0.8237
- The extra MLP may add flexibility, but the current single-seed result does not show a stable advantage over simple averaging.

## 2026-07-27 Teacher Fusion Variant: Bidirectional Cross Attention + Prediction

### 已完成

- 新增 `vit_row_col_cross_attention_fusion`
- 给 `ViTAxisSinusoidal` 增加 `forward_tokens()`，用于返回完整 token sequence：
  - CIFAR-10 默认 shape: `(B, 65, 128)`
- 新增 `MultiHeadCrossAttention`
- 新增 `CrossAttentionBlock`
- 新模型包含：
  - row-wise encoder
  - column-wise encoder
  - row-to-column cross-attention block
  - column-to-row cross-attention block
  - concatenated cls-token prediction head

### 结构

```text
image -> row encoder -> row_tokens: (B, 65, 128)
image -> col encoder -> col_tokens: (B, 65, 128)

row_to_col:
Q = row_tokens
K = col_tokens
V = col_tokens

col_to_row:
Q = col_tokens
K = row_tokens
V = row_tokens

concat(row_cross_cls, col_cross_cls) -> (B, 256)
prediction head -> (B, 10)
```

### 当前认识

- 这个模型比 mean/concat fusion 更接近老师的结构图
- 它不是在最终 cls latent 层面简单融合，而是在完整 token sequence 层面让 row/column representation 互相查询
- 当前版本保持 `embed_dim=128`，所以 concat 后是 256 维；如果后续需要参数量公平性，可以再做 half-dim version

### 下一步

- 运行模型 shape test 和 tiny smoke test
- 再跑 CIFAR-10 seed42 100 epochs：
  - `vit_row_col_cross_attention_fusion`
  - 与 concat fusion、mean fusion、mean + NN fusion、row、column、learnable 一起画 validation loss 对比图

### CIFAR-10 seed42 result

- Experiment: `cifar10_fusion_variants_seed42`
- Run: `row_col_cross_attention_fusion_seed42`
- Early stopped at epoch 64
- Selected checkpoint epoch: 54
- Selected validation accuracy: 78.82%
- Selected test accuracy: 77.21%
- Best observed test accuracy: 77.60%

Initial interpretation:

- Bidirectional cross-attention is the strongest fusion variant so far:
  - concat + MLP selected test acc: 75.48%
  - mean fusion selected test acc: 76.14%
  - mean + NN selected test acc: 76.40%
  - cross-attention selected test acc: 77.21%
- It also improves over individual row-wise and column-wise PE:
  - row-wise selected test acc: 74.79%
  - column-wise selected test acc: 74.33%
- It is still below learnable PE:
  - learnable selected test acc: 78.88%
- The model overfits more strongly than simpler fusion variants:
  - selected validation loss: 0.9936
  - selected test loss: 1.0173
  - final train acc: 98.33%
  - final val acc: 77.98%
- Current conclusion: cross-attention provides the clearest evidence that row/column interaction helps, but full-data CIFAR-10 still favors learnable PE and the cross-attention version likely needs regularization or parameter-count discussion.

## 2026-07-28 Cross-Attention Refinement: Smoother Prediction Head

### 已完成

- 新增 `vit_row_col_cross_attention_mlp_head_fusion`
- 主体与 `vit_row_col_cross_attention_fusion` 完全一致
- 唯一变化是最后 prediction head：

```text
Original:
concat(row_cls, col_cls) -> Linear(256, 10)

Smoother head:
concat(row_cls, col_cls) -> LayerNorm(256) -> Linear(256, 128) -> GELU -> Linear(128, 10)
```

### 当前认识

- 这个 refinement 不改变训练协议，也不加 dropout
- 它测试的问题是：cross-attention 后两个方向的 cls token 是否需要在分类前再做一次非线性融合
- 如果有效，可以说明问题不只在 row/column token interaction，也在最终 fused latent 的分类方式

### 下一步

- 先跑 shape test 和 tiny smoke test
- 再跑 CIFAR-10 seed42：
  - `vit_row_col_cross_attention_mlp_head_fusion`
  - 与原始 `vit_row_col_cross_attention_fusion` 对比 validation loss、selected test acc 和 overfitting 程度

### CIFAR-10 seed42 result

- Experiment: `cifar10_fusion_variants_seed42`
- Run: `row_col_cross_attention_mlp_head_fusion_seed42`
- Early stopped at epoch 57
- Selected checkpoint epoch: 47
- Selected validation accuracy: 78.30%
- Selected test accuracy: 77.15%
- Best observed test accuracy: 77.63%

Comparison with original cross-attention fusion:

| Model | selected val acc | selected val loss | selected test acc | selected test loss | best test acc | final train acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Cross-attention direct head | 78.82% | 0.9936 | 77.21% | 1.0173 | 77.60% | 98.33% |
| Cross-attention MLP head | 78.30% | 0.8168 | 77.15% | 0.8298 | 77.63% | 97.11% |

Initial interpretation:

- Smoother head does not clearly improve selected test accuracy:
  - direct head: 77.21%
  - MLP head: 77.15%
- It slightly improves best observed test accuracy:
  - direct head: 77.60%
  - MLP head: 77.63%
- It meaningfully reduces validation/test loss and final train accuracy:
  - this suggests slightly better calibration or less overconfident overfitting
- Current conclusion: smoother head helps stability/loss more than accuracy. It is useful as a refinement result, but not a new best selected checkpoint.

## 2026-07-28 Paper Figure Standardization

### 已完成

- 新增 `paper_plotting.py`，统一三条绘图链路：
  - 单模型训练图
  - 单 seed 模型对比图
  - 多 seed mean +/- std 图
- 新增 `docs/FIGURE_STANDARD.md`
- 单模型 loss / accuracy 图默认只展示 train 和 validation
- selected checkpoint epoch 继续由 validation 指标决定并用竖线标出
- 所有核心曲线同时输出：
  - 300 dpi PNG
  - vector PDF
- accuracy 统一显示为 `0-100%`
- 多 seed accuracy 的 mean 和 std 都会乘以 `100`
- `generate_comparison_report.py` 默认核心曲线收束为：
  - `val_loss`
  - `val_acc`
  - `train_loss`
  - `train_acc`
- `summarize_seed_sweep.py` 不再生成逐 epoch test 曲线

### 研究协议决定

- test 不作为逐 epoch 论文曲线
- 正式论文只报告 validation-selected checkpoint 的最终 test 指标
- 当前旧 CSV 仍含逐 epoch test 值，只用于历史开发结果；最终 multi-seed 重跑前还要修改训练循环本身

### 下一步

1. 用已有 seed42 结果生成 PE comparison 和 fusion comparison 的规范化预览图
2. 检查 legend、颜色、轴尺度和 PDF 输出
3. 修改正式训练流程，使 test 只在训练结束后评估一次
4. 确定最终模型清单后开始 multi-seed 重跑

### 预览验证

- 已生成 fusion seed42 预览：
  - `results/reports/draft_fusion_curves_seed42/`
- 已生成 PE seed42 预览：
  - `results/reports/draft_pe_curves_seed42/`
- 已用旧的 CIFAR-10 八模型五 seed 结果验证 mean +/- SD：
  - `results/cifar10_positional_8models_5seeds/reports/paper_style_preview_8models_5seeds/`
- 发现并修复跨 experiment 对比时找不到 run 的问题
- 多 seed 曲线现在只保留所有 seed 都有数据的 epoch，避免 early stopping 后样本数逐渐减少

## 2026-07-29 Publication-Style Comparison Outputs

### 已完成

- `generate_comparison_report.py` 现在会额外生成 publication-style report package：
  - 单指标 comparison curves：
    - `val_loss_comparison.png/.pdf`
    - `val_acc_comparison.png/.pdf`
    - `train_loss_comparison.png/.pdf`
    - `train_acc_comparison.png/.pdf`
  - `figures/paper_selected_test_accuracy.png/.pdf`
  - `publication_selected_checkpoints.csv`
  - `figure_captions.md`
- 暂时不自动生成 2x2 拼接图，避免探索阶段一张图承担太多结论
- `paper_selected_test_accuracy` 只汇总 validation-selected checkpoint 的 test accuracy
- `figure_captions.md` 提供可用于 thesis / supervisor email 的图注草稿

### 目的

- 让结果展示更接近正式论文，而不是一组临时训练截图
- 每组 comparison report 自动同时给出：
  - 单张训练过程图
  - 单张泛化过程图
  - final held-out test summary
  - 可复用 caption

### 下一步

- 已用 fusion seed42 结果生成单张 comparison 图预览：
  - `results/cifar10_fusion_variants_seed42/reports/draft_fusion_single_figures_seed42/`
- 当前图形策略收束为：
  - 每张图只比较一个指标
  - 不自动生成 2x2 拼接图
  - selected-test 图只显示 validation-selected checkpoint 的 test accuracy
- 正式 multi-seed 前继续优先修正 test 只在 selected checkpoint 后评估一次的问题

## 2026-07-29 Code/Docs Alignment

### 已完成

- README 的 CLI 参数列表与 `train_cifar10_experiment.py` 对齐：
  - 补充 synthetic dataset 参数
  - 将 `--image-size` 归到通用参数
- `docs/PROJECT_STRUCTURE.md` 与当前代码/目录结构对齐：
  - 日期更新到 2026-07-29
  - 区分正式历史结果目录和 exploratory seed42 / smoke 结果目录
  - report 目录树补齐 comparison PDF 输出
- 确认 comparison report 代码不再生成 `paper_training_dynamics`

### 当前仍需注意

- `results/` 下多数新结果目录仍未纳入正式论文协议，不建议 commit
- 最终 multi-seed 前仍要修改训练流程，让 test 只在 selected checkpoint 后评估一次

## 2026-07-29 Plot Style Standard v2

### 已完成

- `paper_plotting.py` 升级为统一画图样式源：
  - `PAPER_STYLE_VERSION = 2026-07-29-single-metric-v2`
  - 统一主曲线图、bar plot、heatmap 尺寸
  - 统一 300 dpi PNG + PDF 输出
  - 使用更适合论文的高对比、色盲友好调色板
  - 模型颜色和 marker 尽量保持稳定
- 单模型辅助图也接入统一标准：
  - selected per-class metrics
  - confusion matrix
- comparison report 辅助图也接入统一标准：
  - macro metric snapshot
  - grouped per-class metrics
  - generated confusion matrix
- `analyze_per_class_report.py` 不再单独使用 `dpi=200`，并将 per-class 指标统一显示为百分比

### 目的

- 后续所有实验图默认拥有相同格式、色调、字号和导出格式
- 减少论文写作阶段反复手动改图的成本
- 保持探索图和最终论文图之间的视觉规则一致
