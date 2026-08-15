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
- 当前旧 CSV 仍含逐 epoch test 值，只用于历史开发结果；训练循环已在 2026-08-01 改为 selected-checkpoint-only test protocol

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
- 最终论文统计结果需要用 selected-checkpoint-only test protocol 重新跑

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

## 2026-08-01 Final Holdout Protocol

### 已完成

- `train_cifar10_experiment.py` 已改为 final holdout protocol：
  - 每个 epoch 只评估 train 和 validation
  - validation 负责 checkpoint selection 和 early stopping
  - 训练结束后加载 validation-selected checkpoint
  - test split 只评估一次
- `experiment_utils.py` 的 summary 不再写 `best_test_epoch` / `best_test_acc`
- summary JSON 新增：
  - `test_evaluation_protocol = selected_checkpoint_only`
- `run_seed_sweep.py` 支持最终重跑需要的统一参数：
  - `--dataset`
  - `--all-models`
  - `--exclude-models`
  - `--lr-plateau-patience`
  - `--lr-plateau-factor`
  - `--lr-plateau-min-lr`
- README、`docs/RESEARCH_PLAN.md`、`docs/FIGURE_STANDARD.md` 和 `Agent.md` 已同步最终协议。

### 最终 CIFAR-10 multi-seed 建议协议

- dataset: `cifar10`
- seeds: `42 43 44 45 46`
- epochs: `100`
- batch size: `128`
- learning rate: `3e-4`
- weight decay: `0.05`
- validation split: `val_ratio = 0.1`
- early stopping: `val_acc`, patience `10`, min delta `0.001`
- LR scheduler: ReduceLROnPlateau, patience `5`, factor `0.5`, min lr `1e-6`
- 不使用 train/val/test subset

### 下一步

1. 运行 tiny subset smoke test，确认 metrics CSV 不再包含逐 epoch test 列。
2. 在台式机上启动完整 CIFAR-10 multi-seed sweep。
3. 用统一 publication-style reports 收束论文结果图。

## 2026-08-01 Fixed Split for Final Multi-Seed Sweep

### 已完成

- 新增 `--split-seed`，将 CIFAR-10 train/validation 划分与 training seed 分开。
- 最终 sweep 固定 `split_seed=42`，training seeds 使用 `42 43 44 45 46`。
- ViT、ResNet18 scratch 和 ResNet18 ImageNet 的 CIFAR-10 dataloader 使用同一固定 split。
- config JSON 记录 `split_seed`，方便论文实验审计和复现。

### 当前执行协议

- full CIFAR-10，不使用 subset
- batch size `128`
- learning rate `3e-4`
- weight decay `0.05`
- early stopping: `val_acc`, patience `10`, min delta `0.001`
- final holdout test protocol
- fixed split seed `42`
- training seeds `42-46`

### 新工作站 smoke test 修正

- 后台 smoke test 发现 Matplotlib 自动选择 Tk backend，但新机环境没有可用 Tcl/Tk。
- `paper_plotting.py` 现在显式使用无界面的 `Agg` backend，适合长时间后台训练并继续输出 PNG/PDF。

## 2026-08-05 Thesis Comparison Figures v1

### 已完成

- 核查 `cifar10_final_vit_models_5seeds`：32 个 ViT 配置、每个配置 seeds 42-46，共 160 个 summary。
- 已确认论文候选组使用统一完整 CIFAR-10、固定 `split_seed=42`、batch size 128、learning rate `3e-4`、相同 early stopping 和 `selected_checkpoint_only` test protocol。
- 新增 `generate_thesis_figures.py`，只读取 summary JSON 中的 `selected_model` 指标。
- 生成五张候选论文图：
  - basic PE comparison
  - paired shift effect
  - patch-assignment paired deltas
  - patch-assignment schematic
  - fusion performance and parameter count
- 每组同步输出 per-seed CSV、summary CSV、figure captions 和 manifest。
- fusion 图加入 single-encoder references 和 trainable parameter count，避免忽略 dual-encoder capacity confound。

### 当前结论边界

- 这些图已经可以用于选择论文主文材料，但统计显著性检验尚未加入。
- Patch-order 和 fusion 是两组独立实验；当前没有 `fusion × patch order` factorial experiment。
- 旧 `cifar10_positional_8models_5seeds` summary 没有记录 final selected-checkpoint-only protocol，不用于本轮论文图。

### 下一步

1. 确定四张结果图哪些进入主文，哪些移到 Appendix。
2. 对预先定义的 paired comparisons 决定是否报告 Wilcoxon test、effect size 和区间估计。
3. 后续补充 method overview、PE construction 和 bidirectional fusion architecture schematic。

### Epoch-based comparison update

- 按论文图需求新增五组以 `Epoch` 为横坐标的 multi-seed validation curves：basic PE、shift、patch assignment accuracy、patch assignment loss 和 fusion。
- accuracy 与 loss 均来自逐 epoch validation metrics；没有构造逐 epoch test curve。
- 每条曲线显示五个 seeds 的 mean ± 1 sample SD，并在任一 seed early stop 后停止，保证各 epoch 的样本数固定为五。
- 原有 selected-checkpoint test 图继续保留，用于报告最终 holdout performance、paired effects 与 fusion capacity confound。
## 2026-08-07 Dissertation statistics, mapping audit and fixed Word draft

### Completed

- Audited all 160 summaries in `cifar10_final_vit_models_5seeds`: 32 models, seeds 42–46, fixed `split_seed=42`, and `selected_checkpoint_only` test evaluation.
- Added `generate_thesis_statistics.py` and generated accuracy/loss means, sample SDs, two-sided 95% t intervals, paired seed contrasts and parameter counts.
- Added the radial model to the first core PE table without changing the eight-model confirmatory boundary.
- Extracted `fixed_pos_scale` from all five selected hybrid checkpoints; the value varies around zero and is now reported as exploratory evidence.
- Added deterministic physical-patch → sequence-slot → PE-coordinate mapping tests and a CSV report for all four assignment conventions.
- Added CIFAR-100 dataset registration, loader, 100-class model construction and registration tests.
- Added and passed a final-protocol low-data smoke test. A longer low-data command was stopped before producing a summary, and the formal sweep was then intentionally stopped when the working scope was reduced to the 6 August deliverables. No reduced-data result is included in thesis statistics.
- Added `thesis/tools/build_dissertation.py`. It creates one fixed draft with title page → Abstract → six numbered chapters → unnumbered References, no images or image placeholders, Arial throughout and black text throughout.

### Statistical interpretation frozen for writing

- Five training seeds are not treated as five independent datasets.
- Main claims use paired effects, interval width and direction consistency.
- Exact two-sided Wilcoxon with n=5 cannot attain p<0.05; CD diagrams are descriptive only and are not used for a single-dataset significance claim.
- Negative hybrid, fusion, squared and radial outcomes remain part of the evidence map.

### Current external-state blocker

- `thesis/Yikai_Zhao_MSc_Dissertation.docx` is open in Microsoft Word and is protected by an Office lock file. The rebuilt draft has passed structural font/colour/placeholder checks in a hidden temporary file, but final in-place replacement and Word/PDF/PNG visual QA must wait until the document is closed.

## 2026-08-09 Methodology structure and evidence audit

### Completed

- Replaced the placeholder Methodology chapter in the fixed dissertation file with a nine-section structure covering controlled design, datasets, the shared compact ViT, positional encodings, patch assignment, hybrid/fusion extensions, training, statistics and reproducibility.
- Added verified initial prose from the implementation and final CIFAR-10 configurations; no methodology figures were inserted.
- Added `thesis/METHODOLOGY_EVIDENCE_SHEET.md` to map each methodological statement to code, config or generated statistics.
- Preserved the native ViT Zotero citation and added explicit drafting notes for the missing CIFAR and AdamW records.
- Verified 20 methodology headings, Arial/black formatting in the edited chapter, 19 unique native Zotero citation fields, six existing equations and three existing drawings.

### Protocol mismatch to resolve

- `run_low_data_sweep.py` and the research plan specify learned versus multiplicative PE at learning rate `3e-4`.
- Existing 1k/5k four-model result folders use no PE, learned PE, shifted multiplicative PE and the hybrid at learning rate `1e-3`.
- These results must not be presented as one prespecified protocol. Freeze the final low-data model set and learning rate before finalising Section 3.2.3.

### Next action

1. Review the Methodology chapter one subsection at a time, beginning with Sections 3.1--3.3.
2. Convert the PE definitions in Section 3.4 into numbered Word equations and a compact method table.
3. Add the original CIFAR report and AdamW paper to Zotero before final citation numbering.

## 2026-08-09 Methodology dataset and environment tables

### Completed

- Simplified Section 3.2 into a short dataset rationale, a compact CIFAR-10/CIFAR-100 configuration table, shared preprocessing text and a reduced-data protocol paragraph.
- Added a Section 3.9 workstation/software table recording NVIDIA GeForce RTX 5070 Ti, AMD Ryzen 7 9800X3D, 32 GB RAM, Python 3.11.15, PyTorch 2.12.0+cu130 and Torchvision 0.27.0+cu130.
- Converted all three current table captions to sequential Word `SEQ Table` fields and explicitly enforced Arial black text within every table.
- Preserved 19 native Zotero citation fields, six equations and three existing drawings; no new figure was added.
- Exported the fixed DOCX through Microsoft Word and visually checked the relevant pages. The dataset table now stays together on one page, its headers remain readable, and the environment and results tables have no caption separation or row-splitting problem.

### Evidence still to confirm

- Verify the exact Windows edition/build and confirm that the stated workstation produced every headline result before removing the environment drafting note.
- Freeze the final reduced-data model set and learning rate before reporting that experiment as prespecified.

## 2026-08-09 Methodology method definitions and native equations

### Completed

- Expanded Sections 3.3--3.6 directly from `models/vit.py`, `models/vit_axis_sinusoidal.py`, `models/unfolding.py` and `models/registry.py`.
- Added a shared compact-ViT configuration table and a positional-encoding family table without adding any new figure.
- Added native Word Equations (7)--(16) for the standard sinusoidal schedule, row/column encodings, additive and multiplicative combinations, shifted frequency schedules, squared/radial extensions, patch-to-position mapping, hybrid PE, latent fusion and bidirectional cross-attention.
- Clarified that the implemented shifted variants offset the row/column frequency exponents rather than translating image coordinates.
- Kept the PE-family and workstation tables together on single pages and renumbered all five table captions with Word `SEQ Table` fields.
- Verified the final DOCX structurally and through a 25-page Microsoft Word render: 19 Zotero citation fields, 16 native equations and three pre-existing drawing objects remain present.

### Later refinement

- Inline mathematical variables in the surrounding prose can be standardised during the paragraph-by-paragraph language pass.
- Method diagrams remain intentionally deferred until the experimental suites and final figure set are frozen.

## 2026-08-09 Literature Review Sections 2.4--2.6

### Completed

- Replaced the drafting notes in Sections 2.4--2.6 with reviewed prose on patch ordering and patch-to-position assignment, compact/data-limited ViT training, and the related-work synthesis and controlled-comparison gap.
- Distinguished a consistent permutation of tokens and position vectors from reassigning positional coordinates to physical patches.
- Added and verified literature on Dufter's positional-information overview, LOOPE, the NeurIPS 2025 version of REOrder, DeiT, Compact Transformers, iRPE, CPVT and RoPE-ViT.
- Corrected REOrder from an earlier arXiv-only description to its NeurIPS 2025 publication; LOOPE remains identified as a 2025 preprint.
- Added the new sources to the `UCL-dissertation` Zotero collection and inserted native Zotero citation fields in the fixed Word document.
- Removed the stale Section 2.3.2 drafting note that said CPVT would be discussed later, since Section 2.6 now contains that discussion.
- Preserved the five tables, all native equations and three existing drawing objects; no figure was added.
- Structural QA reports 29 Zotero citation fields in total, Arial/black formatting throughout Sections 2.4--2.6 and a valid DOCX package.
- Rendered the 27-page document through Microsoft Word and visually checked pages 10--14. Headings, citations, page breaks and chapter transition render cleanly.

### Later refinement

- The final bibliography still needs a Zotero refresh after the full citation set is frozen.
- Section 2.3.2 can later receive a dedicated source for the separable two-dimensional sine--cosine construction if the paragraph-by-paragraph literature pass retains that claim.

## 2026-08-09 Experiments and Results structure

### Completed

- Rebuilt Chapter 4 around the logical sequence of the experimental programme rather than the chronological order in which runs were produced.
- Added Sections 4.1--4.8 for evaluation conventions, the core PE comparison, shifted schedules, patch-to-position assignment, limited-data performance, CIFAR-100 generalisation, hybrid/fusion extensions and a factual chapter summary.
- Added seven table frameworks (Tables 5--11) for reporting accuracy, loss, confidence intervals, paired effects, parameter counts and cross-dataset rankings. All numerical cells remain explicitly marked `Pending` until populated from final summary files.
- Removed obsolete Tier labels, provisional numerical claims, the CADB subsection and duplicated experimental prose from the Word draft; source code and result directories were not modified.
- Kept causal interpretation out of Chapter 4 and added drafting notes that map each subsection to its required evidence and interpretation boundary.
- Preserved Arial/black formatting, native Word table numbering, the existing 29 Zotero citation fields, 16 equations and three pre-existing drawings. No new figure was inserted.
- Rendered the fixed document to 29 pages through Microsoft Word and visually inspected the full document and Chapter 4 pages at full resolution. Split tables repeat their header rows and show no clipping, overlap or caption-separation problems.

### Evidence still to resolve

- Freeze the exact fixed multiplicative comparator and learning rate for the reduced-data experiment before filling Table 9; older single-seed results remain exploratory.
- Populate Tables 5--11 only from validation-selected, complete five-seed summaries and keep incomplete CIFAR-100 seed sets out of the main comparison.

## 2026-08-09 Provisional Abstract and reduced-data protocol

### Completed

- Rewrote the Abstract as a 231-word provisional account of the problem, controlled study, evaluated model families, datasets, five-seed evaluation protocol and evidence-bounded contribution.
- Removed all provisional performance claims from the Abstract and added a visible drafting note specifying the four result components to insert after the experimental tables are frozen.
- Set the Abstract to begin on its own page so that the title page remains separate and the full provisional Abstract fits cleanly on page 2.
- Froze the reduced-data comparison as learnable absolute PE versus shifted multiplicative PE at the common learning rate specified in Section 3.7.
- Updated the reduced-data Methodology paragraph, Section 4.5, Table 9 caption, all four comparator rows and the associated evidence note. Earlier single-seed or `1e-3` runs remain explicitly exploratory.
- Preserved 29 Zotero citation fields, the existing equations and three drawing objects; the Abstract remains citation-free, Arial and black.
- Rendered all 29 pages through Microsoft Word and visually checked the complete document, with full-resolution review of the title page, Abstract and both pages of Table 9. No clipping, overlap or broken table continuation was found.

### Remaining Abstract evidence

- Add final numerical sentences only after the CIFAR-10, reduced-data and CIFAR-100 tables use complete comparable runs; then remove the Abstract drafting note.

## 2026-08-09 Methodology evidence and cross-reference pass

### Completed

- Added the exact per-channel CIFAR-10 and CIFAR-100 normalisation constants and stated the channel-wise standardisation formula.
- Clarified that the shared compact ViT uses pre-normalisation residual blocks and recorded both residual updates in Section 3.3.
- Added Word bookmarks to Equations (7)--(16) and replaced manual equation-number mentions with ten live `REF` fields.
- Defined the hybrid coefficient `alpha` as the fixed-position scale, linked it to the implementation name `fixed_pos_scale`, and retained `alpha_0 = 0` as its initial value.
- Added the exact trainable parameter counts for the three latent-fusion variants and two bidirectional cross-attention variants, with the single fixed-PE encoder as the capacity reference.
- Replaced the three Methodology drafting notes with verified CIFAR and AdamW Zotero citations plus an evidence-bounded environment and reproducibility statement.
- Added Windows build and repository base-revision information to Table 4 while explicitly distinguishing the current draft environment from historical per-run provenance.
- Preserved the Methodology structure and prose style; no figure was inserted and no experimental result was changed.
- Structural QA confirms 31 native Zotero citation fields, ten equation `REF` fields, bookmarks for Equations (7)--(16), and no remaining Methodology drafting note.
- Rendered the 29-page document through Microsoft Word and visually checked the complete document plus Methodology pages 14--21 at full resolution. Equation references, tables, citations and the Chapter 4 transition render without clipping or broken pagination.

### Next action

- Populate Chapter 4 tables from frozen, validation-selected summary files before adding any result figure.

## 2026-08-09 Chapter 4 frozen-result table pass

### Completed

- Populated Tables 6--8 from the validation-selected five-seed CIFAR-10 summaries, including trainable parameters, test accuracy, test loss and 95% confidence intervals.
- Populated the Full-data reference rows in Table 9 and retained the 1,000-, 5,000- and 10,000-example rows as `Pending` because their frozen five-seed suite is not yet complete.
- Retained Table 10 as `Pending` rather than mixing an incomplete CIFAR-100 seed set into the cross-dataset ranking.
- Expanded Table 11 to report the order-matched hybrid comparison, all five dual-branch fusion variants, two squared variants and radial PE separately.
- Added paired effects and intervals for shifted variants, learnable versus shifted multiplicative PE, the order-matched hybrid comparison and learnable PE versus the best fusion model.
- Reported the selected-checkpoint hybrid fixed-position scale across all five seeds and kept capacity differences visible for every fusion result.
- Replaced Chapter 4 drafting notes with factual, evidence-bounded observations; unresolved low-data and CIFAR-100 cells remain explicitly marked rather than inferred.
- Restored the automatic `SEQ Table` field for Table 6, updated all eleven table-number fields and kept split-table header repetition.
- Rendered the 30-page document through Microsoft Word and visually inspected Chapter 4 pages 22--27. Table 6 repeats its header after the page break, Table 11 remains readable on one page, captions are numbered correctly, and Section 4.8 begins cleanly on page 27.

### Remaining experiment evidence

- Complete the frozen five-seed 1,000-, 5,000- and 10,000-example comparison before replacing the remaining Table 9 cells.
- Complete all four pre-selected CIFAR-100 models across five seeds before populating Table 10 or making cross-dataset claims.
- Add result figures only after those tables and the final model ranking are frozen.

## 2026-08-09 Formula, figure and dense-table QA

### Completed

- Rechecked all 16 native Word equations against the implemented ViT, positional-encoding, unfolding, hybrid and fusion code paths.
- Replaced the ambiguous element-wise-square superscript in Equation (12) with the conventional square notation while retaining the prose definition of an element-wise operation.
- Clarified the shifted row/column frequency construction, the distinction between physical patch coordinates and assigned PE coordinates around Equation (13), and the one-head versus four-head scope of Equation (16).
- Added the missing set braces and readable modulo spacing in Equation (13), and formatted nearby mathematical variables with italic, subscript and superscript styling.
- Rebuilt the mathematical entries in Table 3 with real subscript/superscript runs; removed stretched justified spacing from Table 11 and raised its minimum font size to 8 pt.
- Enlarged Figure 2 within the text width and standardised all three figure captions as centred, bold italic Arial text.
- Structural QA confirms that all 16 equations, Equation (7)--(16) bookmarks, ten live equation `REF` fields, eleven automatic table-number fields and three inline figures remain intact.

### Visual QA note

- The last successful 30-page Word render was reviewed at full resolution for the formula, figure and dense-table pages before these targeted fixes. A final repeat PDF export was blocked because the current Windows session has no default printer; Word opened the revised document successfully, but its PDF rendering call did not return. The revised DOCX package passes structural validation, and no background Word process was left running.

## 2026-08-10 Terminology and contrast-language pass

### Completed

- Removed the undefined adjective `compact` from the dissertation title and general descriptions of the evaluated ViT; the architecture is now identified directly through its explicit configuration.
- Retained `Compact Transformers` only where it is the formal name of the cited method by Hassani et al.
- Replaced every use of `whereas` with shorter sentences, direct comparison or semicolon-based coordination while preserving the original technical meaning.
- Renamed Section 2.5 to `Data-Efficient and Data-Limited ViT Training` and Section 3.3 to `Shared Vision Transformer Architecture`.
- Updated the title to `A Controlled Evaluation of Positional Encoding in Vision Transformers for Low-Resolution Image Classification` and shortened the running header accordingly.
- Verified the DOCX package and restored the automatic Table 2 numbering field and Equation (13) cross-reference after the run-level wording edit.
- Structural QA confirms 31 Zotero citation fields, ten live equation `REF` fields, eleven automatic table-number fields, three figures and eleven tables.

### Render note

- The bundled DOCX renderer could not run because LibreOffice/`soffice` is not installed in the current workspace runtime. The changes are short inline replacements and passed structural validation; no claim of a new full-page render is made for this pass.

## 2026-08-10 Reduced-data and CIFAR-100 figure package

### Completed evidence

- Verified 60 selected-checkpoint summaries for the four-model CIFAR-10
  reduced-data suite: 1,000, 5,000 and 10,000 training examples, seeds 42--46.
- Verified 20 selected-checkpoint summaries for the four-model CIFAR-100 suite,
  seeds 42--46.
- Added `generate_robustness_figures.py` and generated nine publication-facing
  figure pairs under `results/reports/thesis_robustness_figures_v1/`: five
  selected-test comparisons and four validation-dynamics comparisons.
- Exported per-seed CSV files, mean/sample-SD/95%-CI summaries, the paired
  learnable-minus-shifted-multiplicative contrast, validation-epoch summaries,
  draft captions and a machine-readable manifest.
- Visually checked all nine PNG figures for readable labels, legends, error bars,
  line visibility and clipping. PDF counterparts were generated by Matplotlib.

### Protocol correction

- The completed reduced-data runs use four models and `lr=1e-3`; the older
  `run_low_data_sweep.py` proposal still specifies two models and `lr=3e-4`.
- The existing final CIFAR-10 full-data suite uses `lr=3e-4`, while the completed
  reduced-data and CIFAR-100 suites use `lr=1e-3`.
- The new plots therefore show the internally controlled 1k/5k/10k trend without
  connecting the existing full-data point. CIFAR-100 is presented as a separate
  within-dataset ranking check. A full four-model data-size curve requires a
  protocol-aligned rerun before the full-data point can be added without a
  learning-rate confound.

### Next action

- Select which robustness figures enter Chapter 4 and which remain in the
  appendix, then decide whether the learning-rate mismatch warrants a rerun.

## 2026-08-10 Chapter 4 robustness-figure integration

### Completed

- Inserted three selected figures into the fixed dissertation file: reduced-data
  test accuracy, the seed-matched learnable-versus-shifted-multiplicative
  contrast, and CIFAR-100 test accuracy.
- Populated the 1,000-, 5,000- and 10,000-example rows in Table 9 and all four
  CIFAR-100 rows in Table 10 from the generated five-seed summaries.
- Kept the full-data CIFAR-10 rows as a separately labelled reference because
  they use the earlier `lr=3e-4` protocol; the reduced-data figure does not join
  them to the `lr=1e-3` subset trend.
- Replaced the corresponding Pending text in Sections 4.5, 4.6 and 4.8 with
  factual observations, and updated the related Discussion drafting text so it
  no longer claims that cross-dataset evidence is absent.
- Added live Figure 4--6 sequence fields, bookmarks and cross-references, plus
  descriptive alternative text for all three images.
- Structural QA confirms six figures, eleven tables, 31 Zotero fields, fourteen
  sequence fields and twelve cross-reference fields. The inserted captions and
  result text are Arial and black.

### Visual QA note

- The three source PNGs were inspected at full resolution before insertion.
  The bundled DOCX renderer remains unavailable because LibreOffice/`soffice`
  is not installed. A Microsoft Word fallback export was attempted but blocked
  in the background session, so no new full-document page-render claim is made.

## 2026-08-10 Section 3.4 code-aligned revision

### Completed

- Rewrote Section 3.4 to remove repeated architecture and baseline descriptions
  while keeping the no-PE, learnable, fixed and extension configurations clear.
- Renamed Section 3.4.1 to `No-PE Baseline and Learnable Absolute PE` and made
  the learnable tensor shape, zero initialisation and 8,320-parameter count
  explicit.
- Checked the row, column, additive, multiplicative, shifted, squared and radial
  descriptions against `models/vit_axis_sinusoidal.py`.
- Corrected the shifted-frequency explanation to use exponents `-2i/d` and
  `-(2i+1)/d`, and standardised the embedding-dimension symbol to lower-case
  `d` in Equation (8) and the surrounding prose.
- Restricted Table 3 to the configurations defined in Section 3.4; the hybrid
  row was removed because `alpha` is not introduced until Section 3.6.1.
- Structural QA confirms that Equation (6)--(10) bookmarks and live REF fields,
  31 Zotero fields, six figures and eleven tables remain intact. Revised prose
  runs are Arial and black.

### Render note

- The bundled renderer was attempted after the edit but could not run because
  LibreOffice/`soffice` is not installed. No full-page visual QA claim is made.

## 2026-08-10 Thesis figure visual redesign v2

### Completed

- Rebuilt the main CIFAR-10 and robustness figure packages with high-contrast,
  model-stable colours and line patterns.
- Removed triangle, square and diamond point markers from all multi-seed epoch
  curves because those shapes did not encode an additional variable.
- Presented reduced-data validation accuracy and loss as 1k/5k/10k epoch-based
  small multiples, and separated CIFAR-100 validation accuracy and loss into
  readable epoch figures.
- Retained categorical model axes for selected-checkpoint test summaries because
  the holdout test is evaluated only once per run. Faint circles identify seeds;
  diamonds and vertical bars report the mean and 95% t confidence interval.
- Changed comparison y-axis ranges from forced full scales to padded data-driven
  ranges where appropriate, while keeping the metric and units explicit.
- Generated ten PNG/PDF pairs in `thesis_comparison_figures_v2` and six PNG/PDF
  pairs in `thesis_robustness_figures_v2`; preserved the v1 packages for audit
  and rollback.

### Interpretation rule

- Epoch shading is ± 1 sample SD, not a confidence interval.
- Selected-test error bars are 95% t confidence intervals, not significance
  indicators. Marker shape does not encode checkpoint choice or complexity.

## 2026-08-10 First-round validation-figure integration

### Completed

- Replaced the earlier Chapter 4 selected-test plots with seven validation
  figures in the fixed dissertation file; the selected-checkpoint test tables
  remain the primary result evidence.
- Added validation accuracy and loss for the basic PE comparison, shifted
  variants, patch-to-position assignments, reduced-data study, CIFAR-100 study
  and fusion study. Patch-assignment accuracy and loss remain separate because
  each uses five panels.
- Stacked the reduced-data and CIFAR-100 accuracy/loss source plots vertically
  in `results/reports/thesis_word_figures_v2` so labels remain readable at the
  dissertation's full text width.
- Added live Figure 4--10 sequence fields, bookmarks and cross-references, and
  revised only the adjacent factual result paragraphs needed to introduce the
  figures.
- Kept paired-effect, capacity-accuracy, schematic, confusion-matrix and other
  analysis figures out of this first insertion round.
- Added descriptive alternative text to all ten document figures. Structural
  QA confirms ten inline figures, eleven tables, 31 Zotero fields, 18 sequence
  fields and 17 cross-reference fields; the DOCX ZIP package test also passes.
- All new result text and captions use Arial with black text. Images are inline,
  6.15 inches wide, and kept with their immediately following captions.

### Visual QA note

- All nine v2 source plots and both stacked composite assets were inspected at
  full resolution. Full-page DOCX rendering could not be completed because the
  bundled environment still has no LibreOffice/`soffice`; therefore no claim is
  made that final Word pagination has passed page-render inspection.

## 2026-08-10 Figure and table caption tightening

### Completed

- Shortened all eleven table captions to 5--9 words and all ten figure captions
  to 4--14 words while preserving the existing SEQ fields, bookmarks and REF
  cross-references.
- Moved the repeated five-seed curve convention from every Chapter 4 figure
  caption into Section 4.1: validation curves show the seed mean, shaded bands
  show one sample standard deviation, and curves stop at the last epoch shared
  by the five runs for that configuration.
- Reduced the longest table caption (Table 9) from 31 words to 7 words and the
  new result-figure captions from 53--60 words to 7--14 words.
- Structural QA confirms ten figures, eleven tables, 31 Zotero fields, 18 SEQ
  fields and 17 REF fields. The accessibility audit reports no findings, and all
  caption runs remain Arial and black.

### Render note

- The full-document render was attempted again but remains unavailable because
  LibreOffice/`soffice` is not installed. No page-render claim is made.

## 2026-08-11 Mathematical notation audit

### Completed

- Audited all fourteen numbered equations and the inline mathematical
  expressions in the fixed dissertation file.
- Converted the two shifted-frequency exponents in the explanatory text to
  native Word stacked fractions; all remaining division in mathematical
  expressions is represented by a fraction object rather than a slash.
- Standardised ordinary multiplication: dimensions use `×`, scalar
  multiplication uses conventional juxtaposition where appropriate, and the
  numerical confidence-interval factor uses an explicit multiplication sign.
- Reserved `⊙` for element-wise multiplication, defined the symbol in the
  accompanying prose, and verified its use in Equations (7), (9) and (10).
- Rechecked the implementation-aligned grouping in Equation (10): the squared
  variant applies the exponent to the complete row-column Hadamard product.
- Preserved equation bookmarks `eq1`--`eq14`, all 31 Zotero fields, 18 sequence
  fields, 17 cross-reference fields, ten inline figures and eleven tables. The
  accessibility audit reports no findings, and the DOCX ZIP integrity test
  passes.

### Render note

- Full-page rendering was attempted after the notation edit but remains
  unavailable because LibreOffice/`soffice` is not installed. Structural and
  accessibility checks passed, but no final pagination claim is made.

## 2026-08-12 Protocol-aligned reruns and selected-test figure package

### Completed experiments

- Verified all 60 CIFAR-10 low-data summaries for four models, 1k/5k/10k
  training examples and seeds 42--46 at `lr=3e-4`.
- Verified all 20 CIFAR-100 summaries for four prespecified models and seeds
  42--46 at `lr=3e-4`.
- All 80 new summaries record `test_evaluation_protocol = selected_checkpoint_only`.
- Preserved the earlier `lr=1e-3` result directories as exploratory evidence.

### Generated evidence

- Added `generate_final_test_figures.py` and generated the nine-model core PE
  selected-test table, including accuracy, loss, 95% t intervals and radial PE.
- Generated five main figures: core PE accuracy/loss, patch-assignment delta
  heatmap, low-data accuracy/loss, CIFAR-100 accuracy/loss and fusion
  accuracy-versus-parameter trade-off.
- Generated two supporting figures: paired shifted-variant effects and paired
  per-class recall differences.
- Exported 7 PNG/PDF pairs, source per-seed and summary CSV files, captions,
  a configuration audit and a machine-readable manifest under
  `results/reports/thesis_selected_test_figures_v1/`.
- The full-data/low-data gate passed for learning rate, scheduler, augmentation,
  normalisation, split seed, batch size, weight decay, early stopping, registered
  model identifiers and test protocol. Full-data is therefore connected as the
  45k condition in the new low-data figure.

### Visual and statistical QA

- Visually inspected all seven PNGs at full resolution. Corrected the y-axis
  calculation so it includes complete confidence-interval endpoints, then
  regenerated and rechecked the CIFAR-100 figure.
- Verified seven valid Matplotlib PDF counterparts and 300-dpi PNG dimensions.
- Ordinary intervals use the five-seed t interval with four degrees of freedom;
  paired intervals are calculated from the five seed-level differences.
- The per-class plot is labelled recall; no claim is based on the historical
  `per_class_accuracy` field name.

### Next action

- Review the seven standalone figures. After approval, replace the stale
  `lr=1e-3` low-data and CIFAR-100 values and figures in the dissertation in a
  separate Word render-and-QA pass.

## 2026-08-11 Section 3.4.3 readability revision

### Completed

- Revised the three explanatory paragraphs in Section 3.4.3 while retaining
  Equations (7)--(9), their bookmarks and their live cross-references.
- Gave each paragraph one role: define the additive and multiplicative
  combinations, explain the shifted row/column frequency schedules, and
  distinguish frequency shifting from token ordering and patch-to-position
  assignment.
- Preserved the two native stacked fractions in the shifted-frequency
  explanation and the `⊙` notation for element-wise multiplication.
- Rechecked the wording against `models/vit_axis_sinusoidal.py`; no performance
  advantage or mechanism beyond the implemented construction is claimed.
- Structural QA confirms 31 Zotero fields, 18 sequence fields, 17
  cross-reference fields, ten figures and eleven tables. The mathematical and
  accessibility audits pass.

### Render note

- Full-page rendering was attempted but remains unavailable because
  LibreOffice/`soffice` is not installed. No final pagination claim is made.

## 2026-08-13 Full dissertation audit and evidence check

### Completed

- Reviewed the fixed dissertation file for repeated definitions, repeated
  findings, question-style prose, difficult transitions and inconsistent use
  of baseline/reference terminology.
- Reduced overlap across Sections 2.1--2.3, removed the redundant Chapter 4
  findings recap, simplified the Introduction, Methodology, Results and
  Discussion transitions, and standardised subsection capitalisation.
- Standardised Arial/black heading and table formatting, repeated table header
  rows, prevented table rows from splitting, and added missing alt text to the
  two result figures that lacked it.
- Repaired displayed equations (6)--(12), which Word had placed outside their
  logical pages. They now appear beside the relevant Methodology text with
  right-side numbering. Element-wise multiplication is shown with the circled
  dot symbol, and division is written as a fraction or explicit quotient.
- Checked Tables 6, 8, 9, 10 and 11 against selected-checkpoint formal evidence.
  The audit covers core PE, reduced-data, CIFAR-100, fusion, squared, radial and
  preliminary hybrid results. No numerical mismatches were found.
- Checked the shared ViT, fixed PE, shifted PE, radial, squared and fusion
  descriptions against the current model code and formal seed-42 configuration.
  The stated 4 x 4 patch projection, 8 x 8 grid, 128-dimensional tokens, four
  blocks, four heads, 512-dimensional MLP and reported parameter counts agree.
- Exported the document through Microsoft Word and visually checked all 34
  pages. No clipped tables, missing equations, broken cross-references or
  accessibility findings remain in the reviewed draft.

### Citation note

- Twenty existing Zotero citation fields remain in the document. The local
  Zotero record for *Attention Is All You Need* currently reports the 2023
  arXiv revision year, while the dissertation correctly cites the NeurIPS 2017
  publication. Correct that Zotero item before a final bibliography refresh.

## 2026-08-12 Final thesis evidence figures v1

### Completed

- Added `generate_final_thesis_evidence.py` and generated an independent package
  under `results/reports/thesis_final_evidence_figures_v1/`; the earlier package
  was preserved.
- Generated six main validation-epoch figures: core PE, shifted PE, four-panel
  patch assignment, four-panel low-data, CIFAR-100 accuracy/loss and two-panel
  fusion. Each trajectory is a five-seed mean with pointwise 95% t interval and
  stops at the last epoch shared by all five seeds.
- Generated selected-checkpoint test tables for core PE (Table 6), low-data,
  CIFAR-100 and fusion; the fusion table includes trainable parameter counts.
- Retained four test-only auxiliary analyses: paired shifted effects,
  patch-assignment delta heatmap, fusion accuracy-capacity trade-off and
  per-class recall differences. The per-class figure omits seed points.
- Exported ten 300-dpi PNG/PDF pairs, source CSVs, captions, configuration audit
  and manifest. The protocol gate passed and no test metric is plotted by epoch.

### Next action

- Review the new figures, then update the Word dissertation from this package in
  a separate document render-and-QA pass.

## 2026-08-13 Coordinate-aligned unfolding implementation

### Code audit and implementation

- Confirmed that historical fixed-PE unfolding reordered physical patch tokens
  but added the original row-major fixed PE by sequence slot.
- Preserved that behaviour as the default `sequence_slot` assignment and added
  explicit `coordinate_aligned` handling for row, column, additive,
  multiplicative and radial fixed PE under all four unfolding orders.
- The new implementation reorders only the patch portion of fixed PE with the
  same `patch_order`; the CLS PE entry remains untouched.
- Registered 20 independent models under the compact `vit_ca_` prefix. Results
  remain isolated under directories containing `coordinate_aligned`.

### Phase-one verification

- Added mapping tests for physical coordinate → sequence slot → original PE
  coordinate → assigned PE coordinate. All 256 records for an 8×8 grid satisfy
  assigned PE coordinate equals physical patch coordinate.
- Added same-weight/eval-mode forward equivalence tests and 20 model/unfolding
  smoke combinations. Maximum observed absolute logit error was `8.345e-07`.
- Architecture audit found global unmasked self-attention, token-wise
  LayerNorm/MLP, no sequence convolution, no local/window attention, no
  position-dependent mask and no sequence-adjacent operation.
- Ten unit tests pass, including a regression test that the default remains
  sequence-slot assignment. Both model module checks also pass.
- Completed 20 one-epoch tiny training smoke summaries with
  `position_assignment = coordinate_aligned` and selected-checkpoint-only test.

### Formal experiment plan

- The reuse gate passed for 25 existing normal-row runs across five PE families
  and seeds 42--46: all required training, split, scheduler, early-stopping,
  dropout, subset and selected-test protocol fields match.
- Actual new training requirement is 75 runs: three non-identity unfoldings ×
  five PE families × five seeds. No formal summary completed before automatic
  training was stopped at the user's request.
- `run_coordinate_aligned_unfolding_sweep.py --skip-existing` safely runs or
  resumes the 75 conditions and automatically invokes the result validator.

### Next action

- Run the wrapper command in a terminal. After completion, inspect the generated
  `coordinate_aligned_review` tables before making thesis claims.

## 2026-08-11 Abstract and Introduction result integration

### Completed

- Replaced the provisional Abstract with a 257-word result-bearing version and
  removed its drafting note. The Abstract now follows a problem, controlled
  study, protocol, results and bounded-conclusion sequence without citations.
- Rewrote the Introduction as five focused paragraphs covering the task and
  positional problem, the learnable/fixed trade-off, study scope, evaluation
  design and principal results. Removed the visible result placeholder.
- Added only frozen Chapter 4 findings: the three-model CIFAR-10 core comparison,
  the 1,000-example low-data reversal, the preserved four-model CIFAR-100
  ranking and the capacity-adjusted fusion finding.
- Retained cautious language throughout. The text does not claim statistical
  significance, universal superiority or a new general-purpose architecture.
- Preserved the three Zotero citations used by the Introduction and all document
  structures: 31 Zotero fields, 18 sequence fields, 17 cross-reference fields,
  fourteen equation bookmarks, ten figures and eleven tables. Mathematical and
  accessibility audits pass.

### Render note

- Full-page rendering was attempted but remains unavailable because
  LibreOffice/`soffice` is not installed. No final pagination claim is made.
