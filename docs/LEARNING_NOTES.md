# 学习笔记

这个文档用来记录：

- 关键 PyTorch 语法
- 当前项目里容易混淆的实现点
- 读代码时值得记住的概念

## 1. `Dataset` 和 `DataLoader`

在这个项目里，数据流分成两层。

### `Dataset`

作用是定义：

- 第 `i` 个样本是什么

例如：

```python
image, label = dataset[index]
```

### `DataLoader`

作用是：

- 把 `Dataset` 里的样本按 batch 取出来
- 控制 `shuffle`
- 控制 `num_workers`

训练时你看到的：

```python
for images, labels in loader:
    ...
```

这里的 `loader` 就是 `DataLoader`。

## 2. 单标签和多标签的区别

### 单标签任务

一张图只属于一个类别。  
例如：

- `horizontal`
- `vertical`

这种情况下模型常见输出是：

- 一个 `logits` 向量
- 最后用 `argmax` 取一个类

### 多标签任务

一张图可以同时属于多个标签。  
例如当前 `cadb_elements`：

- `horizontal`
- `vertical`
- `diagonal`
- `triangle`
- `symmetric`
- `pattern`

这种情况下模型输出的是一个向量，每一维都单独判断某个标签是否出现。

## 3. 为什么 `cadb_elements` 的 `acc` 看起来会偏高

当前 `cadb_elements` 用的是 multi-label 指标。  
日志里的 `acc` 不是“整张图完全预测正确”的准确率，而更接近：

- 逐标签位的 0/1 准确率

如果负样本很多，`acc` 容易显得不低。  
所以当前更值得看的指标是：

- `macro_f1`
- `per_class_f1`
- `subset_accuracy`

## 4. `subset_accuracy` 是什么

`subset_accuracy` 的意思是：

- 一张图的所有标签都必须预测对
- 才算这张图正确

它比普通 multi-label 的 `acc` 更严格。

## 5. 为什么 `pattern` 结果一直是 0

不是模型单独不会学 `pattern`，而是当前原始标注里：

- `pattern` 字段虽然存在
- 但没有有效正样本

所以：

- train 里没有 `pattern = 1`
- test 里也没有 `pattern = 1`

这属于数据标注问题，不是模型 bug。

## 6. 当前模型分支怎么理解

### `vit_baseline`

- 没有 positional encoding
- 用来回答“位置编码本身有没有帮助”

### `vit_learnable_position`

- learned absolute positional embedding
- 这是当前标准 ViT baseline

### `vit_row_sinusoidal`

- patch token 的位置只看 row
- 同一行共享位置向量

### `vit_col_sinusoidal`

- patch token 的位置只看 column
- 同一列共享位置向量

### `vit_radial_sinusoidal`

- patch token 的位置不再分别看 row / column
- 先计算每个 patch 到左上角原点 `(0, 0)` 的距离：

```python
radial_positions = torch.sqrt(row_positions.pow(2) + col_positions.pow(2))
```

- 再把这个距离送进 sinusoidal PE：

```text
PE_2i   = sin(radial_position / scale_i)
PE_2i+1 = cos(radial_position / scale_i)
```

- 对 CIFAR-10 来说，`image_size=32`、`patch_size=4`，所以 patch grid 是 `8 x 8`
- `row_positions` shape 是 `(64,)`
- `col_positions` shape 是 `(64,)`
- `radial_positions` shape 也是 `(64,)`
- 最终 patch positional embedding shape 是 `(64, embed_dim)`

### `vit_additive_sinusoidal`

- patch token 的位置同时看 row 和 column
- 位置向量由 `row_pe + col_pe` 构成
- 它更接近一种 factorized 2D absolute positional encoding

### `vit_additive_sinusoidal_shifted`

- 也是 additive 结构
- 但 row 和 column 使用错开的 wavelength
- 目标是减少 row / column 使用同一频率时的重合

### `vit_multiplicative_sinusoidal`

- patch token 的位置同时看 row 和 column
- 位置向量由 `row_pe * col_pe` 构成
- 它比 additive 更强调 row / column 的耦合

### `vit_multiplicative_sinusoidal_shifted`

- 也是 multiplicative 结构
- 但 row 和 column 使用错开的 wavelength
- 用来测试“错开频率后再耦合”是否更稳定

### `vit_squared_multiplicative_sinusoidal`

- 先计算普通 multiplicative PE：`row_pe * col_pe`
- 再对每一维平方：`(row_pe * col_pe) ** 2`
- 在 PyTorch 里写成 `.pow(2)`
- 这个版本会丢掉原始乘积的正负号，只保留强度大小

### `vit_squared_multiplicative_sinusoidal_shifted`

- 先使用 shifted multiplicative PE
- 再做逐元素平方
- 用来测试“错开频率 + 更强耦合强度”是否比普通 shifted multiplicative 更好

## 7. `registry.py` 的角色

`models/registry.py` 可以理解成项目的“接线板”。

它负责：

- 注册模型名
- 选择 dataloader
- 提供默认超参数

如果以后要加新模型，通常都要先改这里。

## 8. Patch unfolding / flatten 方式

当前 `PatchEmbedding` 先用卷积把图片切成 patch tokens：

```python
x = self.proj(x)
x = x.flatten(2)
x = x.transpose(1, 2)
```

这一步默认得到的是 `normal_row` 顺序。之后如果需要其他 unfolding，会用：

```python
x = x.index_select(1, patch_order)
```

这里 `x` 的 shape 是：

```text
(batch_size, num_patches, embed_dim)
```

`dim=1` 表示重排 patch token 这一维。

对一个 `4 x 4` patch grid，四种顺序是：

```text
normal_row:
0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15

normal_col:
0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15

proper_row:
0, 1, 2, 3, 7, 6, 5, 4, 8, 9, 10, 11, 15, 14, 13, 12

proper_col:
0, 4, 8, 12, 13, 9, 5, 1, 2, 6, 10, 14, 15, 11, 7, 3
```

这里的 proper 指蛇形展开，目的是减少一行或一列结束后跳回另一侧造成的大距离跳跃。

## 9. 当前结果管理约定

现在项目里区分两类结果。

### 原始结果

放在：

- `results/`

作用：

- 本地分析
- 中间产物
- 可以随时重跑

注意：历史上为了跨设备备份，当前三组正式实验和 56 个 checkpoint 已经被 Git 追踪。`.gitignore` 不会自动取消已追踪文件；不要未经确认直接清理。

### 长期文档

放在：

- `docs/`

作用：

- 记录研究路线

## 10. 小 subset smoke test 的 confusion matrix

做很小的 CIFAR-10 smoke test 时，`val-subset` 或 `test-subset` 可能没有覆盖 10 个类别。  
所以 confusion matrix 不能用当前 subset 里出现过几个 label 来决定大小。

更稳的方式是从模型输出读类别数：

```python
num_classes = logits.shape[1]
```

例如 CIFAR-10 的 logits shape 是：

```text
(batch_size, 10)
```

所以即使某个小 subset 里没有出现第 9 类，confusion matrix 也仍然应该是 `10 x 10`。

## 11. Validation 和 Test 的职责

- `train` 用来更新模型参数
- `validation` 用来选择 checkpoint、early stopping 和调整学习率
- `test` 只用来评估已经由 validation 选定的模型

如果每个 epoch 都查看 test 曲线或报告 `best_test_epoch`，即使代码没有直接用 test 做 early stopping，也可能在人工分析时形成 test leakage。

论文正式协议应当是：

```text
train each epoch
-> evaluate validation
-> select one checkpoint from validation
-> evaluate test once
```

当前训练代码还会每个 epoch 计算 test，这是已经记录在研究计划中的待修正事项。

## 12. Learnable + fixed PE 的写法

新的 hybrid PE 写法是：

```python
x = x + self.pos_embed + self.fixed_pos_scale * fixed_pos_embed
```

这里三个张量/参数的含义是：

- `x`
  token 序列，shape 是 `(batch_size, num_patches + 1, embed_dim)`
- `self.pos_embed`
  可学习 positional embedding，shape 是 `(1, num_patches + 1, embed_dim)`
- `fixed_pos_embed`
  固定的 multiplicative sinusoidal PE，shape 也是 `(1, num_patches + 1, embed_dim)`
- `self.fixed_pos_scale`
  一个可学习标量，shape 是 `(1,)`

PyTorch 会自动 broadcast：

```text
(1,) -> (1, 1, 1)
```

所以 `fixed_pos_scale * fixed_pos_embed` 会给整个 fixed PE 乘上同一个可学习权重。

这个标量初始化为 0：

```python
self.fixed_pos_scale = nn.Parameter(torch.zeros(1))
```

这表示模型一开始几乎等价于普通 learnable PE。训练过程中，如果 fixed PE 有帮助，反向传播会更新
`fixed_pos_scale`，让 fixed PE 分支参与更多；如果没帮助，它可以继续接近 0。

`register_buffer` 的作用是把 fixed PE 放进模型状态里，但不让 optimizer 更新它：

```python
self.register_buffer("fixed_pos_embed", full_fixed_pos_embed, persistent=False)
```

它和 `nn.Parameter` 的区别是：

- `nn.Parameter` 会被训练
- `register_buffer` 会跟着模型移动到 CPU/GPU，但默认不参与训练

## 13. Row/Column latent fusion 的张量流

`vit_row_col_latent_fusion` 不是让 row encoder 只看图片的行、column encoder 只看图片的列。
两个 encoder 都看完整图片，区别只在 positional encoding：

```text
row encoder:    image + row-wise PE
column encoder: image + column-wise PE
```

为了拿到 prediction head 之前的表示，`ViTAxisSinusoidal` 新增了：

```python
def forward_features(self, x):
    ...
    cls_output = x[:, 0]
    return cls_output
```

这里 `cls_output` 是整张图片的 latent representation。

在当前 CIFAR-10 默认设置中：

```text
input image:    (B, 3, 32, 32)
row_latent:     (B, 128)
col_latent:     (B, 128)
```

拼接使用：

```python
fused_latent = torch.cat([row_latent, col_latent], dim=1)
```

因为 `dim=1` 是 feature 维，所以：

```text
(B, 128) + (B, 128) -> (B, 256)
```

然后 fusion MLP 做 projection：

```text
(B, 256) -> (B, 512) -> (B, 128)
```

最后 prediction head 输出：

```text
(B, 128) -> (B, 10)
```

训练时只要不写 `detach()`，也不包 `torch.no_grad()`，最终 loss 会自动反向传播到：

- row encoder
- column encoder
- fusion MLP
- final prediction head

## 14. Row/Column mean fusion 的张量流

`vit_row_col_mean_fusion` 是老师提出的第一个新 fusion baseline：

```text
image -> row encoder -> row_latent
image -> column encoder -> col_latent
mean(row_latent, col_latent) -> prediction head
```

当前 CIFAR-10 默认维度：

```text
row_latent:    (B, 128)
col_latent:    (B, 128)
fused_latent:  (B, 128)
logits:        (B, 10)
```

代码里的 mean fusion 是：

```python
fused_latent = (row_latent + col_latent) / 2
```

这里不是 batch 维度取平均，而是 feature 逐元素平均。比如第 `j` 个 feature：

```text
fused_latent[:, j] = (row_latent[:, j] + col_latent[:, j]) / 2
```

因为输出仍然是 `(B, 128)`，所以它可以直接进入：

```python
self.head = nn.Linear(embed_dim, num_classes)
```

相比 concat fusion：

```python
torch.cat([row_latent, col_latent], dim=1)
```

mean fusion 不会把 feature 维度从 `128` 增加到 `256`，因此它参数更少，也更适合作为 fusion 方法的简单 baseline。

## 15. Row/Column mean + MLP fusion 的张量流

`vit_row_col_mean_mlp_fusion` 对应老师说的第二个模型：

```text
Mean & NN & Prediction
```

它先和 mean fusion 一样得到平均 latent：

```python
fused_latent = (row_latent + col_latent) / 2
```

此时：

```text
fused_latent: (B, 128)
```

然后进入一个小的 fusion MLP：

```python
self.fusion = nn.Sequential(
    nn.LayerNorm(embed_dim),
    nn.Linear(embed_dim, fusion_hidden_dim),
    nn.GELU(),
    nn.Dropout(mlp_dropout),
    nn.Linear(fusion_hidden_dim, embed_dim),
    nn.Dropout(projection_dropout),
)
```

当前 CIFAR-10 默认设置下：

```text
(B, 128) -> LayerNorm(128) -> (B, 128)
(B, 128) -> Linear(128, 512) -> (B, 512)
(B, 512) -> GELU -> (B, 512)
(B, 512) -> Linear(512, 128) -> (B, 128)
(B, 128) -> prediction head -> (B, 10)
```

`nn.Sequential` 的作用是把多个层按顺序串起来。调用：

```python
fused_latent = self.fusion(fused_latent)
```

就等价于依次执行 LayerNorm、Linear、GELU、Dropout、Linear、Dropout。

这个模型和 concat + MLP fusion 的区别是：

- concat + MLP：`(B, 128) + (B, 128) -> (B, 256) -> MLP -> (B, 128)`
- mean + MLP：`(B, 128) 和 (B, 128) 先平均 -> (B, 128) -> MLP -> (B, 128)`

所以 mean + MLP 的 fusion network 输入更小，参数也更少。

## 16. Bidirectional cross-attention fusion 的张量流

`vit_row_col_cross_attention_fusion` 对应老师说的第三个 fusion model：

```text
Bidirectional Cross Attention & Prediction
```

它和前几个 fusion model 最大的区别是：不只融合最终的 cls latent，而是先保留完整 token sequence。

当前 CIFAR-10 默认设置：

```text
image:      (B, 3, 32, 32)
patch size: 4
grid:       8 x 8
tokens:     64 patch tokens + 1 cls token = 65
embed dim:  128
```

row encoder 和 column encoder 分别输出：

```text
row_tokens: (B, 65, 128)
col_tokens: (B, 65, 128)
```

### Row-to-column cross attention

第一条 cross-attention branch：

```text
Q = row_tokens
K = col_tokens
V = col_tokens
```

代码里等价于：

```python
row_cross_tokens = self.row_to_col(row_tokens, col_tokens)
```

含义是：用 row representation 去 column representation 里查询有用信息。

### Column-to-row cross attention

第二条 cross-attention branch：

```text
Q = col_tokens
K = row_tokens
V = row_tokens
```

代码里等价于：

```python
col_cross_tokens = self.col_to_row(col_tokens, row_tokens)
```

含义是：用 column representation 去 row representation 里查询有用信息。

### Cross attention 的内部 shape

`MultiHeadCrossAttention` 里：

```python
q = self.q_proj(query_tokens)
kv = self.kv_proj(context_tokens)
```

如果输入是：

```text
query_tokens:   (B, 65, 128)
context_tokens: (B, 65, 128)
num_heads:      4
head_dim:       32
```

reshape 后：

```text
q: (B, 4, 65, 32)
k: (B, 4, 65, 32)
v: (B, 4, 65, 32)
```

attention map：

```text
q @ k.transpose(-2, -1): (B, 4, 65, 65)
```

最后输出：

```text
cross attention output: (B, 65, 128)
```

### CrossAttentionBlock 的结构

`CrossAttentionBlock` 使用和普通 ViT block 类似的 pre-norm residual 结构：

```python
query_tokens = query_tokens + cross_attn(norm(query_tokens), norm(context_tokens))
query_tokens = query_tokens + mlp(norm(query_tokens))
```

所以 residual connection 保留的是 query branch：

```text
row-to-column residual: row_tokens
column-to-row residual: col_tokens
```

### 最终预测

两个方向更新完成后取 cls token：

```python
row_cls = row_cross_tokens[:, 0]
col_cls = col_cross_tokens[:, 0]
```

shape 是：

```text
row_cls: (B, 128)
col_cls: (B, 128)
```

然后拼接：

```python
fused_latent = torch.cat([row_cls, col_cls], dim=1)
```

得到：

```text
fused_latent: (B, 256)
```

最后 prediction head：

```text
Linear(256, 10) -> logits: (B, 10)
```

## 17. Cross-attention fusion 的 smoother head

`vit_row_col_cross_attention_mlp_head_fusion` 是对 cross-attention fusion 的一个小 refinement。

主体不变：

```text
row encoder
column encoder
row-to-column cross attention
column-to-row cross attention
concat(row_cls, col_cls) -> (B, 256)
```

唯一变化是最后的 prediction head。

原始 cross-attention fusion:

```python
self.head = nn.Linear(embed_dim * 2, num_classes)
```

当前 CIFAR-10 默认是：

```text
Linear(256, 10)
```

smoother head 版本:

```python
self.head = nn.Sequential(
    nn.LayerNorm(embed_dim * 2),
    nn.Linear(embed_dim * 2, embed_dim),
    nn.GELU(),
    nn.Linear(embed_dim, num_classes),
)
```

当前 CIFAR-10 默认 shape:

```text
(B, 256) -> LayerNorm(256) -> (B, 256)
(B, 256) -> Linear(256, 128) -> (B, 128)
(B, 128) -> GELU -> (B, 128)
(B, 128) -> Linear(128, 10) -> (B, 10)
```

为什么叫 smoother head：

- 原始 head 直接把两个 cls token 拼接后分类
- smoother head 先把 `(B, 256)` 投影回 `(B, 128)`
- 中间的 GELU 给分类前的融合增加一点非线性

这个实验不改变 row/column encoder 和 cross-attention block，只测试最终分类头是否太直接。

## 18. 论文曲线为什么要统一尺度

训练代码中的 accuracy 使用 `0-1`：

```python
{"val_acc": 0.7882}
```

论文图使用百分比，所以画图时统一乘以 `100`：

```python
plotted_accuracy = raw_accuracy * 100.0
```

因此：

```text
0.7882 -> 78.82%
```

多 seed 图需要同时缩放 mean 和 standard deviation。假设：

```text
mean = 0.7882
std  = 0.0041
```

图中应显示：

```text
78.82% +/- 0.41 percentage points
```

不能只给 mean 乘 `100` 而遗漏 std，否则阴影带会缩小 100 倍。

## 19. mean +/- std 曲线的含义

对于相同模型的多个 seed，在每个 epoch 分别计算：

```python
epoch_mean = mean(seed_values)
epoch_std = stdev(seed_values)
```

图中：

- 中心线是 `epoch_mean`
- 阴影下界是 `epoch_mean - epoch_std`
- 阴影上界是 `epoch_mean + epoch_std`

这张图同时表达：

- 平均训练趋势
- 不同随机初始化带来的波动

它不能代替最终 test 表格。最终 test 仍然应该报告 validation-selected checkpoint 的
`mean +/- std`。

如果五个 seed 的 early stopping 结束时间不同，例如：

```text
seed 42: 66 epochs
seed 43: 61 epochs
seed 44: 58 epochs
seed 45: 64 epochs
seed 46: 60 epochs
```

那么五 seed mean 曲线应画到 epoch 58。epoch 59 以后已经不是五个 seed 的平均，继续画会让
曲线含义发生变化。

## 20. 为什么单模型图不再包含 test 曲线

即使 test 指标没有参与 `loss.backward()`，每个 epoch 都查看 test 曲线也会影响人工选择：

```text
"这个 epoch 的 test 更高，所以选它"
```

这仍然属于 test information leakage。

所以论文图只画：

```text
Train
Validation
```

selected epoch 由 validation 决定，test 只对这个 checkpoint 评估一次并报告最终数值。

## 21. 为什么现在先保持单张单指标图

正式论文里有时会把相关指标拼成一个 multi-panel figure，例如：

```text
(a) train loss       (b) validation loss
(c) train accuracy   (d) validation accuracy
```

但是当前项目还在模型探索和开会讨论阶段，更适合先保持单张单指标图：

- `val_loss_comparison.png` 专门回答泛化损失
- `val_acc_comparison.png` 专门回答验证准确率
- `train_loss_comparison.png` 专门看优化过程
- `train_acc_comparison.png` 专门看训练集拟合程度

这样做的好处是每张图只承担一个结论，给老师看结果时不需要在同一张图里同时解释
四个指标。等最终模型和数据集收束之后，如果论文排版需要，再手动把几张最终图拼成
multi-panel figure。

现在 `generate_comparison_report.py` 默认不自动生成拼接图。

## 22. 为什么要把画图样式集中到 `paper_plotting.py`

如果每个脚本自己写：

```python
plt.figure(figsize=(8, 6))
plt.savefig(path, dpi=150)
```

后面会出现一个问题：同一篇论文里的图看起来像从不同项目里复制来的。

现在项目把这些规则集中在 `paper_plotting.py`：

```python
PAPER_STYLE_VERSION = "2026-07-29-single-metric-v2"
PAPER_FIGSIZE = (7.2, 4.5)
PAPER_DPI = 300
MODEL_COLORS = {...}
```

这样以后不管是：

- 单模型 loss / accuracy
- 模型对比曲线
- multi-seed mean +/- std
- per-class bar plot
- confusion matrix

都会使用同一套字号、颜色、marker、legend 和 PNG/PDF 输出规则。

这不是改变实验结果，只是统一 visual presentation。论文里这很重要，因为图的风格统一会让读者把注意力放在结果差异上，而不是被格式差异干扰。

## 23. Final holdout protocol 在代码里怎么体现

正式论文实验现在遵守：

```text
train each epoch
evaluate validation each epoch
select checkpoint by validation
load selected checkpoint
evaluate test once
```

这对应到代码结构是：

```python
for epoch in range(1, args.epochs + 1):
    train_metrics = train_one_epoch(...)
    val_metrics = evaluate(...)
    history.append({
        "epoch": epoch,
        "train_loss": train_metrics["loss"],
        "train_acc": train_metrics["acc"],
        "val_loss": val_metrics["loss"],
        "val_acc": val_metrics["acc"],
    })

model.load_state_dict(best_state_dict)
selected_test_metrics = evaluate_with_details(...)
```

注意 `history` 里没有 `test_loss` / `test_acc`。这说明 test 不参与每个 epoch 的观察，也不参与
checkpoint 选择。test 只作为最终报告指标，存到 summary JSON 的 `selected_model` 下面。

从实验设计角度看，validation 像“开发过程中的裁判”，test 像“最后一次考试”。如果每个 epoch
都看 test，就算没有用它反向传播，也可能会影响我们人工选择模型，这会让最终结果偏乐观。

## 24. `argparse` 里的开关参数

`run_seed_sweep.py` 里新增了：

```python
parser.add_argument("--all-models", action="store_true")
```

这个写法表示：命令行里只要出现 `--all-models`，`args.all_models` 就是 `True`；如果没有出现，
就是 `False`。

例如：

```bash
python run_seed_sweep.py --all-models --seeds 42
```

## 24. 为什么 training seed 和 split seed 要分开

正式 multi-seed 实验固定 `split_seed=42`，让所有模型和所有 training seed 使用同一组
CIFAR-10 train/validation indices。`seed=42..46` 仍然控制模型初始化、数据增强和训练顺序。

```text
split_seed = 42       -> 数据划分保持不变
training seed = 42-46 -> 重复训练中的随机性发生变化
```

这样 multi-seed 标准差主要反映训练随机性，而不是把不同数据划分的变化混进结果。

后台训练不需要打开绘图窗口。统一绘图模块使用：

```python
import matplotlib
matplotlib.use("Agg")
```

`Agg` 会直接把图渲染到 PNG/PDF，避免服务器或后台进程因为 Tk GUI/Tcl 环境缺失而失败。

代码里可以这样判断：

```python
model_names = list(EXPERIMENT_REGISTRY.keys()) if args.all_models else list(args.models)
```

意思是：

- 如果用了 `--all-models`，就从 registry 自动拿全部模型
- 如果没用，就只跑 `--models` 后面手动指定的模型

`--exclude-models` 则是在选好的模型列表里删掉某些模型，例如先排除可能需要下载权重的
`resnet18_imagenet`。

## 25. 为什么论文图要显示原始 seed 点和 paired difference

只有五个 seeds 时，单独画 mean ± standard deviation 会隐藏数据的实际分布。论文图同时显示：

```text
translucent circles = individual seeds
diamond             = mean
error bar           = 95% t confidence interval
```

如果两种方法都使用相同的 seed 列表，可以先按 seed 对齐，再计算：

```python
delta_seed = 100 * (shifted_test_acc - unshifted_test_acc)
```

这里乘以 `100` 后，单位是 percentage points。配对差值直接回答“同一 seed 条件下改变设计后发生了什么”，比比较两个独立 error bar 更清楚。

Patch assignment 图同样以 row-major 为 reference：

```python
delta_vs_row_major = 100 * (alternative_order_acc - row_major_acc)
```

注意，这里的配对表示使用相同 training seed 和固定 split；它不能被解释成纯模型初始化方差，也不能自动证明因果或统计显著性。

最终结果必须从 summary JSON 的以下位置读取：

```python
summary["selected_model"]["epoch"]
summary["selected_model"]["test_acc"]
```

顶层 `best_val_epoch` 是历史 validation 曲线上的原始最高点，可能与考虑 early-stopping `min_delta` 后实际加载的 selected checkpoint 不同，因此不能混用。

## 26. 为什么 Epoch 曲线使用 validation 而不是 test

横坐标为 epoch 的曲线适合回答模型如何收敛、是否过拟合以及不同方法的 optimisation
dynamics 是否不同。正式流程中每个 epoch 记录 train 和 validation 指标，因此论文对比曲线使用：

```text
x = epoch
y = validation accuracy or validation loss
band = mean ± pointwise 95% t CI across five seeds
```

test set 是最终 holdout，只在 validation-selected checkpoint 加载后评估一次。逐 epoch
绘制 test accuracy/loss 会把 test set 带入训练过程中的反复观察，因此不应为了图形形式而增加这种评估。

Early stopping 会使不同 seed 的曲线长度不同。如果后段仍对剩余 seed 求均值，读者可能把
样本数变化误认为模型趋势。当前论文图采用共同 epoch 截断：只有当五个 seeds 在该 epoch
都有记录时才计算 mean 和 sample SD。
## 27. Patch sequence order is not the same as PE assignment

For a physical patch `(row, col)`, record three separate objects:

```text
physical patch coordinate -> sequence slot -> assigned PE coordinate/vector
```

A joint permutation of patch tokens and their matched PE vectors is a relabelling for full self-attention. Changing which fixed vector is attached to a physical patch changes the positional information presented to the model. Therefore, an accuracy change must not be attributed vaguely to “destroyed token adjacency” unless the exact mapping has been verified.

## 28. Five-seed 95% intervals and paired contrasts

For five training seeds, use the sample standard deviation and four degrees of freedom:

```text
half_width = 2.776445 * sample_SD / sqrt(5)
95% CI = mean +/- half_width
```

For two models with the same seeds, calculate the per-seed difference first, then summarise those five differences. This removes much of the shared seed-to-seed variation and directly answers what changed under a matched seed. It does not make the five seeds independent datasets.

With five non-zero pairs, the smallest exact two-sided Wilcoxon signed-rank p-value is 0.0625. Report effect size, paired interval and direction consistency instead of writing “statistically significant improvement”.

## 29. CIFAR-100 interface audit

CIFAR-100 uses the same 32×32 RGB input and 8×8 patch grid as CIFAR-10, but the classifier output changes from `(B, 10)` to `(B, 100)`. Dataset-specific normalisation is kept inside `datasets/cifar100_data.py`, while the ViT/PE interfaces remain unchanged.

## 30. Why the full-data point is not connected to the reduced-data curve

A training-size curve should change the amount of training data while holding
the remaining optimisation protocol fixed. The completed 1k, 5k and 10k runs
all use `lr=1e-3`, so those three conditions form an internally controlled
comparison. The existing final CIFAR-10 full-data runs use `lr=3e-4`.

Connecting the existing full-data result to the three reduced-data points would
therefore change both training size and learning rate at the final point. The
observed difference could not be attributed only to data availability. The
current figure omits that connection and records the mismatch in its caption and
manifest. A fully controlled four-point curve requires either rerunning the
reduced-data conditions at `3e-4` or rerunning the full-data reference at
`1e-3`.

Selected-test plots use the validation-selected checkpoint once per run. Their
faint points are the five seed outcomes and their error bars are 95% t intervals:

```text
mean +/- 2.776445 * sample_SD / sqrt(5)
```

Epoch plots serve a different purpose. They use validation accuracy and loss,
show mean +/- a pointwise 95% t interval, and stop each model curve once fewer than all five
seeds remain after early stopping.

## 31. 为什么论文 Epoch 曲线不再使用三角形等 marker

多模型曲线上的三角形、方形和菱形如果只用于区分模型，容易被误读为 selected epoch、
显著性或特殊事件。v2 论文图因此采用：

```text
colour + line pattern = model or assignment
shaded band           = mean +/- pointwise 95% t CI across seeds
point marker          = omitted on epoch curves
```

最终 test 汇总图不是训练轨迹。它只在 validation-selected checkpoint 上产生一个结果，
因此使用 categorical model x-axis，并明确规定：半透明圆点是单个 seed，菱形是均值，
竖向 error bar 是 95% t confidence interval。marker 本身不表示统计显著性。

## 32. Low-data subset seed 的解释

`split_seed=42` 固定 CIFAR train/validation partition，但低数据量抽样调用：

```python
make_subset(train_dataset, train_subset, seed)
```

因此，同一个 training seed 下四个模型得到相同 subset；不同 training seed 同时改变
subset composition、模型初始化、augmentation randomness 和训练顺序。Low-data 的五次
重复及其 interval 反映这些来源的综合 repeatability，不能描述为纯 initialization variance。

## 33. Full-data 何时可以连接到 low-data 曲线

只有在 learning rate、scheduler、augmentation、normalisation、split seed、batch size、
weight decay、early stopping、模型结构和 selected-checkpoint-only test protocol 全部一致时，
full-data 才能作为第四个数据量点。新的 `lr3e4` 结果通过了该自动 gate，因此1k、5k、
10k和full-data可以在同一受控图中连接。旧 `lr=1e-3` subset结果没有通过这一条件。

## 34. Epoch curve 和 final comparison 是两类证据

训练期间每个 epoch 都有 validation metric，因此可以画五个 seeds 的 mean，并用
`2.776445 * sample_SD / sqrt(5)` 形成 pointwise 95% t interval。这类图回答模型怎样
收敛以及轨迹是否稳定。

Test set 只在加载 validation-selected checkpoint 后评估一次，所以不能画 test
accuracy 随 epoch 的曲线。最终 variant 比较应从 selected-checkpoint test
accuracy/loss 表读取。

不同 seeds 因 early stopping 具有不同长度时，先取各 seed 最大 epoch 的最小值，只对
共同范围内恰好包含五个 seeds 的 epoch 汇总，避免曲线末端悄悄减少重复数。

## 35. Sequence-slot assignment 与 coordinate-aligned assignment

`PatchEmbedding` 先从卷积得到 row-major physical patches，再用 `patch_order` 重排 token。
历史 fixed-PE 模型随后直接加 row-major `pos_embed`，所以 sequence slot `s` 得到 PE
coordinate `s`：

```python
patch_tokens = patch_tokens.index_select(1, patch_order)
x = patch_tokens + original_pos_embed
```

新模式只重排固定 PE 的 patch 部分，CLS PE 保持在第零位：

```python
ordered_patch_pe = original_pos_embed[:, 1:].index_select(1, patch_order)
ordered_pos_embed = torch.cat((original_pos_embed[:, :1], ordered_patch_pe), dim=1)
x = patch_tokens + ordered_pos_embed
```

因此 sequence order 可以改变，但每个 physical patch 始终得到由其原始 row/column
coordinate 生成的 PE。当前 encoder 使用 global unmasked self-attention、shared token-wise
LayerNorm/MLP 和最终 CLS readout，没有 window attention、sequence convolution、邻接运算或
position-dependent mask。同步置换 patch 与 PE 时，logits 理论上只应有浮点误差。
