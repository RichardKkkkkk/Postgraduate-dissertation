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
