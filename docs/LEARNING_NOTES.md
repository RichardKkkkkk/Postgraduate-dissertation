# Learning Notes

## 2026-06-13 Row-wise / Column-wise Sinusoidal ViT

这次新增的不是 RoPE 变体，而是两种新的 additive positional embedding：

- `vit_row_sinusoidal`
- `vit_col_sinusoidal`

它们的目标是做一个更干净的方向性实验。

### 1. 核心思想

`vit_row_sinusoidal`：

- patch token 的位置只由 `row index` 决定
- 同一行上的 patch 共享同一个 positional vector
- 不显式区分这一行里不同列的位置

`vit_col_sinusoidal`：

- patch token 的位置只由 `column index` 决定
- 同一列上的 patch 共享同一个 positional vector
- 不显式区分这一列里不同行的位置

一句英文解释：

`Row-wise and column-wise models inject directional bias through additive positional embeddings.`

### 2. 它和 baseline 的边界

这里故意保持 baseline ViT 主体不变：

- 仍然先做 patch embedding
- 仍然保留 `cls token`
- 仍然走标准 Transformer blocks
- 仍然在 token level 上做 additive position embedding

变化只在这里：

- baseline 用的是 learned absolute positional embedding
- 新模型用的是 fixed sinusoidal positional embedding
- 而且位置只沿一个轴定义

所以它是 clean ablation，不是新 backbone。

### 3. 位置向量怎么构造

如果输入是 `32 x 32`，`patch_size = 4`，那么：

```text
grid_size = 32 / 4 = 8
num_patches = 8 * 8 = 64
```

row-wise 的 token 位置索引相当于：

```text
0 0 0 0 0 0 0 0
1 1 1 1 1 1 1 1
2 2 2 2 2 2 2 2
...
7 7 7 7 7 7 7 7
```

column-wise 的 token 位置索引相当于：

```text
0 1 2 3 4 5 6 7
0 1 2 3 4 5 6 7
0 1 2 3 4 5 6 7
...
0 1 2 3 4 5 6 7
```

然后把这些位置索引送进标准 sinusoidal formula，得到每个 patch token 的固定位置向量。

### 4. Tensor shape 怎么看

以当前小 ViT 默认配置为例：

```text
images: [B, 3, 32, 32]
patch tokens: [B, 64, 128]
after cls token: [B, 65, 128]
pos_embed: [1, 65, 128]
after addition: [B, 65, 128]
logits: [B, num_classes]
```

这里的重点是：

- `cls token` 仍然放在最前面
- `cls token` 的 positional vector 设成全零
- patch token 才真正携带 row-wise 或 col-wise positional bias

### 5. 为什么先做 additive 版本

这是为了把变量控制住。

如果一开始就改 attention 机制，你很难回答：

- 提升来自“方向先验”
- 还是来自“attention 算法本身改了”

现在这版更适合论文前期：

- 实现简单
- 可解释性强
- 更容易和 baseline 做一对一比较

## 2026-06-13 Synthetic Orientation Dataset

为了配合老师说的 horizontal / vertical relationship 实验，这次补了一个轻量 synthetic dataset：

- `horizontal`
- `vertical`

### 1. 为什么先做 synthetic

因为它最适合快速验证假设：

- 变量可控
- 类别定义清楚
- 能直接把“横向结构”和“纵向结构”分开
- 很适合先看 epoch 曲线

一句英文解释：

`Synthetic data is the cleanest first test for directional inductive bias.`

### 2. 数据是怎么生成的

每张图像都会：

1. 先生成一个低强度背景
2. 随机采样若干 stripe 的位置
3. 如果标签是 `horizontal`，就在行方向画条纹
4. 如果标签是 `vertical`，就在列方向画条纹
5. 最后再加一点高斯噪声

这样做的好处是：

- 类别差异主要来自方向
- 不是来自复杂语义内容

### 3. split 为什么是固定 seed 生成

当前 `train / val / test` 都是可重复生成的。

这意味着：

- 同一个 seed 下，你的 row / col / baseline 比较更公平
- 不会因为每次重新生成数据而导致比较失真

## 2026-06-13 当前周报应该怎么看

这周不再优先讲 multi-seed stability。

更适合讲的是：

1. 我们把实验平台改成了更规范的 `train / val / test`
2. 现在已经支持方向性 positional embedding 的 clean ablation
3. 现在已经有一个受控 synthetic dataset，可以直接检验 row / col bias
4. 接下来主要看 epoch-based loss / accuracy curves

一句组会常用英文：

`This week the focus shifts from seed stability to directional positional-bias comparison under a controlled setup.`
