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

## 2026-06-14 CADB Orientation Subset

这一步新增的不是一个全新的大模型，而是真实数据集接入层：

- `cadb_orientation`

它的作用是把老师提到的 CADB 转成当前实验平台能直接训练的二分类子任务。

### 1. 为什么不是直接整套 CADB 全上

因为老师这周的问题非常聚焦：

- horizontal relationships
- vertical relationships

所以当前最小可用版本只做：

- `horizontal`
- `vertical`

这样实验问题更干净，也更适合先看 row-wise / col-wise positional bias。

### 2. 当前 loader 默认做了什么假设

当前实现优先读取 `composition_elements.json`，因为真实 CADB 里
`horizontal / vertical` 更直接对应 composition element annotation。

然后默认：

- 保留只属于 `horizontal` 的图像
- 保留只属于 `vertical` 的图像
- 跳过同时属于两者的样本
- 跳过两者都不属于的样本

这就是 `exclusive` label mode。

英文可以这样记：

`Exclusive mode keeps only clean horizontal-only or vertical-only samples.`

### 3. 为什么要自己切 split

CADB 不是像 CIFAR-10 那样自带官方 `train / test`。

修正后发现，真实 CADB 实际上带有 `split.json`。

所以现在 loader 里会优先：

1. 读取官方 `train / test`
2. 再从官方 `train` 里切一个 `validation`

如果某份数据没有 `split.json`，才会退回到全数据上自己切分。

这样做的好处是：

- horizontal / vertical 两类在三个 split 里都更平衡
- seed 固定后，比较可重复

### 4. 这层代码和训练主循环的关系

你现在的统一训练入口根本没变。

变的是：

- `models/registry.py` 多了一条数据集路由
- `datasets/cadb_data.py` 负责把 CADB 解析成标准 `DataLoader`

也就是说，训练层看到的仍然只是：

```text
train_loader, val_loader, test_loader
```

所以对主训练循环来说：

- CIFAR-10
- synthetic orientation
- CADB orientation

都已经被统一成同一接口了。

## Row-wise vs Column-wise Sinusoidal 是怎么改结构的

`vit_row_sinusoidal` 和 `vit_col_sinusoidal` 没有改动主干 Transformer block。

变动点只在 positional embedding：

- `row-wise`
  - patch 位置信息只看它属于第几行
  - 同一行的 patch 会共享同一个 sinusoidal position code
- `column-wise`
  - patch 位置信息只看它属于第几列
  - 同一列的 patch 会共享同一个 sinusoidal position code

可以把它记成一句英文：

`Same ViT backbone, different axis used for fixed sinusoidal indexing.`

所以这类实验特别适合回答：

- 横向/纵向的结构线索，是否更偏好某一种 axis-tied positional bias
- 观察到的变化是不是来自位置编码方向，而不是来自更大的模型结构改写

## 为什么 clean v2 更像 sanity check

`synthetic_orientation_clean` 的作用不是把三种模型完全拉开，
而是先看：

- 方向性位置编码有没有把任务做坏
- baseline / row / col 在一个更干净的 controlled setting 下是否都能顺利收敛

如果三者都接近满分，这不代表实验没用。

它说明的是：

- 这个任务已经太容易
- 这一步适合当作 “clean sanity check”
- 真正更能拉开差异的下一步，应该去 `synthetic_orientation_hard`
  或更难的真实数据集
