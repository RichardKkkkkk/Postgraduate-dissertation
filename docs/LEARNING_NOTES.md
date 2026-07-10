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

## 8. 当前结果管理约定

现在项目里区分两类结果。

### 原始结果

放在：

- `results/`

作用：

- 本地分析
- 中间产物
- 可以随时重跑

### 长期文档

放在：

- `docs/`

作用：

- 记录研究路线

## 9. 小 subset smoke test 的 confusion matrix

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
- 记录项目状态
- 作为跨设备共享记忆
