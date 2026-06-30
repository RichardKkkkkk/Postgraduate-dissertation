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

### `vit_no_pos`

- 没有 positional encoding
- 用来回答“位置编码本身有没有帮助”

### `vit_baseline`

- learned absolute positional embedding
- 这是当前标准 ViT baseline

### `vit_row_sinusoidal`

- patch token 的位置只看 row
- 同一行共享位置向量

### `vit_col_sinusoidal`

- patch token 的位置只看 column
- 同一列共享位置向量

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
- 记录项目状态
- 作为跨设备共享记忆
