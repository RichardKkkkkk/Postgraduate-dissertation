# Learning Notes

这个文件专门记录学习解释。README 只放项目入口、运行方式和文件结构。

## 怎么解释会更清楚

后续每次改代码，我会按这个顺序解释：

1. 先说目标：这次改动解决什么问题。
2. 再说数据流：输入是什么，输出是什么。
3. 再说 tensor shape：每一步 shape 怎么变。
4. 再说 PyTorch 语法：这行代码为什么这么写。
5. 最后说如何验证：用什么命令确认它能跑。

比如解释一个训练 batch 时，不应该只说“训练模型”，而应该说：

```text
images: [B, 3, 32, 32]
labels: [B]
logits = model(images): [B, 10]
loss = CrossEntropyLoss(logits, labels): scalar
loss.backward(): 计算每个可训练参数的梯度
optimizer.step(): 根据梯度更新参数
```

这样你能同时看到“代码在干嘛”和“张量在怎么流动”。

## 为什么先用 CIFAR-10

CIFAR-10 适合当前 MVP：

- 图片尺寸是 `32x32`，当前 `ViT(img_size=32)` 可以直接使用。
- 图片是 RGB，所以输入通道数是 `3`，对应 `in_channels=3`。
- 一共有 10 类，所以分类头输出维度是 `num_classes=10`。
- 数据集比 ImageNet 小很多，适合先在 Mac 上跑通训练逻辑。
- 它仍然是图像分类任务，有二维空间结构，适合后续研究位置编码。

## 当前训练流程

`train_cifar10.py` 的整体流程是：

1. 用 `torchvision.datasets.CIFAR10` 读取 CIFAR-10。
2. 用 `transforms` 把 PIL 图片转成 tensor，并做标准化。
3. 用 `DataLoader` 把样本组成 batch。
4. 创建 `ViT` 模型，并移动到 `mps`、`cuda` 或 `cpu`。
5. 前向传播得到 `logits`。
6. 用 `CrossEntropyLoss` 计算分类损失。
7. 用 `loss.backward()` 反向传播。
8. 用 `optimizer.step()` 更新参数。
9. 每个 epoch 后在测试集上评估 loss 和 accuracy。

## 一个 batch 的 shape

CIFAR-10 的一张图片原始大小是 `32x32`，RGB 三通道。

如果 `batch_size=64`：

```text
images: [64, 3, 32, 32]
labels: [64]
```

进入 ViT 后：

```text
PatchEmbedding 输出: [64, 64, 128]
加 cls token 后: [64, 65, 128]
Transformer blocks 输出: [64, 65, 128]
取 cls token: [64, 128]
分类头输出 logits: [64, 10]
```

这里 `64` 个 patch 来自：

```text
32 / patch_size 4 = 8
8 * 8 = 64 patches
```

## 关键 PyTorch 语法

`images.to(device)` 把图片移动到指定设备。模型和数据必须在同一个设备上。

```python
images = images.to(device)
model = model.to(device)
```

`logits = model(images)` 会调用模型的 `forward` 方法。PyTorch 里一般写 `model(images)`，而不是直接写 `model.forward(images)`。

```python
logits = model(images)
```

`nn.CrossEntropyLoss()` 适合多分类任务。它接收未经过 softmax 的 `logits`，所以模型最后一层不需要手动加 softmax。

```python
criterion = nn.CrossEntropyLoss()
loss = criterion(logits, labels)
```

`optimizer.zero_grad()` 清空上一轮梯度。PyTorch 默认会累加梯度，所以每个 batch 反向传播前要清空。

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

`@torch.no_grad()` 表示评估时不记录计算图，可以减少内存占用，也避免误算梯度。

```python
@torch.no_grad()
def evaluate(...):
    model.eval()
```
