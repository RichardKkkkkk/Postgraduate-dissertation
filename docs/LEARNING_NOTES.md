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
10. 训练结束后保存指标 CSV、摘要 JSON、loss 曲线和 accuracy 曲线。

默认情况下，脚本会使用完整 CIFAR-10：

```text
train: 50000 images
test: 10000 images
```

默认训练 `5` 个 epoch。epoch 可以理解为“完整看一遍训练集”。如果 `epochs=5`，模型会把训练集从头到尾学习 5 遍。

代码里这一行控制训练轮数：

```python
parser.add_argument("--epochs", type=int, default=5)
```

训练循环是：

```python
for epoch in range(1, args.epochs + 1):
    ...
```

如果 `args.epochs = 5`，`range(1, 6)` 会依次产生：

```text
1, 2, 3, 4, 5
```

所以它会跑 5 个 epoch。

`--train-subset` 和 `--test-subset` 只是为了快速 smoke test。例如：

```bash
python train_cifar10.py --epochs 1 --train-subset 2000 --test-subset 500
```

这表示只抽 2000 张训练图、500 张测试图来快速验证代码能跑，不代表正式实验设置。

训练时默认每个 epoch 输出一次汇总结果，而不是每个 batch 都输出。这样日志更适合观察整体趋势：

```text
Epoch 1/5
  train loss=... acc=... | test loss=... acc=...
Epoch 2/5
  train loss=... acc=... | test loss=... acc=...
```

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

## 结果保存

训练结束后，脚本会保存四类结果。

第一类是逐 epoch 指标：

```text
results/metrics/<run_name>_metrics.csv
```

CSV 里面每一行对应一个 epoch：

```text
epoch, train_loss, train_acc, test_loss, test_acc
```

第二类是摘要：

```text
results/metrics/<run_name>_config.json
```

这里记录本次实验的关键设置，包括：

```text
command: 原始运行命令
device: 使用 cpu / mps / cuda
dataset: 数据集名称、实际 train/test 样本数、data_dir
training: epochs、batch_size、lr、weight_decay、seed 等
model: img_size、patch_size、embed_dim、num_heads、num_blocks 等
outputs: results_dir 和 run_name
```

这就是为了防止以后忘记某次实验到底是怎么跑的。

第三类是摘要：

```text
results/metrics/<run_name>_summary.json
```

这里记录最佳测试准确率、最佳 epoch、最后一个 epoch 的结果，以及本次运行的命令行参数。

第四类是曲线图：

```text
results/figures/<run_name>_loss.png
results/figures/<run_name>_accuracy.png
```

loss 曲线用来看模型是不是在学习。如果 train loss 下降，说明模型正在拟合训练集。accuracy 曲线用来看分类结果是否变好。

## 要不要单独分 validation dataset

严格实验里，一般会有 `train / validation / test` 三部分。

它们的分工是：

- `train`：用来反向传播和更新参数。
- `validation`：用来调学习率、模型大小、epoch 数、位置编码方案等超参数。
- `test`：只在最后使用，用来报告最终泛化性能。

CIFAR-10 官方只给了 `train` 和 `test`。所以常见做法是从官方训练集里再切一部分做 validation：

```text
官方 train 50000 -> train 45000 + validation 5000
官方 test 10000 -> final test
```

现在这个 MVP 暂时每个 epoch 直接看 test loss 和 test accuracy，是为了先把训练流程跑通。等开始比较不同位置编码方案时，就应该加 validation split，避免一边调参一边反复看 test，导致 test 结果不再客观。

## CNN baseline

下一步加入 `train_cnn_cifar10.py`，用 ResNet18 作为 CNN baseline。

为什么这个 baseline 有意义：

- CNN 有卷积归纳偏置，天然擅长局部纹理和局部空间结构。
- ResNet18 是经典 CNN，结构成熟，训练稳定。
- ImageNet 预训练权重能提供很强的图像特征起点。
- 和从零训练的小 ViT 对比，可以帮助我们判断：当前差距来自模型结构、数据量，还是预训练。

默认 CNN 脚本使用：

```text
model: torchvision.models.resnet18
weights: ImageNet pretrained
input image size: 224x224
num_classes: 10
```

因为 ImageNet 预训练 ResNet18 原本是在 `224x224` 图片上训练的，所以脚本会把 CIFAR-10 的 `32x32` 图片 resize 到 `224x224`。这样更符合预训练模型原来的输入习惯。

如果不用预训练：

```bash
python train_cnn_cifar10.py --weights none
```

默认输入尺寸会变成 `32x32`，这样更像“从零训练一个 CNN 来分类 CIFAR-10”。

ResNet18 的最后一层原本输出 ImageNet 的 1000 类。这里要改成 CIFAR-10 的 10 类：

```python
model = models.resnet18(weights=weights)
model.fc = nn.Linear(model.fc.in_features, 10)
```

重要 shape：

```text
images: [B, 3, 224, 224]  # pretrained ResNet18 默认
logits: [B, 10]
labels: [B]
loss: scalar
```

这里的 `model.fc` 是 ResNet18 的分类头。前面的卷积层负责提取图像特征，最后的 `fc` 把特征映射到类别分数。
