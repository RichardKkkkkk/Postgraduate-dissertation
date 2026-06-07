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
epoch, train_loss, train_acc, val_loss, val_acc, test_loss, test_acc
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

为了让这个 scratch baseline 更清楚，现在额外提供了：

```text
train_resnet18_scratch_cifar10.py
```

它只是一个明确入口，内部仍然复用 `train_cnn_cifar10.py` 的训练流程。它会强制补上：

```text
--weights none
```

这样以后跑“ResNet18 不加载权重”时，不需要记住额外参数，也不会不小心跑成 ImageNet 预训练版本。

这个文件里的关键 Python 语法是 `sys.argv`。`sys.argv` 保存命令行参数，比如：

```bash
python train_resnet18_scratch_cifar10.py --epochs 1
```

在 Python 里大概会变成：

```python
["train_resnet18_scratch_cifar10.py", "--epochs", "1"]
```

脚本会在调用原训练入口前追加：

```python
sys.argv.extend(["--weights", "none"])
```

所以最终等价于：

```bash
python train_cnn_cifar10.py --epochs 1 --weights none
```

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
## Experiment Reporting

`generate_comparison_report.py` is designed to sit on top of the existing
artifact format instead of being tied to a single model:

- it reads `results/metrics/<run_name>_metrics.csv`
- it reads `results/metrics/<run_name>_config.json`
- it reads `results/metrics/<run_name>_summary.json`
- it compares every shared numeric metric across runs
- it exports plots, CSV summaries, and a simple PPT deck

This keeps the reporting layer reusable when new models or new metrics are
added later, as long as they follow the same result-file convention.

## Validation-Based Training

项目现在采用更规范的 `train / validation / test` 流程。

```text
official CIFAR-10 train -> train split + validation split
official CIFAR-10 test  -> final test split
```

为什么这一步重要：

- `train` 只负责参数更新
- `validation` 用来 early stopping 和模型选择
- `test` 只用于最终报告，不再参与调参

这是后面做结构对比前必须补齐的方法学清理。
否则你会在不知不觉中不断根据 `test` 结果做决策，最后让实验结论失去说服力。

英文可以记一句：
`Validation is for selection, test is for final reporting.`

## Selected-Checkpoint Evaluation Outputs

训练脚本现在保存的是“最佳 validation checkpoint”的评估结果，而不只是最后一个 epoch 的结果。

当前会额外保存：

- `confusion matrix`
- `macro precision`
- `macro recall`
- `macro F1`
- `per-class precision / recall / F1 / accuracy`
- saved best checkpoint under `checkpoints/`

这些结果为什么有用：

- `accuracy` 只能告诉你“整体有没有变好”
- `confusion matrix` 能告诉你“具体是哪些类别的混淆变了”
- `macro` 指标比单纯 accuracy 更平衡，后面如果迁移到医疗或类别不均衡数据，会更重要

## Why Split Baseline And RoPE

对于论文来说，`clean ablation boundary` 很重要。
如果 baseline 模型文件里一直往下堆条件分支，后面就很难说清楚：
“这次性能变化到底来自结构改动，还是来自 baseline 被顺手改过了。”

现在的组织方式是：

- `vit.py`: original ViT baseline with learned absolute positional embedding
- `vit_rope.py`: separate basic RoPE reproduction baseline

这样拆开的好处：

- `vit.py` 可以一直保持为干净的 baseline
- `ViT baseline` 和 `ViT + RoPE` 变成显式的两个模型分支
- 后面如果继续做 `2D RoPE`、`RoPE + locality bias`，边界会更清楚

英文可以记一句：
`Keep the baseline stable, add variants explicitly.`

## Basic RoPE Intuition

最基础的 RoPE 可以先记成一句话：

`RoPE = rotate Q and K by position.`

展开来说就是：

- 它不是像 absolute positional embedding 那样，直接给 token 加一个位置向量
- 它是按照位置对 `Q` 和 `K` 做旋转
- 因此位置信息是进入了 attention 计算本身，而不是作为额外 token 内容拼进去

在当前这版复现里：

- image patches are still produced exactly like the baseline ViT
- `cls token` is still concatenated at the front
- only patch-token `Q` and `K` are rotated
- `V` is not rotated
- the implementation is still a simple 1D sequence-style RoPE

如果按当前 CIFAR-10 的小 ViT 配置来看，shape 是：

```text
images: [B, 3, 32, 32]
patch tokens after PatchEmbedding: [B, 64, 128]
after cls token concat: [B, 65, 128]
q / k / v after multi-head split: [B, 4, 65, 32]
patch-only q / k used for RoPE: [B, 4, 64, 32]
cos / sin cache: [1, 1, 64, 32]
```

这里最关键的是：

- `32x32` 图像切成 `4x4 patch` 后，会得到 `8x8=64` 个 patch
- 加上 `cls token` 之后，序列长度变成 `65`
- 进入多头注意力后，每个 head 的维度是 `32`
- RoPE 只作用在 patch token 上，所以真正被旋转的是 `[B, 4, 64, 32]`

因此这版更适合叫：

- `RoPE learning baseline`
- 还不是 `2D image-aware RoPE`

## 2D RoPE Intuition

这次新增的 `vit_rope_2d.py` 不是完整新架构，而是在当前小 ViT 上做一个更贴近图像网格的轻量扩展。

可以先记成一句话：

`2D RoPE = rotate Q and K by row and column positions separately.`

核心逻辑是：

- patch 仍然来自同一个 `H x W` 网格
- 不再只给每个 patch 一个一维序列位置
- 而是给每个 patch 一个二维位置：`(row, col)`
- `Q` 和 `K` 的每个 head 维度被拆成两半
- 前一半通道用 `row position` 做旋转
- 后一半通道用 `col position` 做旋转
- `cls token` 继续不参与旋转
- `V` 仍然不旋转

这样做的直觉是：

- 1D RoPE 更像“按序列顺序编码”
- 2D RoPE 更像“按图像网格坐标编码”
- 对图像任务来说，这样的设计更容易保留横向和纵向空间关系

这版实现故意保持边界很干净：

- 没有 window attention
- 没有 shifted window
- 没有层级下采样
- 没有 Swin 式大改结构

所以它更适合被理解成：

- `2D-aware RoPE baseline`
- 而不是一个完整的新 backbone

## Early Stopping

现在 `train_cifar10.py` 和 `train_cnn_cifar10.py` 都已经支持 early stopping。

它的作用是：

- 每个 epoch 结束后检查一次监控指标
- 如果连续 `patience` 个 epoch 没有提升，就提前停止
- 把最佳 epoch 和最佳指标值写进 `summary.json`
- 在输出最终评估结果前，先恢复最佳 validation checkpoint

当前支持监控的指标：

- `val_acc`
- `val_loss`

## Unified Experiment Runner

现在项目里除了模型专用入口外，还新增了一个统一实验入口：

```text
train_cifar10_experiment.py
```

它的作用不是替代所有旧脚本，而是提供一个更稳定的实验接口。

当前支持的模型名：

- `vit_baseline`
- `vit_rope`
- `vit_rope_2d`
- `resnet18_scratch`
- `resnet18_imagenet`

这样做的好处是：

- 以后跑实验时，命令风格统一
- 结果文件结构保持一致
- 后面加新模型时，更自然的做法是“往统一入口注册一个新模型”
- 不需要每来一个变体就复制一份新的 `main` 训练脚本

当前近期主实验线建议收紧为：

- `vit_baseline`
- `vit_rope`
- `resnet18_scratch`

然后在第二轮结构比较里再加入：

- `vit_rope_2d`

`resnet18_imagenet` 现在仍然保留在统一入口里，但更适合作为可选参考，而不是近期主结论的一部分。

英文可以记一句：
`Use one experiment runner, register multiple model variants.`

现在这个统一入口内部已经进一步整理成了“注册表模式”：

- 训练主循环只有一份
- 每个模型只需要提供：
  - 如何构建 model
  - 如何构建 dataloader
  - 默认超参数是什么
  - 元信息是什么，比如 `architecture`、`variant`

这意味着后面如果你新增模型，不应该先想“我要不要复制一个新的训练脚本”，
而应该先想：

```text
我能不能把它注册进统一入口
```

这种结构特别适合论文实验，因为：

- 所有模型共享同一套训练控制逻辑
- 所有模型共享同一套结果输出格式
- 结构差异和训练流程差异不会混在一起

## Why Run Multiple Seeds

单次 `seed=42` 的结果可以帮助你判断方向，但还不够支持更强的论文结论。

现在更合理的说法是：

- 单次结果说明 `2D RoPE` 有正信号
- 多 seed 结果才能说明这个信号是不是稳定

为什么要这样做：

- 神经网络训练本身有随机性
- train / validation split 会随 seed 改变
- 参数初始化和 dataloader shuffle 也会改变

所以如果只看一个 seed，很容易把偶然波动误认为结构提升。

当前项目里新增的 `run_seed_sweep.py` 做的事情是：

- 对同一组模型循环多个 `seed`
- 自动生成形如 `model_seed42` 的 run name
- 每个 seed 结束后自动生成一个对应的 comparison report

这样你后面看结果时，可以先回答两个层次的问题：

1. 每个 seed 内部，`vit_baseline`、`vit_rope`、`vit_rope_2d` 谁更好
2. 跨多个 seed，这个排序是否稳定

英文可以记一句：

`Single-seed results show signal, multi-seed results show stability.`

## Why Add A Seed Summary Layer

有了 `run_seed_sweep.py` 之后，你已经能拿到：

- one report per seed
- one set of curves per seed

但这还不够回答论文里更关键的问题：

`Across seeds, is the improvement stable on average?`

这就是 `summarize_seed_sweep.py` 的作用。

它读的输入不是每个 epoch 的曲线，而是每个 run 已经保存好的：

- `results/metrics/<run_name>_summary.json`

然后提取出几个最重要的 selected-checkpoint 指标：

- `best_val_acc`
- `selected_model.test_acc`
- `selected_model.test_macro_f1`
- `selected_model.epoch`

再做聚合：

- `mean`
- `std`
- `min`
- `max`

输出层分成两类：

1. per-seed table
   作用：保留每个 seed 的原始结果，方便排查异常 seed
2. aggregate table
   作用：直接回答“平均表现”和“波动大小”

你可以把它理解成：

- `run_seed_sweep.py` = produce raw repeated experiments
- `summarize_seed_sweep.py` = compress repeated experiments into evidence

英文可以记一句：

`Per-seed reports show behavior, aggregated summaries show reliability.`

## Single Training Entry

现在项目已经进一步收口成：

- 只有一个正式训练脚本：`train_cifar10_experiment.py`
- 训练工具放在 `experiment_utils.py`
- 数据构建放在 `cifar10_data.py`
- 模型注册和默认配置放在 `model_registry.py`

也就是说，后面你要扩展项目时，优先不要再写：

- `train_xxx.py`
- `train_model_y.py`

而应该优先做：

1. 写模型文件
2. 在注册表里接入

这样更容易长期维护，也更适合 clean ablation。

## Model-Aware Reporting Layer

`generate_comparison_report.py` 现在不再主要依赖 run name 猜模型，而是优先读取
结果文件里已经保存好的结构化字段，例如：

- `selected_model.model_name`
- `selected_model.model_family`
- `selected_model.model_variant`
- `position_encoding` 或可以稳定推断它的字段

这样做的原因是：

- `unified_vit_baseline_smoke` 这种 run name 更像工程文件名
- 但组会 PPT 里更应该显示 `ViT Baseline`
- 报告页应该围绕“模型家族 / 变体 / 位置编码 / 初始化方式”来讲，而不是围绕日志文件名来讲

现在报告层支持更清楚地展示：

- `ViT Baseline vs ViT RoPE`
- `ViT vs CNN`
- `ResNet18 scratch vs ImageNet pretrained`

## Analysis Layer vs PPT Layer

这次也把报告脚本内部继续往两层拆了一步：

- analysis layer:
  - load run artifacts
  - infer model metadata
  - detect comparison scenario
  - build metric / macro / per-class payloads
- presentation layer:
  - consume the prepared context
  - place tables, cards, figures, and conclusion text into slides

可以记一句英文：

`Prepare comparison context first, render slides second.`

这样后面如果再新增模型，不需要第一时间去改 PPT 排版代码，
而是先把元信息识别和分析 payload 补上。

## Seed Summary Evidence vs Per-Seed Evidence

这次把报告层又明确分成了两种“证据粒度”：

1. per-seed comparison deck
   作用：
   - 看单个 seed 下不同模型的训练曲线和最终结果
   - 排查异常 seed
   - 看某一次 run 有没有训练崩掉

2. aggregate seed-summary deck
   作用：
   - 直接展示 `mean/std`
   - 回答“平均性能是否更好”
   - 回答“提升是否稳定”

可以把它记成一句话：

`Per-seed decks explain behavior, aggregate decks explain reliability.`

这也是为什么 `cifar10_main_seed_summary` 更适合组会主结论：

- 它不是在讲某一次幸运 run
- 而是在讲 3 个 seed 上的平均表现和波动范围
- 所以结论更像 research evidence，而不是 training log

现在脚本职责也重新收口成两层：

- `summarize_seed_sweep.py` 只负责把多 seed 结果压缩成汇总 artifact
- `generate_comparison_report.py` 统一负责把单次 run 或 aggregate summary 变成 PPT
