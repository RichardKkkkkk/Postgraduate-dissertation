# ViT Learning

这个项目用于逐步复现 Vision Transformer，并为后续研究“如何让 ViT 更好地保留二维位置信息”打基础。

当前 MVP 使用 CIFAR-10 跑通完整训练流程：数据集、DataLoader、模型前向传播、loss、反向传播、参数更新和测试集评估。

## 运行

本项目开发统一使用已有 conda 环境 `vit_research`。先激活环境：

```bash
conda activate vit_research
```

然后先跑一个小规模实验：

```bash
python train_cifar10.py --epochs 2 --train-subset 2000 --test-subset 500
```

确认逻辑没问题后，再逐步变大：

```bash
python train_cifar10.py --epochs 10 --train-subset 10000 --test-subset 2000
```

## 文件结构

- `vit.py`：最小 ViT 模型实现。
- `train_cifar10.py`：CIFAR-10 训练和评估脚本。
- `docs/LEARNING_NOTES.md`：学习笔记，记录代码解释、PyTorch 语法和 tensor shape。

添加新的可运行脚本时，需要同步更新这个 README。
