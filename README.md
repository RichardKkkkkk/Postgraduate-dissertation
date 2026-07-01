# ViT 研究工作区

这个仓库是一个面向论文实验的轻量研究平台。  
当前目标不是“堆很多训练脚本”，而是保持一个统一训练入口，在稳定实验协议下比较不同的 positional encoding 设计。

一句英文概括：

`Use one runner, register many model variants, and keep the experiment contract stable.`

## 当前主线

当前统一主线是：

- 一个训练入口：`train_cifar10_experiment.py`
- 一个模型与数据集注册表：`models/registry.py`
- 一个训练工具模块：`experiment_utils.py`
- 一个结果对比入口：`generate_comparison_report.py`

当前主线模型：

- `vit_no_pos`
- `vit_baseline`
- `vit_row_sinusoidal`
- `vit_col_sinusoidal`

当前主线任务：

- `cadb_elements`

## 环境

统一使用 conda 环境：

```bash
conda activate vit_research
```

安装依赖：

```bash
pip install -r requirements.txt
```

## 项目结构

- `models/`
  放模型定义和模型注册逻辑
- `datasets/`
  放数据读取、标签构造、split 和 dataloader
- `docs/`
  放项目结构、研究计划、项目日志和学习笔记
- `results/`
  放原始实验输出
- `checkpoints/`
  放训练得到的模型参数，不提交
- `data/`
  放本地数据集，不提交

关键文件：

- [train_cifar10_experiment.py](/D:/UCL/UCL-dissertation/Postgraduate-dissertation/train_cifar10_experiment.py:1)
  统一训练入口
- [models/registry.py](/D:/UCL/UCL-dissertation/Postgraduate-dissertation/models/registry.py:1)
  模型注册、数据集路由、默认超参数
- [experiment_utils.py](/D:/UCL/UCL-dissertation/Postgraduate-dissertation/experiment_utils.py:1)
  训练、评估、早停、指标、画图、保存
- [datasets/cadb_data.py](/D:/UCL/UCL-dissertation/Postgraduate-dissertation/datasets/cadb_data.py:1)
  CADB 任务定义和 dataloader
- [generate_comparison_report.py](/D:/UCL/UCL-dissertation/Postgraduate-dissertation/generate_comparison_report.py:1)
  对比图和报告生成

更详细的结构说明见 [PROJECT_STRUCTURE.md](/D:/UCL/UCL-dissertation/Postgraduate-dissertation/docs/PROJECT_STRUCTURE.md:1)。

## 支持的模型

- `vit_no_pos`
  不带 positional encoding 的 ViT 对照模型
- `vit_baseline`
  带 learned absolute positional embedding 的标准 ViT
- `vit_row_sinusoidal`
  只按 row 注入 sinusoidal positional embedding
- `vit_col_sinusoidal`
  只按 column 注入 sinusoidal positional embedding
- `vit_additive_sinusoidal`
  先分别生成 row / column sinusoidal embedding，再逐元素相加
- `vit_additive_sinusoidal_shifted`
  additive 版本，但 row / column 使用错开的 wavelength
- `vit_multiplicative_sinusoidal`
  先分别生成 row / column sinusoidal embedding，再逐元素相乘
- `vit_multiplicative_sinusoidal_shifted`
  multiplicative 版本，但 row / column 使用错开的 wavelength
- `vit_rope`
  1D RoPE 版本
- `vit_rope_2d`
  2D-aware RoPE 版本
- `resnet18_scratch`
  不带 pretrained weights 的 CNN baseline
- `resnet18_imagenet`
  可选的 ImageNet pretrained 参考模型

## 支持的数据集

### `cadb_elements`

这是当前最重要的 multi-label classification 任务。  
每张图对应一个多标签向量，表示多个构图元素是否出现。

当前标签集合：

- `horizontal`
- `vertical`
- `diagonal`
- `triangle`
- `symmetric`
- `pattern`

注意：

- 当前原始 CADB 标注里，`pattern` 没有有效正样本
- 因此这个标签暂时不能作为可靠结论来源

期望目录结构：

```text
data/CADB_Dataset/
├── composition_elements.json
├── split.json
└── images/
```

### 其他数据集

- `cifar10`
- `cadb_orientation`
- `cadb_scene`
- `synthetic_orientation`
- `synthetic_orientation_clean`
- `synthetic_orientation_hard`
- `synthetic_row_code`
- `synthetic_col_code`

## 统一训练入口

项目只保留一个正式训练入口：

```bash
python train_cifar10_experiment.py --model vit_baseline
```

这意味着：

- 训练逻辑只维护一份
- 早停逻辑只维护一份
- metrics / summary / checkpoint 保存逻辑只维护一份
- 新模型通过注册表接入，不再新建单独的 `train_xxx.py`

## 当前主线实验命令

### `vit_no_pos`

```bash
python train_cifar10_experiment.py --model vit_no_pos --dataset cadb_elements --cadb-root data/CADB_Dataset --image-size 96 --epochs 100 --seed 42 --early-stopping-patience 15 --early-stopping-metric val_macro_f1 --early-stopping-min-delta 0.001 --lr-plateau-patience 5 --lr-plateau-factor 0.5 --run-name no_pos_cadb_elements_seed42
```

### `vit_baseline`

```bash
python train_cifar10_experiment.py --model vit_baseline --dataset cadb_elements --cadb-root data/CADB_Dataset --image-size 96 --epochs 100 --seed 42 --early-stopping-patience 15 --early-stopping-metric val_macro_f1 --early-stopping-min-delta 0.001 --lr-plateau-patience 5 --lr-plateau-factor 0.5 --run-name baseline_cadb_elements_seed42
```

### `vit_row_sinusoidal`

```bash
python train_cifar10_experiment.py --model vit_row_sinusoidal --dataset cadb_elements --cadb-root data/CADB_Dataset --image-size 96 --epochs 100 --seed 42 --early-stopping-patience 15 --early-stopping-metric val_macro_f1 --early-stopping-min-delta 0.001 --lr-plateau-patience 5 --lr-plateau-factor 0.5 --run-name row_cadb_elements_seed42
```

### `vit_col_sinusoidal`

```bash
python train_cifar10_experiment.py --model vit_col_sinusoidal --dataset cadb_elements --cadb-root data/CADB_Dataset --image-size 96 --epochs 100 --seed 42 --early-stopping-patience 15 --early-stopping-metric val_macro_f1 --early-stopping-min-delta 0.001 --lr-plateau-patience 5 --lr-plateau-factor 0.5 --run-name col_cadb_elements_seed42
```

### `vit_additive_sinusoidal`

```bash
python train_cifar10_experiment.py --model vit_additive_sinusoidal --dataset cadb_elements --cadb-root data/CADB_Dataset --image-size 96 --epochs 100 --seed 42 --early-stopping-patience 15 --early-stopping-metric val_macro_f1 --early-stopping-min-delta 0.001 --lr-plateau-patience 5 --lr-plateau-factor 0.5 --run-name additive_cadb_elements_seed42
```

### `vit_additive_sinusoidal_shifted`

```bash
python train_cifar10_experiment.py --model vit_additive_sinusoidal_shifted --dataset cadb_elements --cadb-root data/CADB_Dataset --image-size 96 --epochs 100 --seed 42 --early-stopping-patience 15 --early-stopping-metric val_macro_f1 --early-stopping-min-delta 0.001 --lr-plateau-patience 5 --lr-plateau-factor 0.5 --run-name additive_shifted_cadb_elements_seed42
```

### `vit_multiplicative_sinusoidal`

```bash
python train_cifar10_experiment.py --model vit_multiplicative_sinusoidal --dataset cadb_elements --cadb-root data/CADB_Dataset --image-size 96 --epochs 100 --seed 42 --early-stopping-patience 15 --early-stopping-metric val_macro_f1 --early-stopping-min-delta 0.001 --lr-plateau-patience 5 --lr-plateau-factor 0.5 --run-name multiplicative_cadb_elements_seed42
```

### `vit_multiplicative_sinusoidal_shifted`

```bash
python train_cifar10_experiment.py --model vit_multiplicative_sinusoidal_shifted --dataset cadb_elements --cadb-root data/CADB_Dataset --image-size 96 --epochs 100 --seed 42 --early-stopping-patience 15 --early-stopping-metric val_macro_f1 --early-stopping-min-delta 0.001 --lr-plateau-patience 5 --lr-plateau-factor 0.5 --run-name multiplicative_shifted_cadb_elements_seed42
```

### 四模型对比报告

```bash
python generate_comparison_report.py --run no_pos_cadb_elements_seed42="ViT No Pos" --run baseline_cadb_elements_seed42="ViT Baseline" --run row_cadb_elements_seed42="ViT Row-wise" --run col_cadb_elements_seed42="ViT Column-wise" --report-name cadb_elements_positional_controls --title "CADB Elements: No Pos vs Baseline vs Row-wise vs Column-wise" --skip-ppt
```


## 训练输出

每次训练完成后会保存：

- `results/metrics/<model>/<run_name>_metrics.csv`
- `results/metrics/<model>/<run_name>_config.json`
- `results/metrics/<model>/<run_name>_summary.json`
- `results/figures/<model>/<run_name>_loss.png`
- `results/figures/<model>/<run_name>_accuracy.png`
- `checkpoints/<model>/<run_name>_best.pt`

单标签任务还会额外保存 confusion matrix。

核心实验原则：

- `validation` 用于 early stopping 和 model selection
- `test` 用于最终报告

## 常用 CLI 参数

### 通用参数

- `--model`
- `--dataset`
- `--data-dir`
- `--results-dir`
- `--checkpoint-dir`
- `--run-name`
- `--epochs`
- `--batch-size`
- `--lr`
- `--weight-decay`
- `--train-subset`
- `--val-subset`
- `--test-subset`
- `--val-ratio`
- `--num-workers`
- `--seed`
- `--early-stopping-patience`
- `--early-stopping-min-delta`
- `--early-stopping-metric`
- `--lr-plateau-patience`
- `--lr-plateau-factor`
- `--lr-plateau-min-lr`

### ViT 参数

- `--embedding-dropout`
- `--attention-dropout`
- `--projection-dropout`
- `--mlp-dropout`

### RoPE 参数

- `--rope-base`

### CADB 参数

- `--cadb-root`
- `--cadb-test-ratio`
- `--cadb-label-mode`
- `--cadb-balance-mode`
- `--image-size`

## 默认值

### 通用默认值

- `epochs = 5`
- `val_ratio = 0.1`
- `num_workers = 2`
- `seed = 42`
- `early_stopping_metric = val_acc`
- `early_stopping_min_delta = 0.0`
- `early_stopping_patience = disabled by default`

### ViT 默认值

- `batch_size = 128`
- `lr = 3e-4`
- `weight_decay = 0.05`
- 所有 dropout 默认 `0.0`

### ResNet18 默认值

- `batch_size = 64`
- `lr = 1e-4`
- `weight_decay = 0.01`

### CADB 默认值

- `cadb_root = data/CADB_Dataset`
- `cadb_test_ratio = 0.2`
- `cadb_label_mode = exclusive`
- `cadb_balance_mode = none`
- `image_size = 96`

## 结果管理

- `results/`
  放原始实验输出，主要用于本地分析
- `docs/`
  放长期保留的项目说明、研究计划、日志和学习笔记

现在默认不把整个 `results/` 目录当作长期文档区。  
如果有少量必须长期保留的结果图，后续再单独决定单独目录，而不是默认把结果和文档混放。


## Git 工作流

换机器之前建议固定做这 4 步：

1. `git status`
2. 提交本次有价值的代码和文档
3. `git push`
4. 到另一台机器 `git pull`

通常优先提交：

- 代码改动
- `docs/` 下的说明文档
- 与主线实验直接相关的少量可复用配置
