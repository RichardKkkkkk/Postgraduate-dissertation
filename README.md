# ViT 研究工作区

这个仓库是一个面向论文实验的轻量研究平台。  
当前目标不是“堆很多训练脚本”，而是保持一个统一训练入口，在稳定实验协议下比较不同的 positional encoding 设计。

一句英文概括：

`Use one runner, register many model variants, and keep the experiment contract stable.`

## 当前主线

当前统一工程主线是：

- 一个训练入口：`train_cifar10_experiment.py`
- 一个模型与数据集注册表：`models/registry.py`
- 一个训练工具模块：`experiment_utils.py`
- 一个结果对比入口：`generate_comparison_report.py`

当前已经完成正式多 seed 对比的模型：

- `vit_baseline`
- `vit_learnable_position`
- `vit_row_sinusoidal`
- `vit_col_sinusoidal`
- `vit_additive_sinusoidal`
- `vit_additive_sinusoidal_shifted`
- `vit_multiplicative_sinusoidal`
- `vit_multiplicative_sinusoidal_shifted`

当前主要证据来自：

- `cifar10`：8 个位置编码模型、5 个 seed（42-46）的正式对比
- `cadb_elements`：8 个模型、seed 42 的探索性 multi-label 对比

当前待验证的老师方法扩展：

- `vit_squared_multiplicative_sinusoidal`
- `vit_squared_multiplicative_sinusoidal_shifted`
- `vit_radial_sinusoidal`

这三个模型已经实现并注册，但还没有正式训练结果。

## 环境

统一使用 conda 环境 `vit_research`，不要为本项目创建 `.venv`。

首次创建：

```bash
conda env create -f environment.yml
conda activate vit_research
```

如果本机 Conda 默认频道要求额外接受条款，也可以只使用项目指定的 `conda-forge` 创建基础环境：

```bash
conda create -n vit_research --override-channels -c conda-forge python=3.11 pip
conda activate vit_research
python -m pip install -r requirements.txt
```

已有环境时：

```bash
conda activate vit_research
```

### Windows + NVIDIA GPU

`requirements.txt` 从默认 PyPI 安装时可能得到 CPU 版 PyTorch。Windows NVIDIA 机器应在安装常规依赖后，把 PyTorch 替换为官方 CUDA wheel：

```bash
python -m pip install --force-reinstall --no-deps torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cu130
```

当前 Windows 工作站已经验证：

- Python `3.11.15`
- PyTorch `2.12.0+cu130`
- Torchvision `0.27.0+cu130`
- NVIDIA GeForce RTX 5070 Ti
- `torch.cuda.is_available() == True`

macOS 继续使用 `requirements.txt` 的标准 wheel；训练入口会优先选择 MPS，其次选择 CUDA，最后回退到 CPU。

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

- [train_cifar10_experiment.py](train_cifar10_experiment.py)
  统一训练入口
- [models/registry.py](models/registry.py)
  模型注册、数据集路由、默认超参数
- [experiment_utils.py](experiment_utils.py)
  训练、评估、早停、指标、画图、保存
- [datasets/cadb_data.py](datasets/cadb_data.py)
  CADB 任务定义和 dataloader
- [generate_comparison_report.py](generate_comparison_report.py)
  对比图和报告生成

更详细的结构说明见 [PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md)。

## 支持的模型

- `vit_baseline`
  不带 positional encoding 的 ViT 对照模型
- `vit_learnable_position`
  带 learned absolute positional embedding 的标准 ViT
- `vit_row_sinusoidal`
  只按 row 注入 sinusoidal positional embedding
- `vit_col_sinusoidal`
  只按 column 注入 sinusoidal positional embedding
- `vit_radial_sinusoidal`
  使用 `sqrt(row^2 + col^2)` 生成 radial sinusoidal positional embedding
- `vit_additive_sinusoidal`
  先分别生成 row / column sinusoidal embedding，再逐元素相加
- `vit_additive_sinusoidal_shifted`
  additive 版本，但 row / column 使用错开的 wavelength
- `vit_multiplicative_sinusoidal`
  先分别生成 row / column sinusoidal embedding，再逐元素相乘
- `vit_multiplicative_sinusoidal_shifted`
  multiplicative 版本，但 row / column 使用错开的 wavelength
- `vit_squared_multiplicative_sinusoidal`
  squared multiplicative 版本，对 `row_pe * col_pe` 再逐元素平方
- `vit_squared_multiplicative_sinusoidal_shifted`
  shifted squared multiplicative 版本
- `vit_normal_col_learnable_multiplicative_sinusoidal`
  hybrid PE 小实验：`normal_col` unfolding 下，将可学习 absolute PE 与 multiplicative fixed PE 相加，
  fixed 分支由一个可学习标量控制
- `vit_row_col_latent_fusion`
  双 encoder latent fusion：row-wise encoder 和 column-wise encoder 分别输出 cls latent，
  拼接后通过 fusion MLP，再用一个 shared prediction head 输出最终预测
- `vit_row_col_mean_fusion`
  双 encoder mean fusion：row-wise encoder 和 column-wise encoder 分别输出 cls latent，
  对两个 latent 逐元素取平均后直接用 shared prediction head 输出最终预测
- `vit_row_col_mean_mlp_fusion`
  双 encoder mean + NN fusion：先对 row/column cls latent 取平均，
  再通过一个输入输出维度相同的 fusion MLP，最后输出最终预测
- `vit_rope`
  1D RoPE 版本
- `vit_rope_2d`
  2D-aware RoPE 版本
- `resnet18_scratch`
  不带 pretrained weights 的 CNN baseline
- `resnet18_imagenet`
  可选的 ImageNet pretrained 参考模型

### Unfolding variants

当前支持 4 种 patch unfolding / flatten 方式：

- `normal_row`
  当前默认方式，逐行从左到右展开
- `normal_col`
  逐列从上到下展开
- `proper_row`
  逐行蛇形展开，偶数行左到右，奇数行右到左
- `proper_col`
  逐列蛇形展开，偶数列上到下，奇数列下到上

已有的原始模型名使用 `normal_row`。另外 3 种 unfolding 使用：

```text
vit_<unfolding>_<base_model>
```

例如：

```text
vit_proper_row_multiplicative_sinusoidal
vit_normal_col_learnable_position
vit_proper_col_row_sinusoidal
```

## 支持的数据集

### `cadb_elements`

这是已经完成正式探索性对比的 multi-label classification 任务。
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

## 当前 CIFAR-10 实验协议

已经完成的 8 模型、5 seed 实验使用：

- `dataset = cifar10`
- `seeds = 42, 43, 44, 45, 46`
- `epochs = 100`
- `batch_size = 128`
- `lr = 3e-4`
- `weight_decay = 0.05`
- `val_ratio = 0.1`
- `early_stopping_metric = val_acc`
- `early_stopping_patience = 10`
- `early_stopping_min_delta = 0.001`
- `lr_plateau_patience = 5`
- `lr_plateau_factor = 0.5`

现有汇总位于：

```text
results/cifar10_positional_8models_5seeds/reports/cifar10_positional_8models_5seeds_summary/
```

五 seed 的核心结果是：

- `vit_learnable_position`: test accuracy `78.854% ± 0.409 pp`
- `vit_multiplicative_sinusoidal_shifted`: `77.638% ± 0.388 pp`
- `vit_multiplicative_sinusoidal`: `77.458% ± 0.485 pp`
- `vit_baseline`: `71.390% ± 0.567 pp`
- `vit_learnable_position` 在 5/5 个 seed 上取得最高 test accuracy

## 当前下一组正式实验

老师提出的 squared multiplicative 和 radial 变体先统一跑 CIFAR-10 seed 42：

```bash
python train_cifar10_experiment.py --model vit_squared_multiplicative_sinusoidal --dataset cifar10 --epochs 100 --seed 42 --early-stopping-patience 10 --early-stopping-metric val_acc --early-stopping-min-delta 0.001 --lr-plateau-patience 5 --lr-plateau-factor 0.5 --experiment-name cifar10_teacher_extensions --run-name cifar10ext_vit_squared_multiplicative_sinusoidal_seed42
python train_cifar10_experiment.py --model vit_squared_multiplicative_sinusoidal_shifted --dataset cifar10 --epochs 100 --seed 42 --early-stopping-patience 10 --early-stopping-metric val_acc --early-stopping-min-delta 0.001 --lr-plateau-patience 5 --lr-plateau-factor 0.5 --experiment-name cifar10_teacher_extensions --run-name cifar10ext_vit_squared_multiplicative_sinusoidal_shifted_seed42
python train_cifar10_experiment.py --model vit_radial_sinusoidal --dataset cifar10 --epochs 100 --seed 42 --early-stopping-patience 10 --early-stopping-metric val_acc --early-stopping-min-delta 0.001 --lr-plateau-patience 5 --lr-plateau-factor 0.5 --experiment-name cifar10_teacher_extensions --run-name cifar10ext_vit_radial_sinusoidal_seed42
```

只有当某个新变体接近或超过现有 `vit_multiplicative_sinusoidal_shifted`，才扩展到 seed 43-46。

## 已完成的 CADB Elements 实验命令

### `vit_baseline`

```bash
python train_cifar10_experiment.py --model vit_baseline --dataset cadb_elements --cadb-root data/CADB_Dataset --image-size 96 --epochs 100 --seed 42 --early-stopping-patience 15 --early-stopping-metric val_macro_f1 --early-stopping-min-delta 0.001 --lr-plateau-patience 5 --lr-plateau-factor 0.5 --run-name baseline_cadb_elements_seed42
```

### `vit_learnable_position`

```bash
python train_cifar10_experiment.py --model vit_learnable_position --dataset cadb_elements --cadb-root data/CADB_Dataset --image-size 96 --epochs 100 --seed 42 --early-stopping-patience 15 --early-stopping-metric val_macro_f1 --early-stopping-min-delta 0.001 --lr-plateau-patience 5 --lr-plateau-factor 0.5 --run-name learnable_position_cadb_elements_seed42
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

### `vit_normal_col_learnable_multiplicative_sinusoidal`

```bash
python train_cifar10_experiment.py --model vit_normal_col_learnable_multiplicative_sinusoidal --dataset cifar10 --experiment-name cifar10_hybrid_seed42 --epochs 100 --seed 42 --early-stopping-patience 10 --early-stopping-metric val_acc --early-stopping-min-delta 0.001 --run-name normal_col_learnable_multiplicative_seed42
```

### `vit_row_col_latent_fusion`

```bash
python train_cifar10_experiment.py --model vit_row_col_latent_fusion --dataset cifar10 --experiment-name cifar10_latent_fusion_seed42 --epochs 100 --seed 42 --early-stopping-patience 10 --early-stopping-metric val_acc --early-stopping-min-delta 0.001 --run-name row_col_latent_fusion_seed42
```

### `vit_row_col_mean_fusion`

```bash
python train_cifar10_experiment.py --model vit_row_col_mean_fusion --dataset cifar10 --experiment-name cifar10_fusion_variants_seed42 --epochs 100 --seed 42 --early-stopping-patience 10 --early-stopping-metric val_acc --early-stopping-min-delta 0.001 --run-name row_col_mean_fusion_seed42
```

### `vit_row_col_mean_mlp_fusion`

```bash
python train_cifar10_experiment.py --model vit_row_col_mean_mlp_fusion --dataset cifar10 --experiment-name cifar10_fusion_variants_seed42 --epochs 100 --seed 42 --early-stopping-patience 10 --early-stopping-metric val_acc --early-stopping-min-delta 0.001 --run-name row_col_mean_mlp_fusion_seed42
```

### CADB 四模型核心对比报告

```bash
python generate_comparison_report.py --run baseline_cadb_elements_seed42="ViT Baseline (No Pos)" --run learnable_position_cadb_elements_seed42="ViT Learnable Position" --run row_cadb_elements_seed42="ViT Row-wise" --run col_cadb_elements_seed42="ViT Column-wise" --report-name cadb_elements_positional_controls --title "CADB Elements: Baseline vs Learnable Position vs Row-wise vs Column-wise" --skip-ppt
```


## 训练输出

每次训练完成后会保存：

- `results/<experiment_name>/metrics/<model>/<run_name>_metrics.csv`
- `results/<experiment_name>/metrics/<model>/<run_name>_config.json`
- `results/<experiment_name>/metrics/<model>/<run_name>_summary.json`
- `results/<experiment_name>/figures/<model>/<run_name>_loss.png`
- `results/<experiment_name>/figures/<model>/<run_name>_accuracy.png`
- `results/<experiment_name>/reports/<report_name>/...`
- `checkpoints/<experiment_name>/<model>/<run_name>_best.pt`

如果不显式传 `--experiment-name`，系统默认使用 `dataset` 名称作为实验目录。

单标签任务还会额外保存 confusion matrix。

核心实验原则：

- `validation` 用于 early stopping 和 model selection
- `test` 用于最终报告

已知待修正项：当前训练循环仍会在每个 epoch 计算 test 并保存 test 曲线。下一组论文正式实验前，应改成只在 validation 选定 checkpoint 后评估一次 test；详见 [RESEARCH_PLAN.md](docs/RESEARCH_PLAN.md)。

## 常用 CLI 参数

### 通用参数

- `--model`
- `--dataset`
- `--data-dir`
- `--results-dir`
- `--checkpoint-dir`
- `--experiment-name`
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
  放原始实验输出、图和报告
- `checkpoints/`
  放最佳模型参数
- `docs/`
  放长期保留的项目说明、研究计划、日志和学习笔记

当前仓库已经追踪了三组历史实验：

- `cadb_elements_positional_100e`
- `cifar10_positional_8models`
- `cifar10_positional_8models_5seeds`

历史上为了跨设备备份，已有结果和 56 个 checkpoint 被提交进 Git；因此 `.gitignore` 只会阻止新增的未追踪 checkpoint，不能自动取消已有文件的追踪。后续在决定 Git LFS、外部实验存储或清理方案之前，不要直接删除这些历史产物。

长期研究结论仍应写入 `docs/`，不能只存在于 `results/` 或聊天记录中。


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
