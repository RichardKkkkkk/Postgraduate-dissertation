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

训练入口 `--model` 的真实 choices 来自 `models/registry.py` 的 `EXPERIMENT_REGISTRY`。
需要核对完整列表时运行：

```bash
python -c "from models.registry import EXPERIMENT_REGISTRY; print('\n'.join(EXPERIMENT_REGISTRY.keys()))"
```

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
- `vit_row_col_cross_attention_fusion`
  双 encoder bidirectional cross-attention fusion：row token sequence 用 column token sequence 做 K/V，
  column token sequence 用 row token sequence 做 K/V，两个方向更新后的 cls token 拼接后预测
- `vit_row_col_cross_attention_mlp_head_fusion`
  cross-attention fusion 的 smoother-head refinement：主体与 cross-attention fusion 相同，
  但最终分类头从 `Linear(256, num_classes)` 改为 `LayerNorm -> Linear(256, 128) -> GELU -> Linear(128, num_classes)`
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

### `vit_row_col_cross_attention_fusion`

```bash
python train_cifar10_experiment.py --model vit_row_col_cross_attention_fusion --dataset cifar10 --experiment-name cifar10_fusion_variants_seed42 --epochs 100 --seed 42 --early-stopping-patience 10 --early-stopping-metric val_acc --early-stopping-min-delta 0.001 --run-name row_col_cross_attention_fusion_seed42
```

### `vit_row_col_cross_attention_mlp_head_fusion`

```bash
python train_cifar10_experiment.py --model vit_row_col_cross_attention_mlp_head_fusion --dataset cifar10 --experiment-name cifar10_fusion_variants_seed42 --epochs 100 --seed 42 --early-stopping-patience 10 --early-stopping-metric val_acc --early-stopping-min-delta 0.001 --run-name row_col_cross_attention_mlp_head_fusion_seed42
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
- `results/<experiment_name>/figures/<model>/<run_name>_loss.pdf`
- `results/<experiment_name>/figures/<model>/<run_name>_accuracy.png`
- `results/<experiment_name>/figures/<model>/<run_name>_accuracy.pdf`
- `results/<experiment_name>/reports/<report_name>/...`
- `checkpoints/<experiment_name>/<model>/<run_name>_best.pt`

如果不显式传 `--experiment-name`，系统默认使用 `dataset` 名称作为实验目录。

单标签任务还会额外保存 confusion matrix。

核心实验原则：

- `validation` 用于 early stopping 和 model selection
- `test` 只在 validation-selected checkpoint 上评估一次，用于最终报告
- 单模型和模型对比的 epoch 曲线默认只画 train / validation
- PNG 使用 300 dpi，同时保存 PDF 矢量版本
- 所有论文/报告图的样式统一来自 `paper_plotting.py`

完整绘图规范见 [docs/FIGURE_STANDARD.md](docs/FIGURE_STANDARD.md)。

已有 metrics 不需要重新训练即可刷新单模型图：

```bash
python refresh_single_run_figures.py --experiment-name <experiment_name>
```

模型对比报告默认生成 `val_loss`、`val_acc`、`train_loss`、`train_acc`。也可以用
`--metrics` 显式指定：

```bash
python generate_comparison_report.py --run <run_a>="Model A" --run <run_b>="Model B" --metrics val_loss val_acc train_loss train_acc --report-name <report_name> --skip-ppt
```

每个 comparison report 还会额外生成一组更接近论文/一区文章结果展示习惯的文件：

- `figures/val_loss_comparison.png` 和 `.pdf`
- `figures/val_acc_comparison.png` 和 `.pdf`
- `figures/train_loss_comparison.png` 和 `.pdf`
- `figures/train_acc_comparison.png` 和 `.pdf`
  - 每张图只比较一个指标，避免在探索阶段把多个结论挤到一张图里
- `figures/paper_selected_test_accuracy.png` 和 `.pdf`
  - validation-selected checkpoint 上的 test accuracy 汇总
- `publication_selected_checkpoints.csv`
  - 每个模型的 selected epoch 和 selected test metrics
- `figure_captions.md`
  - 可作为论文图注或 weekly email 草稿的 caption

当前训练循环已经采用 final holdout 协议：每个 epoch 只评估 train/validation，训练结束后加载
validation-selected checkpoint，再对 test split 评估一次。历史旧结果 CSV 中仍可能包含逐 epoch
test 指标；正式论文结果应使用新协议重新跑。

## 最终 CIFAR-10 multi-seed 协议

第一轮最终实验建议固定：

- dataset: `cifar10`
- train/validation split: CIFAR-10 train set 中 `val_ratio = 0.1`，固定 `split_seed = 42`
- test split: CIFAR-10 official test set，只在 selected checkpoint 上评估一次
- seeds: `42 43 44 45 46`
- epochs: `100`
- batch size: `128`
- learning rate: `3e-4`
- weight decay: `0.05`
- early stopping metric: `val_acc`
- early stopping patience: `10`
- early stopping min delta: `0.001`
- LR scheduler: ReduceLROnPlateau, patience `5`, factor `0.5`, min lr `1e-6`
- subsets: 不使用 `--train-subset` / `--val-subset` / `--test-subset`

示例入口：

```bash
python run_seed_sweep.py \
  --dataset cifar10 \
  --all-models \
  --exclude-models resnet18_imagenet \
  --seeds 42 43 44 45 46 \
  --split-seed 42 \
  --experiment-name cifar10_final_all_models_5seeds \
  --run-prefix cifar10final \
  --report-prefix cifar10final_compare \
  --epochs 100 \
  --batch-size 128 \
  --lr 3e-4 \
  --weight-decay 0.05 \
  --val-ratio 0.1 \
  --early-stopping-patience 10 \
  --early-stopping-metric val_acc \
  --early-stopping-min-delta 0.001 \
  --lr-plateau-patience 5 \
  --lr-plateau-factor 0.5 \
  --lr-plateau-min-lr 1e-6
```

`--all-models` 会运行 `models/registry.py` 里注册的所有模型。上面示例排除了
`resnet18_imagenet`，因为它可能需要下载 torchvision 的 ImageNet 预训练权重；如果本机已经缓存权重，或者网络稳定，可以删掉这一行。

## 常用 CLI 参数

### 通用参数

- `--model`
- `--dataset`
- `--all-models`
- `--exclude-models`
- `--data-dir`
- `--results-dir`
- `--checkpoint-dir`
- `--experiment-name`
- `--run-name`
- `--epochs`
- `--batch-size`
- `--split-seed`
- `--lr`
- `--weight-decay`
- `--train-subset`
- `--val-subset`
- `--test-subset`
- `--val-ratio`
- `--num-workers`
- `--seed`
- `--image-size`
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

### Synthetic dataset 参数

- `--synthetic-train-size`
- `--synthetic-val-size`
- `--synthetic-test-size`
- `--synthetic-line-width`
- `--synthetic-noise-std`
- `--synthetic-max-stripes`

### CADB 参数

- `--cadb-root`
- `--cadb-test-ratio`
- `--cadb-label-mode`
- `--cadb-balance-mode`

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

### Synthetic dataset 默认值

- `synthetic_train_size = 2400`
- `synthetic_val_size = 600`
- `synthetic_test_size = 600`
- `synthetic_line_width = 3`
- `synthetic_noise_std = 0.08`
- `synthetic_max_stripes = 4`

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

本地还可能存在 `cifar10_positional_squared_seed42`、`cifar10_positional_radial_seed42`、
`cifar10_unfolding_15_seed42`、`cifar10_fusion_variants_seed42` 等探索性结果目录。
这些目录用于筛选方向和调试图形规范，正式论文统计结果需要按最终协议重新 multi-seed 跑。

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

## 论文主结果图

最终 CIFAR-10 selected-checkpoint multi-seed 结果使用独立脚本生成论文图，不再通过 Word 草稿构建脚本绘图：

```bash
conda run -n vit_research python generate_thesis_figures.py
```

默认读取：

```text
results/cifar10_final_vit_models_5seeds/metrics/
```

默认输出：

```text
results/cifar10_final_vit_models_5seeds/reports/thesis_comparison_figures_v2/
```

当前同时生成训练动态曲线和 selected-checkpoint test 总结图。训练动态曲线以
`Epoch` 为横坐标，并使用 validation 指标：

- `basic_pe_validation_dynamics`：基础 PE 的 validation accuracy / loss
- `shift_validation_dynamics`：shift variants 的 validation accuracy / loss
- `patch_assignment_val_acc_epoch`：不同 assignment 的 validation accuracy
- `patch_assignment_val_loss_epoch`：不同 assignment 的 validation loss
- `fusion_validation_dynamics`：fusion variants 的 validation accuracy / loss

selected-checkpoint test 总结图包括：

- `basic_pe_comparison`：No PE、learnable、row、column、additive、multiplicative
- `shift_paired_effect`：additive 与 multiplicative 的 shifted-minus-unshifted 配对差值
- `patch_assignment_paired_deltas`：不同 assignment 相对 row-major 的 seed-level 配对差值
- `patch_assignment_schematic`：四种 8 × 8 patch-to-position assignment
- `fusion_capacity_comparison`：single-encoder references、fusion accuracy 和参数量

脚本会在绘图前检查 seed 覆盖、共享训练配置和
`test_evaluation_protocol = selected_checkpoint_only`，并同步输出 source CSV、manifest 和图注草稿。
Epoch 曲线显示五个 seeds 的 mean ± pointwise 95% t CI；为避免 early stopping 后样本数随 epoch
变化，曲线只画到五个 seeds 都有记录的最后一个 epoch。test 不作逐 epoch 绘图。
v2 的 epoch 曲线不使用三角形、方形或菱形 marker；模型由固定的高对比颜色和线型共同
区分，阴影表示逐 epoch 的 95% t CI。selected-checkpoint test 汇总图中的半透明圆点
表示单个 seed，菱形表示五个 seed 的均值，竖向 error bar 表示 95% t confidence interval。
## Dissertation workflow update (2026-08-07)

The dissertation now uses one fixed working file:

`thesis/Yikai_Zhao_MSc_Dissertation.docx`

The completed CIFAR-10 evidence consists of 32 ViT configurations × seeds 42–46 = 160 selected-checkpoint summaries. Radial and squared fixed encodings, the learned–fixed hybrid, all four patch-assignment conventions, and five fusion models are included in this completed set.

New reproducibility utilities:

- `generate_thesis_statistics.py`: regenerates per-seed metrics, five-seed means, sample SDs, 95% t intervals, paired contrasts, parameter counts and the selected hybrid scale.
- `generate_patch_mapping_report.py`: records physical patch coordinate → sequence slot → assigned PE coordinate for every supported mapping.
- `run_low_data_sweep.py`: runs the prespecified 1k/5k/10k low-data matrix for learned and multiplicative PE.
- `datasets/cifar100_data.py`: provides the CIFAR-100 loader and dataset-specific normalisation.
- `thesis/tools/build_dissertation.py`: rebuilds the fixed Arial, black, no-figure dissertation draft from verified summaries.

Supported datasets now include `cifar100`. CIFAR-100 keeps the 32×32 input interface and changes the classifier to 100 outputs. The main cross-dataset sweep is intentionally limited to no PE, learned PE, shifted additive PE and shifted multiplicative PE.

## Robustness figure package (2026-08-10)

Generate the completed reduced-data CIFAR-10 and CIFAR-100 thesis figures with:

```powershell
conda run -n vit_research python generate_robustness_figures.py
```

The script validates seeds 42--46 and the
`selected_checkpoint_only` protocol before reading any test metric. It writes
PNG/PDF figures, source CSV files, draft captions and a manifest under:

```text
results/reports/thesis_robustness_figures_v2/
```

The current reduced-data result set contains four models at 1k, 5k and 10k
training examples and uses `lr=1e-3`. The completed CIFAR-100 comparison also
uses `lr=1e-3`. The existing final CIFAR-10 full-data suite uses `lr=3e-4`, so
the plotting script intentionally does not connect that full-data point to the
reduced-data trend. `run_low_data_sweep.py` still reflects the earlier two-model
`lr=3e-4` proposal and must not be used as the reproduction command for the
completed four-model result directories.

The v2 robustness package presents the 1k, 5k and 10k conditions as separate
epoch-based panels for validation accuracy and validation loss. CIFAR-100 also
has epoch-based validation accuracy and loss figures. Selected-test figures use
a categorical model x-axis because each test set is evaluated only once after
checkpoint selection; faint circles are seed outcomes, diamonds are means and
error bars are 95% t confidence intervals.

## Frozen selected-test thesis package (2026-08-12)

The protocol-aligned low-data and CIFAR-100 reruns use `lr=3e-4` and are stored
under experiment names ending in `_lr3e4`. Generate the teacher-requested core
table and seven selected-test figures with:

```powershell
conda run -n vit_research python generate_final_test_figures.py
```

The default output is:

```text
results/reports/thesis_selected_test_figures_v1/
```

The package contains the nine-model core PE table (including radial PE), five
main figures, two supporting figures, per-seed and summary CSV files, captions,
a configuration-alignment audit and a manifest. Every performance value is read
from `summary["selected_model"]`. Ordinary error bars use five-seed 95% t
intervals; paired-effect intervals are calculated from seed-level differences.

The low-data figure connects 1k, 5k, 10k and full-data only after an automated
gate verifies the shared learning rate, scheduler, augmentation, normalisation,
split seed, batch size, weight decay, early stopping, model identifiers and test
protocol. The older `lr=1e-3` robustness package remains exploratory and must
not be used as the source for the frozen thesis result tables.

## Final thesis evidence package (2026-08-12)

The paper-facing presentation now separates optimisation evidence from final
performance evidence. Generate the complete package with:

```powershell
conda run -n vit_research python generate_final_thesis_evidence.py
```

The default output is `results/reports/thesis_final_evidence_figures_v1/`.
Six main figures use validation accuracy or validation loss against epoch. The
lines are five-seed means and shaded bands are pointwise 95% t confidence
intervals. Each condition ends at the last epoch shared by all five seeds.

Final comparisons use selected-checkpoint test tables for core PE, low-data,
CIFAR-100 and fusion. Four auxiliary selected-test figures cover paired shifted
effects, patch-assignment deltas, fusion accuracy versus parameter count and
per-class recall differences. No test metric is plotted against epoch.

## Coordinate-aligned unfolding experiment (2026-08-13)

Historical unfolding models use `position_assignment = sequence_slot`: patch
tokens follow the selected unfolding order while fixed PE remains indexed by
sequence slot. This remains the default and preserves every existing model and
result. The new `coordinate_aligned` mode applies the same `patch_order` to the
patch part of fixed PE; the CLS entry is not reordered.

Run the deterministic mapping and forward-equivalence audit with:

```powershell
conda run -n vit_research python audit_coordinate_aligned_unfolding.py
```

Run or safely resume the formal CIFAR-10 sweep with:

```powershell
conda run -n vit_research python run_coordinate_aligned_unfolding_sweep.py --skip-existing
```

The wrapper first validates 25 protocol-matched normal-row source summaries,
then trains only the 75 non-normal-row conditions under
`results/cifar10_coordinate_aligned_unfolding_5seeds/`. After successful
training it checks all artifacts and writes coordinate-aligned and legacy
sequence-slot tables separately. Registered coordinate-aligned model names use
the compact prefix `vit_ca_`; `ca` means coordinate-aligned.
