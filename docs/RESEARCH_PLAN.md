# 研究计划

这个文档回答两个问题：

1. 现在项目的研究主线是什么
2. 当前主线实验应该怎么跑、怎么解释

一句英文概括：

`Keep the training interface stable, then compare positional designs under a clean protocol.`

## 当前研究问题

当前更明确的研究问题是：

- ViT 在图像任务里到底需不需要 positional encoding
- 不同 positional encoding 设计是否会带来不同的图像结构偏置
- row-wise / column-wise positional prior 能否在合适的数据集上体现出来

## 当前主线模型

当前主线先收敛在这 4 个模型上：

- `vit_no_pos`
- `vit_baseline`
- `vit_row_sinusoidal`
- `vit_col_sinusoidal`

它们分别回答：

- `vit_no_pos`
  作为真正的无位置编码对照组
- `vit_baseline`
  作为标准 ViT baseline，使用 learned absolute positional embedding
- `vit_row_sinusoidal`
  只按 row 注入 sinusoidal positional embedding
- `vit_col_sinusoidal`
  只按 column 注入 sinusoidal positional embedding

当前不把 `RoPE` 和 `2D RoPE` 作为本周主线，但保留实现，方便后续继续扩展。

当前已经把下一阶段要用的两种 additive positional encoding 接进仓库：

- `vit_additive_sinusoidal`
- `vit_additive_sinusoidal_shifted`

它们暂时属于“下一阶段结构扩展”，还不是当前主线四模型的一部分。

## 当前主线数据集

当前主线优先任务是：

- `cadb_elements`

这是一个 multi-label classification 任务。  
每张图会输出一个标签向量，表示多个构图元素是否出现。

当前标签集合是：

- `horizontal`
- `vertical`
- `diagonal`
- `triangle`
- `symmetric`
- `pattern`

注意：

- 当前原始标注里，`pattern` 没有有效正样本
- `triangle` 和 `symmetric` 也偏稀疏

所以当前对 `CADB` 的态度是：

- 可以用来做对照实验
- 但不一定足够支撑最终最强结论

## 当前标准实验协议

主线实验默认使用：

- `dataset = cadb_elements`
- `seed = 42`
- `epochs = 100`
- `image_size = 96`
- `early_stopping_metric = val_macro_f1`
- `early_stopping_patience = 15`
- `early_stopping_min_delta = 0.001`
- `lr_plateau_patience = 5`
- `lr_plateau_factor = 0.5`

一句英文提醒：

`Validation is for model selection, test is for final reporting.`

## 当前标准 run name

- `no_pos_cadb_elements_seed42`
- `baseline_cadb_elements_seed42`
- `row_cadb_elements_seed42`
- `col_cadb_elements_seed42`

## 当前标准训练命令

```bash
python train_cifar10_experiment.py --model vit_no_pos --dataset cadb_elements --cadb-root data/CADB_Dataset --image-size 96 --epochs 100 --seed 42 --early-stopping-patience 15 --early-stopping-metric val_macro_f1 --early-stopping-min-delta 0.001 --lr-plateau-patience 5 --lr-plateau-factor 0.5 --run-name no_pos_cadb_elements_seed42
python train_cifar10_experiment.py --model vit_baseline --dataset cadb_elements --cadb-root data/CADB_Dataset --image-size 96 --epochs 100 --seed 42 --early-stopping-patience 15 --early-stopping-metric val_macro_f1 --early-stopping-min-delta 0.001 --lr-plateau-patience 5 --lr-plateau-factor 0.5 --run-name baseline_cadb_elements_seed42
python train_cifar10_experiment.py --model vit_row_sinusoidal --dataset cadb_elements --cadb-root data/CADB_Dataset --image-size 96 --epochs 100 --seed 42 --early-stopping-patience 15 --early-stopping-metric val_macro_f1 --early-stopping-min-delta 0.001 --lr-plateau-patience 5 --lr-plateau-factor 0.5 --run-name row_cadb_elements_seed42
python train_cifar10_experiment.py --model vit_col_sinusoidal --dataset cadb_elements --cadb-root data/CADB_Dataset --image-size 96 --epochs 100 --seed 42 --early-stopping-patience 15 --early-stopping-metric val_macro_f1 --early-stopping-min-delta 0.001 --lr-plateau-patience 5 --lr-plateau-factor 0.5 --run-name col_cadb_elements_seed42
```

## 当前标准对比命令

```bash
python generate_comparison_report.py --run no_pos_cadb_elements_seed42="ViT No Pos" --run baseline_cadb_elements_seed42="ViT Baseline" --run row_cadb_elements_seed42="ViT Row-wise" --run col_cadb_elements_seed42="ViT Column-wise" --report-name cadb_elements_positional_controls --title "CADB Elements: No Pos vs Baseline vs Row-wise vs Column-wise" --skip-ppt
```


## 当前结果解释原则

对于 `cadb_elements`，优先看：

- `val_macro_f1`
- `test_macro_f1`
- `per_class_f1`
- `subset_accuracy`

不要只看：

- `acc`

原因是当前 `acc` 在 multi-label 任务里更接近逐标签位准确率，容易因为负样本太多而偏高。

## 当前阶段判断

目前已经比较明确的判断有：

1. 之前 `row-wise / col-wise` 在 `CADB` 上没有清楚支撑最初假设。
2. 问题不一定只是模型，也可能来自数据标注和任务定义。
3. 所以现在更重要的是先把 `no_pos -> baseline -> row/col` 这条主线补完整。

## 当前不优先做的事

当前暂时不优先：

- 大规模 multi-seed sweep
- 大量 training tricks
- 大改 backbone
- 直接跳到复杂医疗数据集

原因很简单：

- 先把对照组和实验协议收紧
- 先验证 positional encoding 本身到底有没有带来可解释收益

## 下一步建议

当前更合理的顺序是：

1. 保持统一训练入口不再继续分叉
2. 用 4 个主线模型完成干净对照
3. 检查结果是否支持老师提出的新二维位置编码组合
4. 如果 `CADB` 仍然不够清楚，再换更适合的方向性数据集
