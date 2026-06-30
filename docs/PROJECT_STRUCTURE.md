# 项目结构

这个文档只回答一个问题：

**当前仓库里，每个目录和关键文件负责什么。**

## 根目录

- `train_cifar10_experiment.py`
  统一训练入口
- `experiment_utils.py`
  训练、评估、指标、早停、结果保存
- `generate_comparison_report.py`
  读取实验结果并生成对比图和报告
- `run_seed_sweep.py`
  多 seed 批量运行入口
- `summarize_seed_sweep.py`
  多 seed 结果汇总
- `result_paths.py`
  统一管理结果路径

## `models/`

- `vit.py`
  标准 ViT，带 learned absolute positional embedding
- `vit_no_pos.py`
  不带 positional encoding 的 ViT 对照模型
- `vit_axis_sinusoidal.py`
  row-wise 和 column-wise sinusoidal 版本
- `vit_rope.py`
  1D RoPE 版本
- `vit_rope_2d.py`
  2D RoPE 版本
- `registry.py`
  模型注册、数据集路由、默认超参数

一句英文解释：

`registry.py is the switchboard of the project.`

## `datasets/`

- `cifar10_data.py`
  CIFAR-10 dataloader
- `cadb_data.py`
  CADB 标签解析、split 逻辑、dataloader
- `synthetic_orientation_data.py`
  synthetic datasets 的生成与加载

## `docs/`

- `PROJECT_STRUCTURE.md`
  项目结构说明
- `RESEARCH_PLAN.md`
  当前研究主线和实验协议
- `PROJECT_LOG.md`
  项目日志
- `LEARNING_NOTES.md`
  学习笔记

## 运行产物目录

- `data/`
  本地数据集，不提交
- `checkpoints/`
  模型参数，不提交
- `results/`
  原始实验输出

## 推荐理解顺序

如果要熟悉项目代码，建议按这个顺序看：

1. `README.md`
2. `train_cifar10_experiment.py`
3. `models/registry.py`
4. `experiment_utils.py`
5. `models/vit.py`
6. `models/vit_no_pos.py`
7. `models/vit_axis_sinusoidal.py`
8. `datasets/cadb_data.py`
