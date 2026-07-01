# 项目日志

这个文档记录：

- 改了什么
- 学到了什么
- 当前卡在哪里
- 下一步做什么

## 2026-06-30 当前状态收束

### 已完成

- 项目已经统一成一个训练入口：
  - `train_cifar10_experiment.py`
- 项目已经统一成一个模型注册表：
  - `models/registry.py`
- 当前主要模型分支包括：
  - `vit_no_pos`
  - `vit_baseline`
  - `vit_row_sinusoidal`
  - `vit_col_sinusoidal`
  - `vit_additive_sinusoidal`
  - `vit_additive_sinusoidal_shifted`
  - `vit_rope`
  - `vit_rope_2d`
- `cadb_elements` 已经接成 multi-label 任务
- 文档结构开始收束为：
  - `PROJECT_STRUCTURE.md`
  - `RESEARCH_PLAN.md`
  - `PROJECT_LOG.md`
  - `LEARNING_NOTES.md`

### 当前认识

- 之前 `row-wise / col-wise` 在 `CADB` 上没有清楚支撑最初假设
- `pattern` 标签没有有效正样本，说明原始标注本身存在问题
- `triangle` 和 `symmetric` 也比较稀疏，导致这些类更难学
- 因此当前更应该先补齐真正的对照组，而不是继续硬解释旧结果

### 当前主线

当前主线实验集中在：

- `vit_no_pos`
- `vit_baseline`
- `vit_row_sinusoidal`
- `vit_col_sinusoidal`

运行任务：

- `cadb_elements`

### 下一步

1. 继续保持项目结构和文档口径清晰
2. 决定哪些旧实验结果需要保留，哪些可以删除
3. 重新跑主线四模型对照
4. 再进入老师要求的新位置编码组合实验

## 2026-07-01 Additive PE 接入

### 已完成

- 新增 `vit_additive_sinusoidal`
- 新增 `vit_additive_sinusoidal_shifted`
- 两者都已经接入 `models/registry.py`
- 两者都可以通过统一入口 `train_cifar10_experiment.py --model ...` 直接训练

### 当前理解

- `vit_additive_sinusoidal`
  对应 `row_pe + col_pe`
- `vit_additive_sinusoidal_shifted`
  对应 row / column 使用错开的 wavelength 后再相加

### 下一步

1. 先做最小 smoke test
2. 再决定先在 `CADB` 还是更干净的数据集上跑 additive 系列

## 2026-07-01 Multiplicative PE 接入

### 已完成

- 新增 `vit_multiplicative_sinusoidal`
- 新增 `vit_multiplicative_sinusoidal_shifted`
- 两者都已经接入 `models/registry.py`
- 两者都可以通过统一入口 `train_cifar10_experiment.py --model ...` 直接训练

### 当前理解

- `vit_multiplicative_sinusoidal`
  对应 `row_pe * col_pe`
- `vit_multiplicative_sinusoidal_shifted`
  对应 row / column 使用错开的 wavelength 后再相乘

## 2026-07-01 CADB 八模型总报告

### 已完成

- `cadb_elements_positional_100e` 已经扩展到 8 个模型：
  - `vit_no_pos`
  - `vit_baseline`
  - `vit_row_sinusoidal`
  - `vit_col_sinusoidal`
  - `vit_additive_sinusoidal`
  - `vit_additive_sinusoidal_shifted`
  - `vit_multiplicative_sinusoidal`
  - `vit_multiplicative_sinusoidal_shifted`
- 已生成统一总报告：
  - `results/cadb_elements_positional_100e/reports/cadb_elements_positional_8models_report`
- 旧的 4 模型和 6 模型报告目录已删除，只保留 8 模型版本

### 下一步

1. 读取 8 模型总报告并比较 `macro_f1 / per_class_f1`
2. 决定是否继续保留 shifted 版本，还是开始筛掉弱分支
3. 再决定是否整理结果生成目录结构

## 2026-06-25 到 2026-06-29 阶段总结

### 这段时间主要做了什么

- 接入 `CADB` 的多标签分支
- 补了 `macro_f1`、`per_class_f1` 等多标签指标
- 生成了多组报告图和汇报材料
- 逐步发现当前 `CADB` 任务定义并不适合清晰验证最初的方向性假设

### 这段时间最重要的结论

- 当前 `CADB` 结果不能强有力支持：
  - row-wise 更适合 horizontal
  - column-wise 更适合 vertical
- 更大的原因不一定是代码错误，也可能是：
  - 数据集标注不干净
  - 任务本身不是纯方向性分类问题

## 当前待办

- [ ] 保持 `docs/` 文档口径一致
- [ ] 明确哪些旧结果要删、哪些保留
- [ ] 重跑新的四模型主线实验
- [ ] 再进入更复杂的二维位置编码组合实验
