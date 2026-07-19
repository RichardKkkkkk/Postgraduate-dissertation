# 项目日志

这个文档记录三件事：

- 最近改了什么
- 当前知道了什么
- 下一步该做什么

## 2026-06-30 项目收束

### 已完成

- 训练入口统一为 `train_cifar10_experiment.py`
- 模型和数据集统一通过 `models/registry.py` 管理
- 当前主线模型包括：
  - `vit_baseline`
  - `vit_learnable_position`
  - `vit_row_sinusoidal`
  - `vit_col_sinusoidal`
  - `vit_additive_sinusoidal`
  - `vit_additive_sinusoidal_shifted`
  - `vit_multiplicative_sinusoidal`
  - `vit_multiplicative_sinusoidal_shifted`
  - `vit_rope`
  - `vit_rope_2d`
- `cadb_elements` 已接成 multi-label 任务

### 当前认识

- CADB 上 row-wise / column-wise 的差异没有像最初预期那样明显
- `pattern` 标签本身不可靠，不能作为强证据
- 当前更需要 clean protocol 和可解释实验，而不是继续堆很多 trick

## 2026-07-01 位置编码扩展

### 已完成

- 接入 additive sinusoidal variants
- 接入 multiplicative sinusoidal variants
- 这些模型都已经注册到统一训练入口里
- 已支持统一生成 comparison report 和 PPT

### 当前认识

- additive / multiplicative 系列已经具备工程可运行性
- 后续重点应放在更干净的数据集设计和更清楚的对照实验上

## 2026-07-02 结果目录对齐

### 已完成

- 保留 `--experiment-name`，让新实验可以按实验名归档
- 结果结构对齐回 CADB 风格：
  - `results/<experiment_name>/metrics/<model>/...`
  - `results/<experiment_name>/figures/<model>/...`
  - `results/<experiment_name>/reports/<report_name>/...`
  - `checkpoints/<experiment_name>/<model>/<run_name>_best.pt`
- 保留对旧版 `results/metrics` / `results/figures` 的读取兼容
- 保留对误建的深层 `runs/...` 目录的读取兼容，避免之前的结果直接失效

### 当前认识

- 你真正想要的是“每次实验一个总文件夹”，不是再往里面加一层很深的 run 目录
- 这样和之前 `cadb_elements_positional_100e` 的使用习惯一致，也更容易人工查看

## 2026-07-02 当时待办（后续状态已更新）

- [x] 用新的 `experiment_name` 口径重新跑一组 CIFAR-10 对比
- [x] 检查 comparison report 是否完整落在同一个 experiment 文件夹下
- [ ] 决定哪些旧结果需要保留，哪些可以清理
- [x] 接入老师要求的 squared multiplicative 和 radial positional encoding
- [ ] 为 squared multiplicative 和 radial positional encoding 生成正式结果

## 2026-07-04 Multi-Seed Summary Plot Update

### 已完成

- `summarize_seed_sweep.py` 现在除了最终 mean/std 表格外，还会输出按 `epoch` 聚合的 mean ± std 曲线
- 新图会对每个模型画均值曲线，并用阴影显示标准差
- 这些曲线直接从每个 run 的 `metrics.csv` 聚合得到
- 后续 multi-seed 汇总不再保留旧版柱状图和按 seed 折线图，主图改为 epoch-wise mean ± std

### 当前认识

- 对老师汇报来说，`epoch-wise mean ± std` 比只看最终柱状图更有解释力
- 这样能直接看到：
  - 收敛速度
  - 波动大小
  - 不同模型是否稳定

## 2026-07-08 Teacher Method Expansion: Squared Multiplicative PE

### 已完成

- 根据老师的新实验建议，新增两个 CIFAR-10 positional encoding 候选模型：
  - `vit_squared_multiplicative_sinusoidal`
  - `vit_squared_multiplicative_sinusoidal_shifted`
- 两个模型都通过 `models/registry.py` 接入统一训练入口
- 修复 `experiment_utils.evaluate()` 在 very small subset smoke test 上的 confusion matrix 类别数推断问题
- squared multiplicative 的核心定义是对 multiplicative PE 再逐元素平方：
  - normal: `(row_pe * col_pe) ** 2`
  - shifted: `(shifted_row_pe * shifted_col_pe) ** 2`

### 当前认识

- 这是对之前 “multiplicative 优于 additive” 结果的直接外推实验
- 平方会去掉乘积的正负号，因此它测试的是更强的 row/column 耦合强度，而不是相位符号
- 两个新模型已经通过 1-epoch CIFAR-10 subset smoke test
- 下一步可以在 CIFAR-10 上跑正式对比

## 2026-07-10 Teacher Method Expansion: Radial PE

### 已完成

- 新增 `vit_radial_sinusoidal`
- 按老师邮件原文实现 radial distance：
  - `r = sqrt(row^2 + col^2)`
  - 原点是 patch grid 左上角 `(0, 0)`
- 模型已接入 `models/registry.py`，可通过统一训练入口运行

### 下一步

- 先跑 CIFAR-10 seed42 单实验
- 如果结果接近或超过 multiplicative / squared multiplicative，再考虑 multi-seed

## 2026-07-20 Windows 工作站环境与文档对齐

### 新工作站环境

- 已创建 Conda 环境 `vit_research`
- Python `3.11.15`
- PyTorch `2.12.0+cu130`
- Torchvision `0.27.0+cu130`
- GPU 为 NVIDIA GeForce RTX 5070 Ti
- 已验证 `torch.cuda.is_available() == True`
- 已在 CUDA 上完成基础 ViT 前向测试，输入 `(8, 3, 32, 32)`，输出 `(8, 10)`
- 15 个注册模型均可正常导入
- 当前 Windows 工作站尚未放置 `data/` 数据目录

### 文档与实际状态的不一致

- README 和研究计划仍把 CADB 四模型写成当前主线，但实际上已经完成 CIFAR-10 八模型、五 seed 对比
- squared multiplicative 和 radial 已实现，但还没有正式结果
- 研究计划中的 `no_pos` / `baseline` 命名与代码现状混用
- README 包含旧电脑绝对路径和一行乱码
- 历史 checkpoint 与 results 实际已被 Git 追踪，与“checkpoint 不提交”的后续规则不完全一致
- 训练代码每个 epoch 都计算 test；虽然 checkpoint 只按 validation 选择，但这不符合最严格的最终 holdout 叙事

### 本次选择的对齐方案

- 当前主要证据改为 CIFAR-10 八模型五 seed 结果
- CADB 改为探索性补充证据，并明确不能用逐标签位 accuracy 作为主结论
- `vit_baseline` 统一表示无位置编码，`vit_learnable_position` 表示标准 learned position baseline
- 下一组实验统一归档到 `cifar10_teacher_extensions`
- 在跑下一组正式实验前，优先修正 test 只在 selected checkpoint 上评估一次的流程
- 未制定 Git LFS 或外部存储方案前，不清理历史结果与 checkpoint

### 当前五 seed 结论

- `vit_learnable_position` 平均 test accuracy 为 `78.854% ± 0.409 pp`，并赢得 5/5 个 seed
- `vit_multiplicative_sinusoidal_shifted` 为 `77.638% ± 0.388 pp`，是当前最接近 learned position 的固定位置编码
- `vit_baseline` 为 `71.390% ± 0.567 pp`
- 当前证据支持位置编码有效，也支持继续研究 multiplicative coupling，但还不支持“自定义固定 PE 超过 learned PE”

### 下一步

1. 修正 test holdout 流程并运行 smoke test
2. 正式运行 squared multiplicative、shifted squared multiplicative 和 radial 的 CIFAR-10 seed 42
3. 只把有竞争力的新模型扩展到 seed 43-46
