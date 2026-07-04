# CIFAR-10 Positional Encoding Comparison

Generated: 2026-07-04T15:12:20
Comparison scenario: `generic_multi_run`
Reference run: `ViT Baseline (No Pos)`

## Headline Takeaways

- Primary comparison metric: Test ACC. Best final result is ViT Learnable Position with 78.03%.
- Against reference run ViT Baseline (No Pos), ViT Learnable Position changes Test ACC by +6.61 pp.
- Largest train-to-test accuracy gap appears in ViT Baseline (No Pos) (23.58 pp), worth calling out during generalization discussion.
- Early stopping summary: ViT Baseline (No Pos): stopped early; ViT Learnable Position: stopped early; ViT Row-wise Sinusoidal: stopped early

## Runs

- `ViT Baseline (No Pos)` (`vit_baseline_cifar10_seed42`), model: `ViT Baseline (No Pos)`, family: `vit`, variant: `baseline`, position encoding: `none`, epochs completed: `59`, device: `cuda`
- `ViT Learnable Position` (`vit_learnable_position_cifar10_seed42`), model: `ViT Learnable Position`, family: `vit`, variant: `learnable_position`, position encoding: `absolute`, epochs completed: `66`, device: `cuda`
- `ViT Row-wise Sinusoidal` (`vit_row_cifar10_seed42`), model: `ViT Row-wise Sinusoidal`, family: `vit`, variant: `row_sinusoidal`, position encoding: `row_sinusoidal`, epochs completed: `61`, device: `cuda`
- `ViT Column-wise Sinusoidal` (`vit_col_cifar10_seed42`), model: `ViT Column-wise Sinusoidal`, family: `vit`, variant: `col_sinusoidal`, position encoding: `col_sinusoidal`, epochs completed: `50`, device: `cuda`
- `ViT Additive Sinusoidal` (`vit_additive_cifar10_seed42`), model: `ViT Additive Sinusoidal`, family: `vit`, variant: `additive_sinusoidal`, position encoding: `additive_sinusoidal`, epochs completed: `70`, device: `cuda`
- `ViT Additive Sinusoidal Shifted` (`vit_additive_shifted_cifar10_seed42`), model: `ViT Additive Sinusoidal Shifted`, family: `vit`, variant: `additive_sinusoidal_shifted`, position encoding: `additive_sinusoidal_shifted`, epochs completed: `67`, device: `cuda`
- `ViT Multiplicative Sinusoidal` (`vit_multi_cifar10_seed42`), model: `ViT Multiplicative Sinusoidal`, family: `vit`, variant: `multiplicative_sinusoidal`, position encoding: `multiplicative_sinusoidal`, epochs completed: `48`, device: `cuda`
- `ViT Multiplicative Sinusoidal Shifted` (`vit_multi_shifted_cifar10_seed42`), model: `ViT Multiplicative Sinusoidal Shifted`, family: `vit`, variant: `multiplicative_sinusoidal_shifted`, position encoding: `multiplicative_sinusoidal_shifted`, epochs completed: `70`, device: `cuda`

## Key Metrics

### ViT Baseline (No Pos)
- Test ACC: final 71.42%, best 71.62% at epoch 58
- Val ACC: final 71.36%, best 71.84% at epoch 49
- Test Loss: final 1.1672, best 0.8904 at epoch 32
- Val Loss: final 1.1645, best 0.8650 at epoch 32
- Train ACC: final 95.00%, best 95.00% at epoch 59
- Train Loss: final 0.1457, best 0.1457 at epoch 59
- Val Macro F1: final 71.24%, best 71.59% at epoch 49
- Test Macro F1: final 71.35%, best 71.53% at epoch 58
- Selected Test Macro Precision: 70.96%
- Selected Test Macro Recall: 71.04%
- Selected Val Macro Precision: 71.71%
- Selected Val Macro Recall: 71.63%

### ViT Learnable Position
- Test ACC: final 78.03%, best 78.88% at epoch 56
- Val ACC: final 78.88%, best 79.24% at epoch 60
- Test Loss: final 0.9615, best 0.6784 at epoch 30
- Val Loss: final 0.9667, best 0.6858 at epoch 32
- Train ACC: final 97.74%, best 97.74% at epoch 66
- Train Loss: final 0.0658, best 0.0658 at epoch 66
- Val Macro F1: final 78.85%, best 79.09% at epoch 60
- Test Macro F1: final 78.05%, best 78.92% at epoch 50
- Selected Test Macro Precision: 78.79%
- Selected Test Macro Recall: 78.88%
- Selected Val Macro Precision: 79.03%
- Selected Val Macro Recall: 78.98%

### ViT Row-wise Sinusoidal
- Test ACC: final 74.62%, best 74.90% at epoch 60
- Val ACC: final 74.92%, best 75.18% at epoch 58
- Test Loss: final 1.1507, best 0.7810 at epoch 27
- Val Loss: final 1.1300, best 0.7617 at epoch 27
- Train ACC: final 97.86%, best 97.86% at epoch 61
- Train Loss: final 0.0695, best 0.0695 at epoch 61
- Val Macro F1: final 74.80%, best 75.07% at epoch 51
- Test Macro F1: final 74.58%, best 74.82% at epoch 51
- Selected Test Macro Precision: 74.96%
- Selected Test Macro Recall: 74.79%
- Selected Val Macro Precision: 75.25%
- Selected Val Macro Recall: 75.03%

### ViT Column-wise Sinusoidal
- Test ACC: final 74.21%, best 74.33% at epoch 40
- Val ACC: final 74.78%, best 75.56% at epoch 44
- Test Loss: final 0.8657, best 0.7819 at epoch 40
- Val Loss: final 0.8380, best 0.7405 at epoch 40
- Train ACC: final 88.85%, best 88.85% at epoch 50
- Train Loss: final 0.3096, best 0.3096 at epoch 50
- Val Macro F1: final 74.51%, best 75.36% at epoch 44
- Test Macro F1: final 74.05%, best 74.23% at epoch 40
- Selected Test Macro Precision: 74.30%
- Selected Test Macro Recall: 74.33%
- Selected Val Macro Precision: 75.49%
- Selected Val Macro Recall: 75.32%

### ViT Additive Sinusoidal
- Test ACC: final 76.05%, best 77.68% at epoch 52
- Val ACC: final 77.14%, best 77.70% at epoch 64
- Test Loss: final 0.8797, best 0.6946 at epoch 44
- Val Loss: final 0.8371, best 0.6896 at epoch 44
- Train ACC: final 92.24%, best 92.30% at epoch 69
- Train Loss: final 0.2163, best 0.2163 at epoch 70
- Val Macro F1: final 76.71%, best 77.73% at epoch 64
- Test Macro F1: final 75.77%, best 77.69% at epoch 52
- Selected Test Macro Precision: 77.04%
- Selected Test Macro Recall: 77.01%
- Selected Val Macro Precision: 77.65%
- Selected Val Macro Recall: 77.49%

### ViT Additive Sinusoidal Shifted
- Test ACC: final 77.19%, best 77.62% at epoch 58
- Val ACC: final 77.58%, best 77.78% at epoch 57
- Test Loss: final 0.8616, best 0.7090 at epoch 44
- Val Loss: final 0.8473, best 0.7049 at epoch 38
- Train ACC: final 94.49%, best 94.49% at epoch 67
- Train Loss: final 0.1572, best 0.1572 at epoch 67
- Val Macro F1: final 77.50%, best 77.68% at epoch 57
- Test Macro F1: final 77.25%, best 77.49% at epoch 58
- Selected Test Macro Precision: 77.11%
- Selected Test Macro Recall: 76.92%
- Selected Val Macro Precision: 77.81%
- Selected Val Macro Recall: 77.66%

### ViT Multiplicative Sinusoidal
- Test ACC: final 76.18%, best 76.86% at epoch 47
- Val ACC: final 77.14%, best 77.72% at epoch 43
- Test Loss: final 0.7343, best 0.6938 at epoch 43
- Val Loss: final 0.7078, best 0.6662 at epoch 38
- Train ACC: final 85.94%, best 85.94% at epoch 48
- Train Loss: final 0.3934, best 0.3934 at epoch 48
- Val Macro F1: final 76.79%, best 77.59% at epoch 43
- Test Macro F1: final 75.90%, best 76.77% at epoch 47
- Selected Test Macro Precision: 76.62%
- Selected Test Macro Recall: 76.61%
- Selected Val Macro Precision: 77.51%
- Selected Val Macro Recall: 77.37%

### ViT Multiplicative Sinusoidal Shifted
- Test ACC: final 77.56%, best 78.18% at epoch 69
- Val ACC: final 78.48%, best 79.18% at epoch 62
- Test Loss: final 0.8706, best 0.6965 at epoch 36
- Val Loss: final 0.8236, best 0.6791 at epoch 47
- Train ACC: final 95.65%, best 95.65% at epoch 70
- Train Loss: final 0.1241, best 0.1241 at epoch 70
- Val Macro F1: final 78.39%, best 79.03% at epoch 62
- Test Macro F1: final 77.54%, best 78.14% at epoch 69
- Selected Test Macro Precision: 77.72%
- Selected Test Macro Recall: 77.78%
- Selected Val Macro Precision: 78.99%
- Selected Val Macro Recall: 78.94%

## Suggested Meeting Conclusion

- Current headline result: ViT Learnable Position leads on Test ACC with 78.03%.
- When presenting baseline vs comparison, use ViT Baseline (No Pos) as reference and report ViT Learnable Position at +6.61 pp on Test ACC.
- Per-class pages are now included when selected-checkpoint classwise metrics exist, so class imbalance or hard classes can be discussed directly in meetings.
- The deck is now organized for weekly reporting: setup, overview, key metrics, curves, error analysis, and a conclusion slide instead of raw log-style export.