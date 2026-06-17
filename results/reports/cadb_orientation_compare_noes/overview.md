# Experiment Comparison: ViT Baseline vs ViT Row-wise vs ViT Column-wise

Generated: 2026-06-15T13:54:12
Comparison scenario: `generic_multi_run`
Reference run: `ViT Baseline`

## Headline Takeaways

- Primary comparison metric: Test ACC. Best final result is ViT Row-wise with 76.23%.
- Against reference run ViT Baseline, ViT Row-wise changes Test ACC by +6.56 pp.
- Largest train-to-test accuracy gap appears in ViT Baseline (22.72 pp), worth calling out during generalization discussion.

## Runs

- `ViT Baseline` (`baseline_cadb_seed42_noes`), model: `ViT Baseline`, family: `vit`, variant: `baseline`, position encoding: `absolute`, epochs completed: `20`, device: `cuda`
- `ViT Row-wise` (`row_cadb_seed42_noes`), model: `ViT Row-wise Sinusoidal`, family: `vit`, variant: `row_sinusoidal`, position encoding: `row_sinusoidal`, epochs completed: `20`, device: `cuda`
- `ViT Column-wise` (`col_cadb_seed42_noes`), model: `ViT Column-wise Sinusoidal`, family: `vit`, variant: `col_sinusoidal`, position encoding: `col_sinusoidal`, epochs completed: `20`, device: `cuda`

## Key Metrics

### ViT Baseline
- Test ACC: final 69.67%, best 75.82% at epoch 13
- Val ACC: final 75.09%, best 79.42% at epoch 11
- Test Loss: final 1.0030, best 0.5170 at epoch 10
- Val Loss: final 0.8395, best 0.4541 at epoch 8
- Train ACC: final 92.40%, best 92.40% at epoch 20
- Train Loss: final 0.1752, best 0.1752 at epoch 20
- Selected Test Macro F1: 70.96%
- Selected Val Macro F1: 74.86%
- Selected Test Macro Precision: 74.35%
- Selected Test Macro Recall: 70.04%
- Selected Val Macro Precision: 74.96%
- Selected Val Macro Recall: 74.77%

### ViT Row-wise
- Test ACC: final 76.23%, best 76.64% at epoch 15
- Val ACC: final 76.53%, best 81.23% at epoch 11
- Test Loss: final 0.5436, best 0.4943 at epoch 15
- Val Loss: final 0.4771, best 0.4242 at epoch 15
- Train ACC: final 85.27%, best 85.51% at epoch 19
- Train Loss: final 0.3179, best 0.3179 at epoch 20
- Selected Test Macro F1: 64.99%
- Selected Val Macro F1: 73.58%
- Selected Test Macro Precision: 71.96%
- Selected Test Macro Recall: 64.67%
- Selected Val Macro Precision: 80.41%
- Selected Val Macro Recall: 71.21%

### ViT Column-wise
- Test ACC: final 75.00%, best 77.46% at epoch 16
- Val ACC: final 80.51%, best 81.95% at epoch 15
- Test Loss: final 0.5457, best 0.4832 at epoch 16
- Val Loss: final 0.4516, best 0.3997 at epoch 16
- Train ACC: final 83.75%, best 83.75% at epoch 20
- Train Loss: final 0.3337, best 0.3337 at epoch 20
- Selected Test Macro F1: 70.99%
- Selected Val Macro F1: 76.33%
- Selected Test Macro Precision: 73.36%
- Selected Test Macro Recall: 70.17%
- Selected Val Macro Precision: 79.19%
- Selected Val Macro Recall: 74.69%

## Suggested Meeting Conclusion

- Current headline result: ViT Row-wise leads on Test ACC with 76.23%.
- When presenting baseline vs comparison, use ViT Baseline as reference and report ViT Row-wise at +6.56 pp on Test ACC.
- Per-class pages are now included when selected-checkpoint classwise metrics exist, so class imbalance or hard classes can be discussed directly in meetings.
- The deck is now organized for weekly reporting: setup, overview, key metrics, curves, error analysis, and a conclusion slide instead of raw log-style export.