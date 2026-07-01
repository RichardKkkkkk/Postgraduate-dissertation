# CADB Multi-label Positional Encoding Comparison

Generated: 2026-06-30T20:38:43
Comparison scenario: `generic_multi_run`
Reference run: `ViT No Pos`

## Headline Takeaways

- Primary comparison metric: Test ACC. Best final result is ViT No Pos with 79.48%.
- Largest train-to-test accuracy gap appears in ViT Column-wise (22.49 pp), worth calling out during generalization discussion.
- Early stopping summary: ViT No Pos: stopped early; ViT Baseline: stopped early; ViT Row-wise: stopped early

## Runs

- `ViT No Pos` (`no_pos_cadb_elements_seed42`), model: `ViT No Pos`, family: `vit`, variant: `no_pos`, position encoding: `none`, epochs completed: `70`, device: `cuda`
- `ViT Baseline` (`baseline_cadb_elements_seed42`), model: `ViT Baseline`, family: `vit`, variant: `baseline`, position encoding: `absolute`, epochs completed: `43`, device: `cuda`
- `ViT Row-wise` (`row_cadb_elements_seed42`), model: `ViT Row-wise Sinusoidal`, family: `vit`, variant: `row_sinusoidal`, position encoding: `row_sinusoidal`, epochs completed: `54`, device: `cuda`
- `ViT Column-wise` (`col_cadb_elements_seed42`), model: `ViT Column-wise Sinusoidal`, family: `vit`, variant: `col_sinusoidal`, position encoding: `col_sinusoidal`, epochs completed: `50`, device: `cuda`

## Key Metrics

### ViT No Pos
- Test ACC: final 79.48%, best 82.31% at epoch 20
- Val ACC: final 79.50%, best 82.39% at epoch 21
- Test Loss: final 0.9451, best 0.3883 at epoch 18
- Val Loss: final 0.9655, best 0.3841 at epoch 18
- Train ACC: final 99.96%, best 99.97% at epoch 65
- Train Loss: final 0.0072, best 0.0072 at epoch 70
- Val Macro F1: final 33.93%, best 35.92% at epoch 55
- Test Macro F1: final 33.50%, best 35.07% at epoch 46
- Selected Test Macro Precision: 36.25%
- Selected Test Macro Recall: 32.74%
- Selected Val Macro Precision: 38.34%
- Selected Val Macro Recall: 34.64%
- Selected Test Subset Accuracy: 30.98%
- Selected Val Subset Accuracy: 30.80%

### ViT Baseline
- Test ACC: final 78.97%, best 81.93% at epoch 16
- Val ACC: final 78.07%, best 80.99% at epoch 12
- Test Loss: final 0.9462, best 0.4079 at epoch 13
- Val Loss: final 0.9890, best 0.4005 at epoch 9
- Train ACC: final 99.94%, best 99.94% at epoch 41
- Train Loss: final 0.0095, best 0.0095 at epoch 43
- Val Macro F1: final 32.17%, best 33.91% at epoch 28
- Test Macro F1: final 33.43%, best 35.67% at epoch 24
- Selected Test Macro Precision: 34.08%
- Selected Test Macro Recall: 34.80%
- Selected Val Macro Precision: 34.43%
- Selected Val Macro Recall: 34.51%
- Selected Test Subset Accuracy: 27.15%
- Selected Val Subset Accuracy: 27.49%

### ViT Row-wise
- Test ACC: final 78.62%, best 80.85% at epoch 21
- Val ACC: final 80.15%, best 81.61% at epoch 27
- Test Loss: final 0.9471, best 0.4096 at epoch 16
- Val Loss: final 0.8844, best 0.3916 at epoch 16
- Train ACC: final 99.96%, best 99.97% at epoch 52
- Train Loss: final 0.0086, best 0.0086 at epoch 54
- Val Macro F1: final 36.55%, best 37.44% at epoch 39
- Test Macro F1: final 32.08%, best 34.06% at epoch 30
- Selected Test Macro Precision: 34.92%
- Selected Test Macro Recall: 31.82%
- Selected Val Macro Precision: 39.89%
- Selected Val Macro Recall: 36.45%
- Selected Test Subset Accuracy: 28.30%
- Selected Val Subset Accuracy: 32.55%

### ViT Column-wise
- Test ACC: final 77.41%, best 81.10% at epoch 22
- Val ACC: final 79.66%, best 81.51% at epoch 22
- Test Loss: final 0.9386, best 0.4090 at epoch 20
- Val Loss: final 0.8815, best 0.3893 at epoch 19
- Train ACC: final 99.90%, best 99.90% at epoch 49
- Train Loss: final 0.0130, best 0.0130 at epoch 50
- Val Macro F1: final 35.01%, best 37.11% at epoch 45
- Test Macro F1: final 32.61%, best 35.46% at epoch 41
- Selected Test Macro Precision: 36.84%
- Selected Test Macro Recall: 34.18%
- Selected Val Macro Precision: 39.48%
- Selected Val Macro Recall: 36.45%
- Selected Test Subset Accuracy: 27.92%
- Selected Val Subset Accuracy: 30.60%

## Suggested Meeting Conclusion

- Current headline result: ViT No Pos leads on Test ACC with 79.48%.
- When presenting baseline vs comparison, use ViT No Pos as reference and report ViT Baseline at -0.51 pp on Test ACC.
- Per-class pages are now included when selected-checkpoint classwise metrics exist, so class imbalance or hard classes can be discussed directly in meetings.
- The deck is now organized for weekly reporting: setup, overview, key metrics, curves, error analysis, and a conclusion slide instead of raw log-style export.