# Weekly Comparison: CNN Baseline vs ViT Dropout 0.1

Generated: 2026-05-27T22:33:57
Reference run: `CNN Baseline`

## Headline Takeaways

- Primary comparison metric: Test ACC. Best final result is CNN Baseline with 94.92%.
- Largest train-to-test accuracy gap appears in CNN Baseline (4.58 pp), worth calling out during generalization discussion.

## Runs

- `CNN Baseline` (`cnn_resnet18_baseline`), model: `resnet18`, epochs completed: `20`, device: `cuda`
- `ViT Dropout 0.1` (`vit_dropout_01`), model: `vit`, epochs completed: `20`, device: `cuda`

## Key Metrics

### CNN Baseline
- Test ACC: final 94.92%, best 95.30% at epoch 18
- Test Loss: final 0.2116, best 0.1588 at epoch 4
- Train ACC: final 99.50%, best 99.50% at epoch 20
- Train Loss: final 0.0145, best 0.0145 at epoch 20

### ViT Dropout 0.1
- Test ACC: final 73.40%, best 73.40% at epoch 20
- Test Loss: final 0.7734, best 0.7682 at epoch 19
- Train ACC: final 71.49%, best 71.49% at epoch 20
- Train Loss: final 0.7968, best 0.7968 at epoch 20

## Suggested Meeting Conclusion

- Current headline result: CNN Baseline leads on Test ACC with 94.92%.
- When presenting baseline vs comparison, use CNN Baseline as reference and report ViT Dropout 0.1 at -21.52 pp on Test ACC.
- The deck is now organized for weekly reporting: setup, overview, key metrics, curves, error analysis, and a conclusion slide instead of raw log-style export.