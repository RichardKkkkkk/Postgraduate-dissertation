# Seed Sweep Summary

- Seeds: 42, 43, 44, 45, 46
- Models: vit_no_pos, vit_baseline, vit_row_sinusoidal, vit_col_sinusoidal, vit_additive_sinusoidal, vit_additive_sinusoidal_shifted, vit_multiplicative_sinusoidal, vit_multiplicative_sinusoidal_shifted
- Reference model: vit_baseline

- Epoch curve metrics: train_loss, val_loss, test_loss, train_acc, val_acc, test_acc, val_macro_f1, test_macro_f1

## Headline Insights

- ViT Baseline has the highest mean test accuracy: 0.7885 +- 0.0041.
- vit_no_pos changes mean test accuracy by -0.0746 versus ViT Baseline.
- vit_row_sinusoidal changes mean test accuracy by -0.0394 versus ViT Baseline.
- vit_col_sinusoidal changes mean test accuracy by -0.0428 versus ViT Baseline.
- vit_additive_sinusoidal changes mean test accuracy by -0.0230 versus ViT Baseline.
- vit_additive_sinusoidal_shifted changes mean test accuracy by -0.0203 versus ViT Baseline.
- vit_multiplicative_sinusoidal changes mean test accuracy by -0.0140 versus ViT Baseline.
- vit_multiplicative_sinusoidal_shifted changes mean test accuracy by -0.0122 versus ViT Baseline.
- ViT Baseline wins 5/5 seeds on test accuracy.

## Aggregate Table

| Model | best_val_acc mean +- std | test_acc mean +- std | macro_f1 mean +- std |
|---|---|---|---|
| vit_no_pos | 0.7218 +- 0.0106 | 0.7139 +- 0.0057 | 0.7135 +- 0.0062 |
| ViT Baseline | 0.7965 +- 0.0072 | 0.7885 +- 0.0041 | 0.7883 +- 0.0037 |
| vit_row_sinusoidal | 0.7559 +- 0.0037 | 0.7492 +- 0.0018 | 0.7488 +- 0.0023 |
| vit_col_sinusoidal | 0.7534 +- 0.0055 | 0.7457 +- 0.0062 | 0.7453 +- 0.0064 |
| vit_additive_sinusoidal | 0.7718 +- 0.0072 | 0.7655 +- 0.0046 | 0.7650 +- 0.0052 |
| vit_additive_sinusoidal_shifted | 0.7774 +- 0.0074 | 0.7682 +- 0.0035 | 0.7680 +- 0.0038 |
| vit_multiplicative_sinusoidal | 0.7845 +- 0.0069 | 0.7746 +- 0.0049 | 0.7735 +- 0.0061 |
| vit_multiplicative_sinusoidal_shifted | 0.7852 +- 0.0113 | 0.7764 +- 0.0039 | 0.7765 +- 0.0035 |

## Delta Vs Reference

| Model | best_val_acc delta | test_acc delta | macro_f1 delta |
|---|---|---|---|
| vit_no_pos | -0.0747 | -0.0746 | -0.0748 |
| ViT Baseline | +0.0000 | +0.0000 | +0.0000 |
| vit_row_sinusoidal | -0.0406 | -0.0394 | -0.0395 |
| vit_col_sinusoidal | -0.0431 | -0.0428 | -0.0430 |
| vit_additive_sinusoidal | -0.0247 | -0.0230 | -0.0233 |
| vit_additive_sinusoidal_shifted | -0.0192 | -0.0203 | -0.0203 |
| vit_multiplicative_sinusoidal | -0.0120 | -0.0140 | -0.0148 |
| vit_multiplicative_sinusoidal_shifted | -0.0113 | -0.0122 | -0.0118 |
