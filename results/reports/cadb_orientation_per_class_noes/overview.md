# Per-Class Comparison Overview

## Overall Metrics

- ViT Baseline: test_acc=0.7500, macro_f1=0.7096
- ViT Row-wise: test_acc=0.7172, macro_f1=0.6499
- ViT Column-wise: test_acc=0.7459, macro_f1=0.7099

## Headline Takeaways

- Best overall run in this comparison: ViT Baseline (test_acc=0.7500, macro_f1=0.7096).
- Reference run for class-level deltas: ViT Baseline.

## Per-Class Accuracy Changes

### ViT Row-wise vs ViT Baseline

- Top improved classes: horizontal (+0.0260), vertical (-0.1333)
- Largest drops / weakest changes: vertical (-0.1333), horizontal (+0.0260)

### ViT Column-wise vs ViT Baseline

- Top improved classes: vertical (+0.0222), horizontal (-0.0195)
- Largest drops / weakest changes: horizontal (-0.0195), vertical (+0.0222)

## Weakest Classes In The Best Run

- Lowest per-class accuracy in ViT Baseline: vertical (0.5111), horizontal (0.8896)

## Suggested PPT Narrative

- Use this page to explain whether performance differences come from a specific class bias rather than only the overall accuracy.
- For directional experiments, focus on whether one model is stronger on the horizontal class while another is stronger on the vertical class.
- If one model collapses to a majority class, the confusion matrix and per-class F1 should make that visible immediately.
