# CIFAR-10 Low-Data vs Full-Data Comparison

Selected test accuracy, seed 42. Checkpoints are selected by validation accuracy.

| Training size | Learnable PE | Normal-col multiplicative PE | Row/column latent fusion |
| --- | ---: | ---: | ---: |
| 1k | 36.10% | 40.62% | 40.92% |
| 5k | 54.16% | 56.55% | 54.69% |
| 10k | 63.06% | 62.74% | 63.01% |
| Full | 78.88% | 77.91% | 75.48% |
