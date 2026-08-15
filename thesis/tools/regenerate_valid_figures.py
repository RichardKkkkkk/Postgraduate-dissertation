from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paper_plotting import (
    MODEL_COLORS,
    MODEL_LABELS,
    MODEL_LINESTYLES,
    PAPER_GRID_COLOR,
    PAPER_TEXT_COLOR,
    setup_paper_plot_style,
)


ROOT = Path(r"D:\code\Postgraduate-dissertation")
REPORT = ROOT / "results" / "reports" / "thesis_final_evidence_figures_v1"
OUT = ROOT / "thesis" / "assets"

LOW_DATA_MODELS = (
    "vit_baseline",
    "vit_learnable_position",
    "vit_multiplicative_sinusoidal_shifted",
)

FUSION_LEFT = (
    "vit_learnable_position",
    "vit_row_col_mean_fusion",
    "vit_row_col_mean_mlp_fusion",
    "vit_row_col_latent_fusion",
)

FUSION_RIGHT = (
    "vit_learnable_position",
    "vit_row_col_cross_attention_fusion",
    "vit_row_col_cross_attention_mlp_head_fusion",
)


def read_rows(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def group_curves(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["condition"], row["series"])].append(row)
    for key in grouped:
        grouped[key].sort(key=lambda item: int(item["epoch"]))
    return grouped


def plot_curve(axis, rows, model):
    epochs = [int(row["epoch"]) for row in rows]
    means = [float(row["mean"]) for row in rows]
    lower = [float(row["ci95_lower"]) for row in rows]
    upper = [float(row["ci95_upper"]) for row in rows]
    color = MODEL_COLORS[model]
    axis.plot(
        epochs,
        means,
        color=color,
        linestyle=MODEL_LINESTYLES.get(model, "-"),
        linewidth=2.1,
        label=MODEL_LABELS[model],
    )
    axis.fill_between(epochs, lower, upper, color=color, alpha=0.16, linewidth=0)
    return min(lower), max(upper)


def clean_axis(axis, title):
    axis.set_title(title, pad=10)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Validation accuracy (%)")
    axis.grid(True, axis="y", linestyle="--", linewidth=0.7, color=PAPER_GRID_COLOR)
    axis.grid(False, axis="x")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def save(figure, name):
    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / f"{name}.png"
    pdf = OUT / f"{name}.pdf"
    figure.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf, bbox_inches="tight", facecolor="white")
    print(png)
    print(pdf)


def make_low_data():
    rows = read_rows(REPORT / "low_data_validation_epoch_summary.csv")
    curves = group_curves(rows)
    setup_paper_plot_style()
    figure, axes = plt.subplots(2, 2, figsize=(12.2, 8.0), sharey=True)
    sizes = (("1000", "1k training examples"), ("5000", "5k training examples"),
             ("10000", "10k training examples"), ("45000", "Full (45k) training examples"))
    lows, highs = [], []
    for axis, (size, title) in zip(axes.flat, sizes):
        for model in LOW_DATA_MODELS:
            low, high = plot_curve(axis, curves[(size, model)], model)
            lows.append(low)
            highs.append(high)
        clean_axis(axis, title)
    span = max(highs) - min(lows)
    for axis in axes.flat:
        axis.set_ylim(max(0, min(lows) - 0.08 * span), max(highs) + 0.08 * span)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.005))
    figure.suptitle("CIFAR-10 training-set size: validation trajectories", y=1.01, color=PAPER_TEXT_COLOR)
    figure.subplots_adjust(bottom=0.13, hspace=0.28, wspace=0.16)
    save(figure, "low_data_validation_accuracy_epoch_valid")
    plt.close(figure)


def make_fusion():
    rows = read_rows(REPORT / "fusion_validation_epoch_summary.csv")
    curves = group_curves(rows)
    setup_paper_plot_style()
    figure, axes = plt.subplots(1, 2, figsize=(12.2, 5.0), sharey=True)
    for axis, models, title in (
        (axes[0], FUSION_LEFT, "(a) Aggregation-based fusion"),
        (axes[1], FUSION_RIGHT, "(b) Cross-attention fusion"),
    ):
        for model in models:
            plot_curve(axis, curves[("fusion", model)], model)
        clean_axis(axis, title)
        axis.legend(loc="lower right", frameon=False)
    figure.suptitle("Single-branch reference and dual-branch fusion: validation trajectories", y=1.01, color=PAPER_TEXT_COLOR)
    figure.subplots_adjust(bottom=0.12, wspace=0.12)
    save(figure, "fusion_validation_accuracy_epoch_valid")
    plt.close(figure)


if __name__ == "__main__":
    make_low_data()
    make_fusion()
