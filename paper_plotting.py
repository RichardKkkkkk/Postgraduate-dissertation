import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("results/matplotlib_cache")))

import matplotlib.pyplot as plt


PAPER_STYLE_VERSION = "2026-07-29-single-metric-v2"
PAPER_FIGSIZE = (7.2, 4.5)
PAPER_BAR_FIGSIZE = (7.2, 4.2)
PAPER_HEATMAP_FIGSIZE = (6.2, 5.4)
PAPER_DPI = 300
PAPER_TEXT_COLOR = "#1f2937"
PAPER_GRID_COLOR = "#e5e7eb"

SPLIT_STYLES = {
    "train": {"color": "#0072B2", "linestyle": "-", "marker": "o"},
    "val": {"color": "#D55E00", "linestyle": "--", "marker": "s"},
    "test": {"color": "#009E73", "linestyle": ":", "marker": "^"},
}

MODEL_LABELS = {
    "vit_baseline": "No PE",
    "vit_learnable_position": "Learnable PE",
    "vit_row_sinusoidal": "Row-wise PE",
    "vit_col_sinusoidal": "Column-wise PE",
    "vit_additive_sinusoidal": "Additive PE",
    "vit_additive_sinusoidal_shifted": "Shifted Additive PE",
    "vit_multiplicative_sinusoidal": "Multiplicative PE",
    "vit_multiplicative_sinusoidal_shifted": "Shifted Multiplicative PE",
    "vit_squared_multiplicative_sinusoidal": "Squared Multiplicative PE",
    "vit_squared_multiplicative_sinusoidal_shifted": "Shifted Squared Multiplicative PE",
    "vit_radial_sinusoidal": "Radial PE",
    "vit_normal_col_learnable_multiplicative_sinusoidal": "Learnable + Multiplicative PE",
    "vit_row_col_latent_fusion": "Concat + MLP Fusion",
    "vit_row_col_mean_fusion": "Mean Fusion",
    "vit_row_col_mean_mlp_fusion": "Mean + MLP Fusion",
    "vit_row_col_cross_attention_fusion": "Bidirectional Cross-Attention",
    "vit_row_col_cross_attention_mlp_head_fusion": "Cross-Attention + MLP Head",
    "vit_rope": "RoPE",
    "vit_rope_2d": "2D RoPE",
    "resnet18_scratch": "ResNet18 (Scratch)",
    "resnet18_imagenet": "ResNet18 (ImageNet)",
}

MODEL_COLORS = {
    "vit_baseline": "#4b5563",
    "vit_learnable_position": "#0072B2",
    "vit_row_sinusoidal": "#E69F00",
    "vit_col_sinusoidal": "#009E73",
    "vit_additive_sinusoidal": "#CC79A7",
    "vit_additive_sinusoidal_shifted": "#AA4499",
    "vit_multiplicative_sinusoidal": "#D55E00",
    "vit_multiplicative_sinusoidal_shifted": "#56B4E9",
    "vit_squared_multiplicative_sinusoidal": "#A6761D",
    "vit_squared_multiplicative_sinusoidal_shifted": "#8C510A",
    "vit_radial_sinusoidal": "#882255",
    "vit_normal_col_learnable_multiplicative_sinusoidal": "#44AA99",
    "vit_row_col_latent_fusion": "#7A3DB8",
    "vit_row_col_mean_fusion": "#117733",
    "vit_row_col_mean_mlp_fusion": "#B8860B",
    "vit_row_col_cross_attention_fusion": "#332288",
    "vit_row_col_cross_attention_mlp_head_fusion": "#CC6677",
    "vit_rope": "#88CCEE",
    "vit_rope_2d": "#999933",
    "resnet18_scratch": "#78716c",
    "resnet18_imagenet": "#111827",
}

FALLBACK_COLORS = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#56B4E9",
    "#E69F00",
    "#882255",
    "#4b5563",
)

LINE_STYLES = ("-", "--", "-.", ":")
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "<")
MODEL_MARKERS = {
    "vit_baseline": "o",
    "vit_learnable_position": "s",
    "vit_row_sinusoidal": "^",
    "vit_col_sinusoidal": "D",
    "vit_additive_sinusoidal": "v",
    "vit_additive_sinusoidal_shifted": "P",
    "vit_multiplicative_sinusoidal": "X",
    "vit_multiplicative_sinusoidal_shifted": "<",
    "vit_squared_multiplicative_sinusoidal": ">",
    "vit_squared_multiplicative_sinusoidal_shifted": "h",
    "vit_radial_sinusoidal": "*",
    "vit_row_col_latent_fusion": "o",
    "vit_row_col_mean_fusion": "s",
    "vit_row_col_mean_mlp_fusion": "^",
    "vit_row_col_cross_attention_fusion": "D",
    "vit_row_col_cross_attention_mlp_head_fusion": "v",
}
UNFOLDING_STYLES = {
    "vit_normal_col_": ("Normal Column", "--"),
    "vit_proper_row_": ("Proper Row", "-."),
    "vit_proper_col_": ("Proper Column", ":"),
}


def setup_paper_plot_style():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.8,
            "axes.edgecolor": "#cbd5e1",
            "axes.labelcolor": PAPER_TEXT_COLOR,
            "axes.titlecolor": PAPER_TEXT_COLOR,
            "lines.linewidth": 2.0,
            "xtick.color": PAPER_TEXT_COLOR,
            "ytick.color": PAPER_TEXT_COLOR,
            "savefig.dpi": PAPER_DPI,
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def get_model_label(model_name: str):
    if model_name in MODEL_LABELS:
        return MODEL_LABELS[model_name]
    for prefix, (unfolding_label, _) in UNFOLDING_STYLES.items():
        if model_name.startswith(prefix):
            base_model_name = f"vit_{model_name.removeprefix(prefix)}"
            base_label = MODEL_LABELS.get(
                base_model_name,
                base_model_name.removeprefix("vit_").replace("_", " ").title(),
            )
            return f"{unfolding_label} + {base_label}"
    return model_name.replace("_", " ").title()


def get_model_style(model_name: str, index: int):
    color = MODEL_COLORS.get(model_name)
    linestyle = LINE_STYLES[(index // len(FALLBACK_COLORS)) % len(LINE_STYLES)]
    marker = MODEL_MARKERS.get(model_name, MARKERS[index % len(MARKERS)])
    for prefix, (_, unfolding_linestyle) in UNFOLDING_STYLES.items():
        if model_name.startswith(prefix):
            base_model_name = f"vit_{model_name.removeprefix(prefix)}"
            color = MODEL_COLORS.get(base_model_name, color)
            marker = MODEL_MARKERS.get(base_model_name, marker)
            linestyle = unfolding_linestyle
            break
    return {
        "color": color or FALLBACK_COLORS[index % len(FALLBACK_COLORS)],
        "linestyle": linestyle,
        "marker": marker,
    }


def mark_every(point_count: int):
    return max(1, point_count // 12)


def is_percentage_metric(metric_name: str):
    lowered = metric_name.lower()
    return any(token in lowered for token in ("acc", "accuracy", "precision", "recall", "f1"))


def scale_metric_values(metric_name: str, values):
    scale = 100.0 if is_percentage_metric(metric_name) else 1.0
    return [float(value) * scale for value in values]


def finish_epoch_axis(axis, metric_name: str, title: str, show_legend: bool = True):
    axis.set_xlabel("Epoch")
    if is_percentage_metric(metric_name):
        ylabel = "Accuracy (%)" if "acc" in metric_name.lower() else "Score (%)"
    elif "loss" in metric_name.lower():
        ylabel = "Loss"
    else:
        ylabel = "Value"
    axis.set_ylabel(ylabel)
    axis.set_title(title, pad=10)
    axis.grid(True, axis="y", linestyle="--", linewidth=0.7, color=PAPER_GRID_COLOR, alpha=0.9)
    axis.grid(False, axis="x")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    if is_percentage_metric(metric_name):
        axis.set_ylim(0.0, 100.0)
    else:
        axis.set_ylim(bottom=0.0)
    if show_legend:
        axis.legend(loc="best", frameon=False)


def place_comparison_legend(axis, series_count: int):
    columns = 1 if series_count == 1 else min(3, series_count)
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.19),
        ncol=columns,
        frameon=False,
        columnspacing=1.1,
        handlelength=2.6,
    )


def finish_bar_axis(axis, title: str, ylabel: str = "Percentage (%)", y_max: float = 100.0):
    axis.set_ylabel(ylabel)
    axis.set_title(title, pad=10)
    axis.set_ylim(0, y_max)
    axis.grid(True, axis="y", linestyle="--", linewidth=0.7, color=PAPER_GRID_COLOR, alpha=0.9)
    axis.grid(False, axis="x")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def annotate_bars(axis, bars, values, suffix: str = "", decimals: int = 1):
    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.8,
            f"{value:.{decimals}f}{suffix}",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color=PAPER_TEXT_COLOR,
        )


def save_figure_pair(figure, png_path: Path):
    png_path = Path(png_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = png_path.with_suffix(".pdf")
    figure.tight_layout()
    figure.savefig(png_path, dpi=PAPER_DPI, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    return {"png": png_path, "pdf": pdf_path}
