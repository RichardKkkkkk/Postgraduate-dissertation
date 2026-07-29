import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("results/matplotlib_cache")))

import matplotlib.pyplot as plt


PAPER_FIGSIZE = (7.2, 4.5)
PAPER_DPI = 300

SPLIT_STYLES = {
    "train": {"color": "#2563eb", "linestyle": "-", "marker": "o"},
    "val": {"color": "#ea580c", "linestyle": "--", "marker": "s"},
    "test": {"color": "#16a34a", "linestyle": ":", "marker": "^"},
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
    "vit_learnable_position": "#2563eb",
    "vit_row_sinusoidal": "#ea580c",
    "vit_col_sinusoidal": "#16a34a",
    "vit_additive_sinusoidal": "#7c3aed",
    "vit_additive_sinusoidal_shifted": "#a16207",
    "vit_multiplicative_sinusoidal": "#dc2626",
    "vit_multiplicative_sinusoidal_shifted": "#0f766e",
    "vit_squared_multiplicative_sinusoidal": "#c2410c",
    "vit_squared_multiplicative_sinusoidal_shifted": "#854d0e",
    "vit_radial_sinusoidal": "#db2777",
    "vit_normal_col_learnable_multiplicative_sinusoidal": "#0891b2",
    "vit_row_col_latent_fusion": "#9333ea",
    "vit_row_col_mean_fusion": "#65a30d",
    "vit_row_col_mean_mlp_fusion": "#ca8a04",
    "vit_row_col_cross_attention_fusion": "#0284c7",
    "vit_row_col_cross_attention_mlp_head_fusion": "#be123c",
    "vit_rope": "#0d9488",
    "vit_rope_2d": "#6366f1",
    "resnet18_scratch": "#78716c",
    "resnet18_imagenet": "#111827",
}

FALLBACK_COLORS = (
    "#2563eb",
    "#ea580c",
    "#16a34a",
    "#dc2626",
    "#7c3aed",
    "#0f766e",
    "#db2777",
    "#4b5563",
)

LINE_STYLES = ("-", "--", "-.", ":")
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "<")
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
            "lines.linewidth": 2.0,
            "savefig.dpi": PAPER_DPI,
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
    for prefix, (_, unfolding_linestyle) in UNFOLDING_STYLES.items():
        if model_name.startswith(prefix):
            base_model_name = f"vit_{model_name.removeprefix(prefix)}"
            color = MODEL_COLORS.get(base_model_name, color)
            linestyle = unfolding_linestyle
            break
    return {
        "color": color or FALLBACK_COLORS[index % len(FALLBACK_COLORS)],
        "linestyle": linestyle,
        "marker": MARKERS[index % len(MARKERS)],
    }


def mark_every(point_count: int):
    return max(1, point_count // 10)


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
    axis.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
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
    columns = 1 if series_count == 1 else 2
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=columns,
        frameon=False,
        columnspacing=1.2,
        handlelength=2.6,
    )


def save_figure_pair(figure, png_path: Path):
    png_path = Path(png_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = png_path.with_suffix(".pdf")
    figure.tight_layout()
    figure.savefig(png_path, dpi=PAPER_DPI, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    return {"png": png_path, "pdf": pdf_path}
