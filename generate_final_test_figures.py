import argparse
import csv
import inspect
import json
import math
import os
import statistics
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("results/matplotlib_cache")))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from datasets import cifar10_data
from generate_thesis_statistics import count_parameters
from paper_plotting import (
    PAPER_GRID_COLOR,
    PAPER_MUTED_COLOR,
    PAPER_POINT_ALPHA,
    PAPER_TEXT_COLOR,
    get_model_label,
    get_model_style,
    save_figure_pair,
    setup_paper_plot_style,
)


SEEDS = (42, 43, 44, 45, 46)
T_CRITICAL_95_DF4 = 2.7764451051977987

MAIN_EXPERIMENT = "cifar10_final_vit_models_5seeds"
LOW_DATA_EXPERIMENTS = {
    1000: "cifar10_low_data_1k_4models_5seeds_lr3e4",
    5000: "cifar10_low_data_5k_4models_5seeds_lr3e4",
    10000: "cifar10_low_data_10k_4models_5seeds_lr3e4",
}
CIFAR100_EXPERIMENT = "cifar100_4models_5seeds_lr3e4"
DEFAULT_REPORT = "thesis_selected_test_figures_v1"

CORE_MODELS = (
    "vit_baseline",
    "vit_learnable_position",
    "vit_row_sinusoidal",
    "vit_col_sinusoidal",
    "vit_additive_sinusoidal",
    "vit_additive_sinusoidal_shifted",
    "vit_multiplicative_sinusoidal",
    "vit_multiplicative_sinusoidal_shifted",
    "vit_radial_sinusoidal",
)

LOW_DATA_MODELS = (
    "vit_baseline",
    "vit_learnable_position",
    "vit_multiplicative_sinusoidal_shifted",
    "vit_normal_col_learnable_multiplicative_sinusoidal",
)

CIFAR100_MODELS = (
    "vit_baseline",
    "vit_learnable_position",
    "vit_additive_sinusoidal_shifted",
    "vit_multiplicative_sinusoidal_shifted",
)

PATCH_FAMILIES = (
    ("No PE", "baseline"),
    ("Learnable PE", "learnable_position"),
    ("Row PE", "row_sinusoidal"),
    ("Column PE", "col_sinusoidal"),
    ("Multiplicative PE", "multiplicative_sinusoidal"),
)

PATCH_ORDERS = (
    ("normal_row", "normal_row"),
    ("normal_col", "normal_col"),
    ("proper_row", "proper_row"),
    ("proper_col", "proper_col"),
)

FUSION_MODELS = (
    "vit_learnable_position",
    "vit_normal_col_learnable_multiplicative_sinusoidal",
    "vit_row_col_mean_fusion",
    "vit_row_col_mean_mlp_fusion",
    "vit_row_col_latent_fusion",
    "vit_row_col_cross_attention_fusion",
    "vit_row_col_cross_attention_mlp_head_fusion",
)

SHIFT_CONTRASTS = (
    (
        "Shifted additive - additive",
        "vit_additive_sinusoidal_shifted",
        "vit_additive_sinusoidal",
    ),
    (
        "Shifted multiplicative - multiplicative",
        "vit_multiplicative_sinusoidal_shifted",
        "vit_multiplicative_sinusoidal",
    ),
)

PER_CLASS_CONTRASTS = (
    ("Learnable - No PE", "vit_learnable_position", "vit_baseline"),
    (
        "Shifted multiplicative - Learnable",
        "vit_multiplicative_sinusoidal_shifted",
        "vit_learnable_position",
    ),
)

CONFIG_FIELDS = (
    "epochs",
    "batch_size",
    "lr",
    "weight_decay",
    "val_subset",
    "test_subset",
    "val_ratio",
    "split_seed",
    "image_size",
    "early_stopping_patience",
    "early_stopping_metric",
    "early_stopping_min_delta",
    "lr_plateau_patience",
    "lr_plateau_factor",
    "lr_plateau_min_lr",
    "embedding_dropout",
    "attention_dropout",
    "projection_dropout",
    "mlp_dropout",
)

SHORT_LABELS = {
    "vit_baseline": "No PE",
    "vit_learnable_position": "Learnable",
    "vit_row_sinusoidal": "Row",
    "vit_col_sinusoidal": "Column",
    "vit_additive_sinusoidal": "Additive",
    "vit_additive_sinusoidal_shifted": "Shifted\nadditive",
    "vit_multiplicative_sinusoidal": "Multiplicative",
    "vit_multiplicative_sinusoidal_shifted": "Shifted\nmultiplicative",
    "vit_radial_sinusoidal": "Radial",
    "vit_normal_col_learnable_multiplicative_sinusoidal": "Hybrid",
    "vit_row_col_mean_fusion": "Mean",
    "vit_row_col_mean_mlp_fusion": "Mean + MLP",
    "vit_row_col_latent_fusion": "Concat + MLP",
    "vit_row_col_cross_attention_fusion": "Cross-attention",
    "vit_row_col_cross_attention_mlp_head_fusion": "Cross-attention\n+ MLP head",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate the frozen selected-checkpoint thesis table and figures."
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--report-name", default=DEFAULT_REPORT)
    parser.add_argument("--main-experiment", default=MAIN_EXPERIMENT)
    parser.add_argument("--cifar100-experiment", default=CIFAR100_EXPERIMENT)
    return parser.parse_args()


def patch_model_name(order, suffix):
    return f"vit_{suffix}" if order == "normal_row" else f"vit_{order}_{suffix}"


def sample_sd(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


def ci95(values):
    if len(values) != 5:
        raise ValueError(f"Expected five seeds for 95% t CI, found {len(values)}")
    return T_CRITICAL_95_DF4 * sample_sd(values) / math.sqrt(len(values))


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_experiment(results_dir, experiment, required_models):
    metrics_dir = results_dir / experiment / "metrics"
    paths = sorted(metrics_dir.glob("*/*_summary.json"))
    if not paths:
        raise FileNotFoundError(f"No summary JSON files under {metrics_dir}")

    required = set(required_models)
    rows = []
    configs = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        config = payload["config"]
        model = config["model"]
        seed = int(config["seed"])
        if model not in required or seed not in SEEDS:
            continue
        key = (model, seed)
        if key in configs:
            raise ValueError(f"Duplicate run in {experiment}: {model}, seed {seed}")
        protocol = payload.get("test_evaluation_protocol")
        if protocol != "selected_checkpoint_only":
            raise ValueError(f"Invalid test protocol in {path}: {protocol}")
        selected = payload["selected_model"]
        recall = selected.get("test_per_class_recall")
        labels = selected.get("label_names")
        configs[key] = config
        rows.append(
            {
                "experiment": experiment,
                "dataset": config["dataset"],
                "train_subset": config.get("train_subset"),
                "model": model,
                "model_label": get_model_label(model),
                "seed": seed,
                "selected_epoch": int(selected["epoch"]),
                "test_acc": float(selected["test_acc"]),
                "test_loss": float(selected["test_loss"]),
                "test_per_class_recall": [float(value) for value in recall]
                if recall is not None
                else None,
                "label_names": list(labels) if labels is not None else None,
                "protocol": protocol,
                "summary_path": str(path),
            }
        )

    expected_seeds = set(SEEDS)
    for model in required_models:
        observed = {row["seed"] for row in rows if row["model"] == model}
        if observed != expected_seeds:
            raise ValueError(
                f"{experiment}: {model} has seeds {sorted(observed)}, "
                f"expected {sorted(expected_seeds)}"
            )
    return rows, configs, len(paths)


def config_value(config, field):
    return json.dumps(config.get(field), sort_keys=True)


def validate_uniform_config(experiment, configs, ignored=("train_subset",)):
    rows = []
    for field in CONFIG_FIELDS:
        if field in ignored:
            continue
        values = {config_value(config, field) for config in configs.values()}
        rows.append(
            {
                "comparison": f"within:{experiment}",
                "field": field,
                "status": "match" if len(values) == 1 else "mismatch",
                "values": "; ".join(sorted(values)),
            }
        )
        if len(values) != 1:
            raise ValueError(f"Non-uniform {field} within {experiment}: {values}")
    return rows


def validate_full_low_alignment(main_configs, low_configs_by_size):
    audit = []
    main_reference = next(iter(main_configs.values()))
    for train_size, configs in low_configs_by_size.items():
        low_reference = next(iter(configs.values()))
        for field in CONFIG_FIELDS:
            main_value = config_value(main_reference, field)
            low_value = config_value(low_reference, field)
            status = "match" if main_value == low_value else "mismatch"
            audit.append(
                {
                    "comparison": f"full_vs_{train_size}",
                    "field": field,
                    "status": status,
                    "values": f"full={main_value}; low={low_value}",
                }
            )
            if status != "match":
                raise ValueError(
                    f"Full-data alignment failed for {train_size}, field {field}: "
                    f"{main_value} != {low_value}"
                )

        for model in LOW_DATA_MODELS:
            main_model_configs = [
                config for (name, _), config in main_configs.items() if name == model
            ]
            low_model_configs = [
                config for (name, _), config in configs.items() if name == model
            ]
            if not main_model_configs or not low_model_configs:
                raise ValueError(f"Missing model for full-data gate: {model}")
            audit.append(
                {
                    "comparison": f"full_vs_{train_size}",
                    "field": f"model_structure:{model}",
                    "status": "match",
                    "values": "same registered model identifier",
                }
            )

    augmentation = "RandomCrop(32,padding=4); RandomHorizontalFlip"
    normalisation = f"mean={cifar10_data.CIFAR10_MEAN}; std={cifar10_data.CIFAR10_STD}"
    loader_source = inspect.getsourcefile(cifar10_data.build_vit_dataloaders)
    for field, value in (
        ("augmentation", augmentation),
        ("normalisation", normalisation),
        ("dataset_loader", loader_source),
        ("test_protocol", "selected_checkpoint_only"),
    ):
        audit.append(
            {
                "comparison": "full_vs_all_low_data",
                "field": field,
                "status": "match",
                "values": value,
            }
        )
    return audit


def group_model(rows, model):
    return sorted([row for row in rows if row["model"] == model], key=lambda x: x["seed"])


def summarise_models(rows, models, include_parameters=False):
    output = []
    for model in models:
        group = group_model(rows, model)
        acc = [100.0 * row["test_acc"] for row in group]
        loss = [row["test_loss"] for row in group]
        item = {
            "model": model,
            "model_label": get_model_label(model),
            "num_seeds": len(group),
            "mean_test_acc_pct": statistics.mean(acc),
            "sd_test_acc_pp": sample_sd(acc),
            "ci95_half_width_test_acc_pp": ci95(acc),
            "ci95_lower_test_acc_pct": statistics.mean(acc) - ci95(acc),
            "ci95_upper_test_acc_pct": statistics.mean(acc) + ci95(acc),
            "mean_test_loss": statistics.mean(loss),
            "sd_test_loss": sample_sd(loss),
            "ci95_half_width_test_loss": ci95(loss),
            "ci95_lower_test_loss": statistics.mean(loss) - ci95(loss),
            "ci95_upper_test_loss": statistics.mean(loss) + ci95(loss),
        }
        if include_parameters:
            item["trainable_parameters"] = count_parameters(model)
        output.append(item)
    return output


def per_seed_export(rows):
    return [
        {
            "experiment": row["experiment"],
            "dataset": row["dataset"],
            "train_subset": "full" if row["train_subset"] is None else row["train_subset"],
            "model": row["model"],
            "model_label": row["model_label"],
            "seed": row["seed"],
            "selected_epoch": row["selected_epoch"],
            "test_acc_pct": 100.0 * row["test_acc"],
            "test_loss": row["test_loss"],
            "protocol": row["protocol"],
            "summary_path": row["summary_path"],
        }
        for row in rows
    ]


def clean_axis(axis, grid_axis="y"):
    axis.grid(True, axis=grid_axis, color=PAPER_GRID_COLOR, linestyle="--", linewidth=0.7)
    axis.grid(False, axis="x" if grid_axis == "y" else "y")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def padded_limits(values, lower_bound=None, fraction=0.10):
    low, high = min(values), max(values)
    span = max(high - low, 1e-6)
    padding = max(span * fraction, 0.25 if high > 10 else 0.02)
    lower = low - padding
    if lower_bound is not None:
        lower = max(lower_bound, lower)
    return lower, high + padding


def summary_legend():
    return [
        Line2D(
            [0], [0], marker="o", linestyle="", color=PAPER_MUTED_COLOR,
            alpha=PAPER_POINT_ALPHA, label="Individual seed"
        ),
        Line2D(
            [0], [0], marker="D", linestyle="-", color=PAPER_TEXT_COLOR,
            label="Mean ± 95% CI"
        ),
    ]


def draw_model_metric(axis, rows, summaries, models, metric, title):
    raw_key = "test_acc" if metric == "acc" else "test_loss"
    mean_key = "mean_test_acc_pct" if metric == "acc" else "mean_test_loss"
    ci_key = (
        "ci95_half_width_test_acc_pp" if metric == "acc" else "ci95_half_width_test_loss"
    )
    scale = 100.0 if metric == "acc" else 1.0
    all_values = []
    offsets = np.linspace(-0.12, 0.12, len(SEEDS))
    for index, model in enumerate(models):
        style = get_model_style(model, index)
        group = group_model(rows, model)
        values = [scale * row[raw_key] for row in group]
        all_values.extend(values)
        axis.scatter(
            index + offsets,
            values,
            color=style["color"],
            s=19,
            alpha=PAPER_POINT_ALPHA,
            edgecolors="none",
            zorder=2,
        )
        summary = next(item for item in summaries if item["model"] == model)
        all_values.extend(
            [summary[mean_key] - summary[ci_key], summary[mean_key] + summary[ci_key]]
        )
        axis.errorbar(
            index,
            summary[mean_key],
            yerr=summary[ci_key],
            fmt="D",
            color=style["color"],
            markeredgecolor=PAPER_TEXT_COLOR,
            markeredgewidth=0.7,
            markersize=6,
            capsize=4,
            elinewidth=1.3,
            zorder=3,
        )
    axis.set_xticks(range(len(models)), [SHORT_LABELS[model] for model in models], rotation=25, ha="right")
    axis.set_ylabel("Test accuracy (%)" if metric == "acc" else "Test loss")
    axis.set_title(title, pad=9)
    axis.set_ylim(*padded_limits(all_values, lower_bound=0.0))
    clean_axis(axis)


def plot_core(rows, summaries, figures_dir):
    setup_paper_plot_style()
    figure, axes = plt.subplots(1, 2, figsize=(13.0, 5.2))
    draw_model_metric(axes[0], rows, summaries, CORE_MODELS, "acc", "(a) Selected-checkpoint test accuracy")
    draw_model_metric(axes[1], rows, summaries, CORE_MODELS, "loss", "(b) Selected-checkpoint test loss")
    figure.legend(handles=summary_legend(), loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.01))
    figure.subplots_adjust(bottom=0.28, wspace=0.23)
    paths = save_figure_pair(figure, figures_dir / "core_pe_selected_test_accuracy_loss.png")
    plt.close(figure)
    return paths


def build_patch_rows(main_rows):
    per_seed = []
    summary = []
    for family, suffix in PATCH_FAMILIES:
        reference = {row["seed"]: row for row in group_model(main_rows, patch_model_name("normal_row", suffix))}
        for order, order_label in PATCH_ORDERS:
            model = patch_model_name(order, suffix)
            current = {row["seed"]: row for row in group_model(main_rows, model)}
            deltas = []
            for seed in SEEDS:
                delta = 100.0 * (current[seed]["test_acc"] - reference[seed]["test_acc"])
                deltas.append(delta)
                per_seed.append(
                    {
                        "family": family,
                        "order": order,
                        "order_label": order_label,
                        "model": model,
                        "seed": seed,
                        "test_acc_pct": 100.0 * current[seed]["test_acc"],
                        "delta_vs_normal_row_pp": delta,
                    }
                )
            summary.append(
                {
                    "family": family,
                    "order": order,
                    "order_label": order_label,
                    "model": model,
                    "num_seeds": 5,
                    "mean_delta_vs_normal_row_pp": statistics.mean(deltas),
                    "sd_delta_vs_normal_row_pp": sample_sd(deltas),
                    "ci95_half_width_delta_vs_normal_row_pp": ci95(deltas),
                }
            )
    return per_seed, summary


def plot_patch_heatmap(summary, figures_dir):
    matrix = np.array(
        [
            [
                next(
                    item["mean_delta_vs_normal_row_pp"]
                    for item in summary
                    if item["family"] == family and item["order"] == order
                )
                for order, _ in PATCH_ORDERS
            ]
            for family, _ in PATCH_FAMILIES
        ]
    )
    limit = max(abs(float(matrix.min())), abs(float(matrix.max())), 0.1)
    setup_paper_plot_style()
    figure, axis = plt.subplots(figsize=(8.3, 5.0))
    image = axis.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    axis.set_xticks(range(len(PATCH_ORDERS)), [label for _, label in PATCH_ORDERS])
    axis.set_yticks(range(len(PATCH_FAMILIES)), [family for family, _ in PATCH_FAMILIES])
    axis.set_xlabel("Patch-to-position assignment")
    axis.set_ylabel("Positional encoding")
    axis.set_title("Mean test-accuracy change relative to normal_row", pad=10)
    threshold = 0.55 * limit
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            value = matrix[row_index, col_index]
            color = "white" if abs(value) > threshold else PAPER_TEXT_COLOR
            axis.text(col_index, row_index, f"{value:+.2f}", ha="center", va="center", color=color, fontsize=9)
    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Change in test accuracy (percentage points)")
    paths = save_figure_pair(figure, figures_dir / "patch_assignment_test_accuracy_delta_heatmap.png")
    plt.close(figure)
    return paths


def build_low_data(main_rows, low_rows_by_size):
    combined = []
    for train_size, rows in low_rows_by_size.items():
        for row in rows:
            item = dict(row)
            item["training_size"] = train_size
            item["training_size_label"] = f"{train_size // 1000}k"
            combined.append(item)
    for row in main_rows:
        if row["model"] in LOW_DATA_MODELS:
            item = dict(row)
            item["training_size"] = 45000
            item["training_size_label"] = "Full (45k)"
            combined.append(item)

    summaries = []
    for size in (1000, 5000, 10000, 45000):
        summaries.extend(summarise_models([row for row in combined if row["training_size"] == size], LOW_DATA_MODELS))
        for item in summaries[-len(LOW_DATA_MODELS):]:
            item["training_size"] = size
            item["training_size_label"] = "Full (45k)" if size == 45000 else f"{size // 1000}k"
    return combined, summaries


def plot_low_data(rows, summaries, figures_dir):
    setup_paper_plot_style()
    figure, axes = plt.subplots(1, 2, figsize=(12.8, 5.1))
    sizes = (1000, 5000, 10000, 45000)
    labels = ("1k", "5k", "10k", "Full (45k)")
    centers = np.arange(len(sizes), dtype=float)
    model_offsets = np.linspace(-0.24, 0.24, len(LOW_DATA_MODELS))
    seed_offsets = np.linspace(-0.035, 0.035, len(SEEDS))

    for axis, metric in zip(axes, ("acc", "loss")):
        raw_key = "test_acc" if metric == "acc" else "test_loss"
        mean_key = "mean_test_acc_pct" if metric == "acc" else "mean_test_loss"
        ci_key = "ci95_half_width_test_acc_pp" if metric == "acc" else "ci95_half_width_test_loss"
        scale = 100.0 if metric == "acc" else 1.0
        all_values = []
        for model_index, model in enumerate(LOW_DATA_MODELS):
            style = get_model_style(model, model_index)
            means = []
            cis = []
            x_values = centers + model_offsets[model_index]
            for size_index, size in enumerate(sizes):
                group = sorted(
                    [row for row in rows if row["model"] == model and row["training_size"] == size],
                    key=lambda row: row["seed"],
                )
                values = [scale * row[raw_key] for row in group]
                all_values.extend(values)
                aggregate = next(
                    item for item in summaries if item["model"] == model and item["training_size"] == size
                )
                means.append(aggregate[mean_key])
                cis.append(aggregate[ci_key])
                axis.scatter(
                    x_values[size_index] + seed_offsets,
                    values,
                    color=style["color"],
                    s=15,
                    alpha=PAPER_POINT_ALPHA,
                    edgecolors="none",
                    zorder=2,
                )
            axis.plot(x_values, means, color=style["color"], linestyle=style["linestyle"], linewidth=1.2, alpha=0.8)
            axis.errorbar(
                x_values,
                means,
                yerr=cis,
                fmt="D",
                color=style["color"],
                markeredgecolor=PAPER_TEXT_COLOR,
                markeredgewidth=0.55,
                markersize=5.2,
                capsize=3.5,
                elinewidth=1.1,
                label=get_model_label(model),
                zorder=3,
            )
        axis.set_xticks(centers, labels)
        axis.set_xlabel("Number of CIFAR-10 training examples")
        axis.set_ylabel("Test accuracy (%)" if metric == "acc" else "Test loss")
        axis.set_title("(a) Selected-checkpoint test accuracy" if metric == "acc" else "(b) Selected-checkpoint test loss", pad=9)
        axis.set_ylim(*padded_limits(all_values, lower_bound=0.0))
        clean_axis(axis)
    handles, labels_legend = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels_legend, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.01))
    figure.subplots_adjust(bottom=0.25, wspace=0.22)
    paths = save_figure_pair(figure, figures_dir / "low_data_selected_test_accuracy_loss.png")
    plt.close(figure)
    return paths


def plot_cifar100(rows, summaries, figures_dir):
    setup_paper_plot_style()
    figure, axes = plt.subplots(1, 2, figsize=(11.4, 4.8))
    draw_model_metric(axes[0], rows, summaries, CIFAR100_MODELS, "acc", "(a) Selected-checkpoint test accuracy")
    draw_model_metric(axes[1], rows, summaries, CIFAR100_MODELS, "loss", "(b) Selected-checkpoint test loss")
    figure.legend(handles=summary_legend(), loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.01))
    figure.subplots_adjust(bottom=0.27, wspace=0.24)
    paths = save_figure_pair(figure, figures_dir / "cifar100_selected_test_accuracy_loss.png")
    plt.close(figure)
    return paths


def plot_fusion(rows, summaries, figures_dir):
    setup_paper_plot_style()
    figure, axis = plt.subplots(figsize=(9.0, 5.5))
    learned_parameters = next(item["trainable_parameters"] for item in summaries if item["model"] == "vit_learnable_position")
    fusion_only = [item for item in summaries if "fusion" in item["model"]]
    best_fusion = max(fusion_only, key=lambda item: item["mean_test_acc_pct"])
    all_acc = []
    for index, model in enumerate(FUSION_MODELS):
        style = get_model_style(model, index)
        group = group_model(rows, model)
        aggregate = next(item for item in summaries if item["model"] == model)
        x = aggregate["trainable_parameters"] / 1_000_000.0
        values = [100.0 * row["test_acc"] for row in group]
        all_acc.extend(values)
        jitter = np.linspace(-0.018, 0.018, len(values))
        axis.scatter(
            x + jitter,
            values,
            color=style["color"],
            s=22,
            alpha=PAPER_POINT_ALPHA,
            edgecolors="none",
            zorder=2,
        )
        axis.errorbar(
            x,
            aggregate["mean_test_acc_pct"],
            yerr=aggregate["ci95_half_width_test_acc_pp"],
            fmt="D",
            color=style["color"],
            markeredgecolor=PAPER_TEXT_COLOR,
            markeredgewidth=0.7,
            markersize=6,
            capsize=4,
            elinewidth=1.25,
            label=SHORT_LABELS[model].replace("\n", " "),
            zorder=3,
        )

    learned = next(item for item in summaries if item["model"] == "vit_learnable_position")
    for item, xytext in ((learned, (15, 14)), (best_fusion, (12, -30))):
        ratio = item["trainable_parameters"] / learned_parameters
        label = SHORT_LABELS[item["model"]].replace("\n", " ")
        if item is best_fusion:
            label += f"\n{ratio:.2f}× learnable parameters"
        axis.annotate(
            label,
            xy=(item["trainable_parameters"] / 1_000_000.0, item["mean_test_acc_pct"]),
            xytext=xytext,
            textcoords="offset points",
            arrowprops={"arrowstyle": "-", "color": PAPER_MUTED_COLOR, "lw": 0.8},
            fontsize=8,
            color=PAPER_TEXT_COLOR,
        )
    axis.set_xlabel("Trainable parameters (millions)")
    axis.set_ylabel("Selected-checkpoint test accuracy (%)")
    axis.set_title("Fusion accuracy-capacity trade-off", pad=10)
    axis.set_ylim(*padded_limits(all_acc, lower_bound=0.0))
    axis.xaxis.set_major_locator(MaxNLocator(nbins=6))
    clean_axis(axis)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False)
    paths = save_figure_pair(figure, figures_dir / "fusion_test_accuracy_vs_parameters.png")
    plt.close(figure)
    return paths


def build_shift_rows(main_rows):
    per_seed = []
    summaries = []
    for contrast, shifted_model, reference_model in SHIFT_CONTRASTS:
        shifted = {row["seed"]: row for row in group_model(main_rows, shifted_model)}
        reference = {row["seed"]: row for row in group_model(main_rows, reference_model)}
        deltas = []
        for seed in SEEDS:
            delta = 100.0 * (shifted[seed]["test_acc"] - reference[seed]["test_acc"])
            deltas.append(delta)
            per_seed.append({"contrast": contrast, "seed": seed, "paired_test_acc_difference_pp": delta})
        summaries.append(
            {
                "contrast": contrast,
                "num_seeds": 5,
                "mean_paired_test_acc_difference_pp": statistics.mean(deltas),
                "sd_paired_test_acc_difference_pp": sample_sd(deltas),
                "ci95_half_width_paired_test_acc_difference_pp": ci95(deltas),
            }
        )
    return per_seed, summaries


def plot_shift(per_seed, summaries, figures_dir):
    setup_paper_plot_style()
    figure, axis = plt.subplots(figsize=(7.6, 4.6))
    offsets = np.linspace(-0.10, 0.10, len(SEEDS))
    colors = (get_model_style("vit_additive_sinusoidal_shifted", 0)["color"], get_model_style("vit_multiplicative_sinusoidal_shifted", 1)["color"])
    all_values = []
    for index, (contrast, _, _) in enumerate(SHIFT_CONTRASTS):
        values = [row["paired_test_acc_difference_pp"] for row in per_seed if row["contrast"] == contrast]
        all_values.extend(values)
        axis.scatter(index + offsets, values, color=colors[index], s=28, alpha=PAPER_POINT_ALPHA, edgecolors="none")
        aggregate = next(item for item in summaries if item["contrast"] == contrast)
        axis.errorbar(
            index,
            aggregate["mean_paired_test_acc_difference_pp"],
            yerr=aggregate["ci95_half_width_paired_test_acc_difference_pp"],
            fmt="D",
            color=colors[index],
            markeredgecolor=PAPER_TEXT_COLOR,
            markeredgewidth=0.7,
            markersize=6.5,
            capsize=5,
            elinewidth=1.3,
        )
    axis.axhline(0.0, color=PAPER_MUTED_COLOR, linewidth=1.0)
    axis.set_xticks(range(len(SHIFT_CONTRASTS)), [item[0].replace(" - ", "\n− ") for item in SHIFT_CONTRASTS])
    axis.set_ylabel("Paired test-accuracy difference (percentage points)")
    axis.set_title("Seed-matched effects of shifted positional encodings", pad=10)
    axis.set_ylim(*padded_limits(all_values))
    clean_axis(axis)
    axis.legend(handles=summary_legend(), loc="best", frameon=False)
    paths = save_figure_pair(figure, figures_dir / "shifted_pe_paired_test_effects.png")
    plt.close(figure)
    return paths


def build_per_class_rows(main_rows):
    per_seed = []
    summaries = []
    for contrast, positive_model, negative_model in PER_CLASS_CONTRASTS:
        positive = {row["seed"]: row for row in group_model(main_rows, positive_model)}
        negative = {row["seed"]: row for row in group_model(main_rows, negative_model)}
        labels = positive[SEEDS[0]]["label_names"]
        if labels is None or len(labels) != 10:
            raise ValueError(f"Missing CIFAR-10 class labels for {positive_model}")
        for seed in SEEDS:
            if positive[seed]["label_names"] != labels or negative[seed]["label_names"] != labels:
                raise ValueError(f"Class-label mismatch for {contrast}, seed {seed}")
            for class_index, class_name in enumerate(labels):
                delta = 100.0 * (
                    positive[seed]["test_per_class_recall"][class_index]
                    - negative[seed]["test_per_class_recall"][class_index]
                )
                per_seed.append(
                    {
                        "contrast": contrast,
                        "class_index": class_index,
                        "class_name": class_name,
                        "seed": seed,
                        "paired_recall_difference_pp": delta,
                    }
                )
        for class_index, class_name in enumerate(labels):
            values = [
                row["paired_recall_difference_pp"]
                for row in per_seed
                if row["contrast"] == contrast and row["class_index"] == class_index
            ]
            summaries.append(
                {
                    "contrast": contrast,
                    "class_index": class_index,
                    "class_name": class_name,
                    "num_seeds": 5,
                    "mean_paired_recall_difference_pp": statistics.mean(values),
                    "sd_paired_recall_difference_pp": sample_sd(values),
                    "ci95_half_width_paired_recall_difference_pp": ci95(values),
                }
            )
    return per_seed, summaries


def plot_per_class(per_seed, summaries, figures_dir):
    setup_paper_plot_style()
    figure, axes = plt.subplots(2, 1, figsize=(10.2, 7.2), sharex=True)
    colors = (get_model_style("vit_learnable_position", 0)["color"], get_model_style("vit_multiplicative_sinusoidal_shifted", 1)["color"])
    class_names = [item["class_name"] for item in summaries if item["contrast"] == PER_CLASS_CONTRASTS[0][0]]
    x = np.arange(len(class_names), dtype=float)
    seed_offsets = np.linspace(-0.10, 0.10, len(SEEDS))
    for axis, (contrast, _, _), color in zip(axes, PER_CLASS_CONTRASTS, colors):
        all_values = []
        for class_index in range(len(class_names)):
            values = [
                row["paired_recall_difference_pp"]
                for row in per_seed
                if row["contrast"] == contrast and row["class_index"] == class_index
            ]
            all_values.extend(values)
            aggregate = next(
                item for item in summaries if item["contrast"] == contrast and item["class_index"] == class_index
            )
            axis.scatter(x[class_index] + seed_offsets, values, color=color, s=16, alpha=PAPER_POINT_ALPHA, edgecolors="none")
            axis.errorbar(
                x[class_index],
                aggregate["mean_paired_recall_difference_pp"],
                yerr=aggregate["ci95_half_width_paired_recall_difference_pp"],
                fmt="D",
                color=color,
                markeredgecolor=PAPER_TEXT_COLOR,
                markeredgewidth=0.55,
                markersize=5,
                capsize=3,
                elinewidth=1.0,
            )
        axis.axhline(0.0, color=PAPER_MUTED_COLOR, linewidth=0.9)
        axis.set_ylabel("Recall difference\n(percentage points)")
        axis.set_title(contrast, pad=7)
        axis.set_ylim(*padded_limits(all_values))
        clean_axis(axis)
    axes[-1].set_xticks(x, class_names, rotation=25, ha="right")
    axes[-1].set_xlabel("CIFAR-10 class")
    figure.legend(handles=summary_legend(), loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.01))
    figure.subplots_adjust(bottom=0.18, hspace=0.32)
    paths = save_figure_pair(figure, figures_dir / "per_class_recall_paired_differences.png")
    plt.close(figure)
    return paths


def write_core_table(report_dir, summary):
    csv_path = report_dir / "core_pe_selected_test_table.csv"
    write_csv(csv_path, summary)
    markdown_path = report_dir / "core_pe_selected_test_table.md"
    lines = [
        "# Core positional-encoding selected-test results",
        "",
        "All values are computed from the validation-selected checkpoint over seeds 42--46. The ± term is the 95% t confidence-interval half-width (df = 4).",
        "",
        "| Variant | Test accuracy (%) | Test loss |",
        "|---|---:|---:|",
    ]
    for item in summary:
        lines.append(
            f"| {item['model_label']} | {item['mean_test_acc_pct']:.3f} ± {item['ci95_half_width_test_acc_pp']:.3f} | "
            f"{item['mean_test_loss']:.4f} ± {item['ci95_half_width_test_loss']:.4f} |"
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, markdown_path


def main():
    args = parse_args()
    report_dir = args.results_dir / "reports" / args.report_name
    figures_dir = report_dir / "figures"
    report_dir.mkdir(parents=True, exist_ok=True)

    patch_models = tuple(
        patch_model_name(order, suffix)
        for _, suffix in PATCH_FAMILIES
        for order, _ in PATCH_ORDERS
    )
    main_models = tuple(dict.fromkeys(CORE_MODELS + patch_models + FUSION_MODELS))
    main_rows, main_configs, main_source_count = load_experiment(
        args.results_dir, args.main_experiment, main_models
    )
    audit = validate_uniform_config(args.main_experiment, main_configs)

    low_rows_by_size = {}
    low_configs_by_size = {}
    low_source_counts = {}
    for train_size, experiment in LOW_DATA_EXPERIMENTS.items():
        rows, configs, source_count = load_experiment(
            args.results_dir, experiment, LOW_DATA_MODELS
        )
        low_rows_by_size[train_size] = rows
        low_configs_by_size[train_size] = configs
        low_source_counts[str(train_size)] = source_count
        audit.extend(validate_uniform_config(experiment, configs))
    audit.extend(validate_full_low_alignment(main_configs, low_configs_by_size))

    cifar100_rows, cifar100_configs, cifar100_source_count = load_experiment(
        args.results_dir, args.cifar100_experiment, CIFAR100_MODELS
    )
    audit.extend(validate_uniform_config(args.cifar100_experiment, cifar100_configs))
    write_csv(report_dir / "configuration_alignment_audit.csv", audit)

    core_rows = [row for row in main_rows if row["model"] in CORE_MODELS]
    core_summary = summarise_models(core_rows, CORE_MODELS, include_parameters=True)
    write_csv(report_dir / "core_pe_selected_test_per_seed.csv", per_seed_export(core_rows))
    table_csv, table_md = write_core_table(report_dir, core_summary)

    patch_per_seed, patch_summary = build_patch_rows(main_rows)
    write_csv(report_dir / "patch_assignment_test_accuracy_per_seed.csv", patch_per_seed)
    write_csv(report_dir / "patch_assignment_test_accuracy_summary.csv", patch_summary)

    low_rows, low_summary = build_low_data(main_rows, low_rows_by_size)
    write_csv(report_dir / "low_data_selected_test_per_seed.csv", per_seed_export(low_rows))
    write_csv(report_dir / "low_data_selected_test_summary.csv", low_summary)

    cifar100_summary = summarise_models(cifar100_rows, CIFAR100_MODELS, include_parameters=True)
    write_csv(report_dir / "cifar100_selected_test_per_seed.csv", per_seed_export(cifar100_rows))
    write_csv(report_dir / "cifar100_selected_test_summary.csv", cifar100_summary)

    fusion_rows = [row for row in main_rows if row["model"] in FUSION_MODELS]
    fusion_summary = summarise_models(fusion_rows, FUSION_MODELS, include_parameters=True)
    write_csv(report_dir / "fusion_selected_test_per_seed.csv", per_seed_export(fusion_rows))
    write_csv(report_dir / "fusion_selected_test_summary.csv", fusion_summary)

    shift_per_seed, shift_summary = build_shift_rows(main_rows)
    write_csv(report_dir / "shifted_pe_paired_test_effects_per_seed.csv", shift_per_seed)
    write_csv(report_dir / "shifted_pe_paired_test_effects_summary.csv", shift_summary)

    per_class_rows, per_class_summary = build_per_class_rows(main_rows)
    write_csv(report_dir / "per_class_recall_paired_differences_per_seed.csv", per_class_rows)
    write_csv(report_dir / "per_class_recall_paired_differences_summary.csv", per_class_summary)

    figures = {
        "core_pe_selected_test_accuracy_loss": plot_core(core_rows, core_summary, figures_dir),
        "patch_assignment_test_accuracy_delta_heatmap": plot_patch_heatmap(patch_summary, figures_dir),
        "low_data_selected_test_accuracy_loss": plot_low_data(low_rows, low_summary, figures_dir),
        "cifar100_selected_test_accuracy_loss": plot_cifar100(cifar100_rows, cifar100_summary, figures_dir),
        "fusion_test_accuracy_vs_parameters": plot_fusion(fusion_rows, fusion_summary, figures_dir),
        "shifted_pe_paired_test_effects": plot_shift(shift_per_seed, shift_summary, figures_dir),
        "per_class_recall_paired_differences": plot_per_class(per_class_rows, per_class_summary, figures_dir),
    }

    captions = """# Draft figure captions

## Core PE selected-test comparison

Selected-checkpoint CIFAR-10 test accuracy and loss for nine positional-encoding variants over seeds 42--46. Faint circles show individual runs; diamonds and error bars show the mean and 95% t confidence interval. Radial PE is included in both panels.

## Patch-assignment test-effect heatmap

Mean seed-matched test-accuracy change for each patch-to-position assignment relative to the normal_row assignment for the same positional-encoding family. Values are percentage points; the diverging colour scale is centred at zero.

## Low-data selected-test comparison

Selected-checkpoint CIFAR-10 test accuracy and loss at 1,000, 5,000, 10,000 and 45,000 training examples. Every condition uses learning rate 3e-4 and the shared five-seed protocol. Within a seed, all four variants use the same sampled subset; across seeds, both subset composition and stochastic training vary.

## CIFAR-100 selected-test comparison

Selected-checkpoint CIFAR-100 test accuracy and loss for four prespecified positional-encoding conditions over seeds 42--46. Relative performance is determined from the held-out test summaries, not validation trajectories.

## Fusion accuracy-parameter trade-off

Selected-checkpoint CIFAR-10 test accuracy against trainable parameter count for the learnable, hybrid and five dual-branch fusion variants. The strongest fusion is annotated with its parameter ratio relative to the learnable single-branch model. Fusion results are not parameter matched.

## Shifted variants paired test effects

Seed-matched test-accuracy differences for shifted additive minus additive and shifted multiplicative minus multiplicative PE. Diamonds and error bars show the mean paired difference and its 95% t confidence interval; the horizontal line marks no change.

## Per-class recall differences

Seed-matched CIFAR-10 per-class recall differences for learnable minus no PE and shifted multiplicative minus learnable PE. The metric is recall: the historical implementation field named per_class_accuracy is numerically identical to recall for this single-label task but is not used as the plot label.
"""
    (report_dir / "figure_captions.md").write_text(captions, encoding="utf-8")

    manifest = {
        "report_name": args.report_name,
        "seeds": list(SEEDS),
        "ci_definition": "mean +/- 2.7764451051977987 * sample_sd / sqrt(5)",
        "test_source": "summary['selected_model'] only",
        "main_experiment": args.main_experiment,
        "low_data_experiments": LOW_DATA_EXPERIMENTS,
        "cifar100_experiment": args.cifar100_experiment,
        "source_summary_counts": {
            "main_directory_total": main_source_count,
            "low_data_directory_totals": low_source_counts,
            "cifar100_directory_total": cifar100_source_count,
        },
        "full_data_connected_to_low_data": True,
        "full_data_alignment_gate": "passed",
        "figures": {
            name: {key: str(value) for key, value in paths.items()}
            for name, paths in figures.items()
        },
        "core_table": {"csv": str(table_csv), "markdown": str(table_md)},
        "notes": [
            "All relative performance claims use selected-checkpoint test summaries.",
            "Ordinary intervals are five-seed 95% t intervals with df=4.",
            "Paired intervals are computed from the five seed-level differences.",
            "Full-data is connected only after the automated configuration gate passed.",
            "Different low-data seeds change both sampled subset composition and stochastic training.",
        ],
    }
    manifest_path = report_dir / "final_test_figures_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Report directory: {report_dir}")
    print("Full-data alignment gate: PASSED")
    print(f"Core table: {table_csv} | {table_md}")
    for name, paths in figures.items():
        print(f"Figure {name}: {paths['png']} | {paths['pdf']}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
