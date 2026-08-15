import argparse
import csv
import json
import os
import statistics
from argparse import Namespace
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("results/matplotlib_cache")))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from models.registry import EXPERIMENT_REGISTRY
from models.unfolding import build_patch_order
from paper_plotting import (
    PAPER_FIGSIZE,
    PAPER_EPOCH_PANEL_FIGSIZE,
    PAPER_FACET_FIGSIZE,
    PAPER_GRID_COLOR,
    PAPER_MUTED_COLOR,
    PAPER_POINT_ALPHA,
    PAPER_TALL_FIGSIZE,
    PAPER_TEXT_COLOR,
    get_assignment_style,
    get_model_label,
    get_model_style,
    save_figure_pair,
    setup_paper_plot_style,
)


DEFAULT_EXPERIMENT = "cifar10_final_vit_models_5seeds"
DEFAULT_REPORT = "thesis_comparison_figures_v2"
DEFAULT_SEEDS = (42, 43, 44, 45, 46)
T_CRITICAL_95_DF4 = 2.7764451051977987

BASIC_MODELS = (
    "vit_baseline",
    "vit_learnable_position",
    "vit_row_sinusoidal",
    "vit_col_sinusoidal",
    "vit_additive_sinusoidal",
    "vit_multiplicative_sinusoidal",
)

SHIFT_CONTRASTS = (
    ("Additive", "vit_additive_sinusoidal", "vit_additive_sinusoidal_shifted"),
    (
        "Multiplicative",
        "vit_multiplicative_sinusoidal",
        "vit_multiplicative_sinusoidal_shifted",
    ),
)

PATCH_FAMILIES = (
    ("No PE", "baseline"),
    ("Learnable PE", "learnable_position"),
    ("Row PE", "row_sinusoidal"),
    ("Column PE", "col_sinusoidal"),
    ("Multiplicative PE", "multiplicative_sinusoidal"),
)

PATCH_ORDERS = (
    ("normal_row", "Row-major"),
    ("normal_col", "Column-major"),
    ("proper_row", "Serpentine rows"),
    ("proper_col", "Serpentine columns"),
)

FUSION_MODELS = (
    "vit_row_sinusoidal",
    "vit_col_sinusoidal",
    "vit_learnable_position",
    "vit_row_col_mean_fusion",
    "vit_row_col_mean_mlp_fusion",
    "vit_row_col_latent_fusion",
    "vit_row_col_cross_attention_fusion",
    "vit_row_col_cross_attention_mlp_head_fusion",
)

CONFIG_FIELDS = (
    "dataset",
    "epochs",
    "batch_size",
    "lr",
    "weight_decay",
    "train_subset",
    "val_subset",
    "test_subset",
    "val_ratio",
    "split_seed",
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

EPOCH_METRICS = ("val_acc", "val_loss")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate thesis comparisons from selected-checkpoint multi-seed summaries."
    )
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--report-name", default=DEFAULT_REPORT)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    return parser.parse_args()


def patch_model_name(order, suffix):
    return f"vit_{suffix}" if order == "normal_row" else f"vit_{order}_{suffix}"


def required_models():
    models = set(BASIC_MODELS)
    for _, unshifted, shifted in SHIFT_CONTRASTS:
        models.update((unshifted, shifted))
    for _, suffix in PATCH_FAMILIES:
        for order, _ in PATCH_ORDERS:
            models.add(patch_model_name(order, suffix))
    models.update(FUSION_MODELS)
    return models


def load_selected_results(metrics_dir, seeds):
    rows = []
    configs = {}
    paths = sorted(metrics_dir.glob("*/*_summary.json"))
    if not paths:
        raise FileNotFoundError(f"No summary JSON files found under {metrics_dir}")

    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        config = data["config"]
        selected = data["selected_model"]
        model = config["model"]
        seed = int(config["seed"])
        key = (model, seed)
        if key in configs:
            raise ValueError(f"Duplicate summary for {model}, seed {seed}")
        configs[key] = config
        rows.append(
            {
                "model": model,
                "model_label": get_model_label(model),
                "seed": seed,
                "selected_epoch": int(selected["epoch"]),
                "val_acc": float(selected["val_acc"]),
                "test_acc": float(selected["test_acc"]),
                "test_macro_f1": float(selected["test_macro_f1"]),
                "protocol": data.get("test_evaluation_protocol"),
                "summary_path": str(path),
            }
        )

    available_models = {row["model"] for row in rows}
    missing_models = sorted(required_models() - available_models)
    if missing_models:
        raise ValueError(f"Required models are missing: {missing_models}")

    expected_seeds = set(seeds)
    selected_rows = [row for row in rows if row["model"] in required_models()]
    for model in sorted(required_models()):
        model_subset = [row for row in selected_rows if row["model"] == model]
        observed = {row["seed"] for row in model_subset}
        if observed != expected_seeds:
            raise ValueError(
                f"{model} has seeds {sorted(observed)}, expected {sorted(expected_seeds)}"
            )
        if {row["protocol"] for row in model_subset} != {"selected_checkpoint_only"}:
            raise ValueError(f"{model} does not use selected-checkpoint-only test evaluation")

    selected_configs = [configs[(row["model"], row["seed"])] for row in selected_rows]
    for field in CONFIG_FIELDS:
        values = {json.dumps(config.get(field), sort_keys=True) for config in selected_configs}
        if len(values) != 1:
            raise ValueError(f"Non-uniform comparison config field {field}: {sorted(values)}")
    signature = {field: selected_configs[0].get(field) for field in CONFIG_FIELDS}
    return rows, configs, signature, len(paths)


def selected_rows(rows, models, seeds):
    model_set = set(models)
    seed_set = set(seeds)
    return [row for row in rows if row["model"] in model_set and row["seed"] in seed_set]


def model_rows(rows, model, seeds):
    seed_set = set(seeds)
    return sorted(
        [row for row in rows if row["model"] == model and row["seed"] in seed_set],
        key=lambda row: row["seed"],
    )


def sample_sd(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


def ci95_half_width(values):
    if len(values) != 5:
        raise ValueError("The thesis 95% t interval currently expects five seeds")
    return T_CRITICAL_95_DF4 * sample_sd(values) / np.sqrt(len(values))


def summarise(rows, models):
    summaries = []
    for model in models:
        subset = [row for row in rows if row["model"] == model]
        test_acc = [row["test_acc"] for row in subset]
        test_f1 = [row["test_macro_f1"] for row in subset]
        epochs = [row["selected_epoch"] for row in subset]
        summaries.append(
            {
                "model": model,
                "model_label": get_model_label(model),
                "num_seeds": len(subset),
                "mean_test_acc": statistics.mean(test_acc),
                "sd_test_acc": sample_sd(test_acc),
                "ci95_half_width_test_acc": ci95_half_width(test_acc),
                "min_test_acc": min(test_acc),
                "max_test_acc": max(test_acc),
                "mean_test_macro_f1": statistics.mean(test_f1),
                "sd_test_macro_f1": sample_sd(test_f1),
                "ci95_half_width_test_macro_f1": ci95_half_width(test_f1),
                "mean_selected_epoch": statistics.mean(epochs),
            }
        )
    return summaries


def load_epoch_history_rows(result_rows, models, seeds):
    histories = []
    for result in selected_rows(result_rows, models, seeds):
        summary_path = Path(result["summary_path"])
        metrics_path = summary_path.with_name(
            summary_path.name.replace("_summary.json", "_metrics.csv")
        )
        if not metrics_path.exists():
            raise FileNotFoundError(f"Missing metrics CSV for {summary_path}: {metrics_path}")
        with metrics_path.open(newline="", encoding="utf-8") as handle:
            for raw in csv.DictReader(handle):
                epoch = int(float(raw["epoch"]))
                for metric in EPOCH_METRICS:
                    value = raw.get(metric)
                    if value in (None, ""):
                        continue
                    histories.append(
                        {
                            "model": result["model"],
                            "model_label": result["model_label"],
                            "seed": result["seed"],
                            "epoch": epoch,
                            "metric": metric,
                            "value": float(value),
                            "metrics_path": str(metrics_path),
                        }
                    )
    return histories


def aggregate_epoch_history(histories, models, seeds):
    expected_count = len(seeds)
    aggregated = []
    for model in models:
        for metric in EPOCH_METRICS:
            epochs = sorted(
                {
                    row["epoch"]
                    for row in histories
                    if row["model"] == model and row["metric"] == metric
                }
            )
            for epoch in epochs:
                values = [
                    row["value"]
                    for row in histories
                    if row["model"] == model
                    and row["metric"] == metric
                    and row["epoch"] == epoch
                ]
                if len(values) != expected_count:
                    continue
                aggregated.append(
                    {
                        "model": model,
                        "model_label": get_model_label(model),
                        "metric": metric,
                        "epoch": epoch,
                        "count": len(values),
                        "mean": statistics.mean(values),
                        "sd": sample_sd(values),
                        "min": min(values),
                        "max": max(values),
                    }
                )
    return aggregated


def epoch_curve_rows(aggregated, model, metric):
    return sorted(
        [
            row
            for row in aggregated
            if row["model"] == model and row["metric"] == metric
        ],
        key=lambda row: row["epoch"],
    )


def draw_epoch_curve(axis, aggregated, model, metric, index, label=None, style_override=None):
    curve = epoch_curve_rows(aggregated, model, metric)
    if not curve:
        raise ValueError(f"No full-seed epoch curve for {model}, {metric}")
    epochs = np.array([row["epoch"] for row in curve])
    scale = 100.0 if metric.endswith("acc") else 1.0
    means = scale * np.array([row["mean"] for row in curve])
    deviations = scale * np.array([row["sd"] for row in curve])
    style = style_override or get_model_style(model, index)
    axis.plot(
        epochs,
        means,
        color=style["color"],
        linestyle=style["linestyle"],
        linewidth=1.8,
        label=label or get_model_label(model),
    )
    axis.fill_between(
        epochs,
        means - deviations,
        means + deviations,
        color=style["color"],
        alpha=0.11,
        linewidth=0,
    )


def epoch_axis_limits(aggregated, models, metric):
    model_set = set(models)
    rows = [
        row
        for row in aggregated
        if row["model"] in model_set and row["metric"] == metric
    ]
    scale = 100.0 if metric == "val_acc" else 1.0
    lower_values = [scale * (row["mean"] - row["sd"]) for row in rows]
    upper_values = [scale * (row["mean"] + row["sd"]) for row in rows]
    lower = min(lower_values)
    upper = max(upper_values)
    padding = max(1.0 if metric == "val_acc" else 0.05, 0.06 * (upper - lower))
    upper_bound = 100.0 if metric == "val_acc" else upper + padding
    return max(0.0, lower - padding), min(100.0, upper + padding) if metric == "val_acc" else upper_bound


def finish_epoch_comparison_axis(axis, metric, title, y_limits=None):
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Validation accuracy (%)" if metric == "val_acc" else "Validation loss")
    axis.set_title(title, pad=8)
    axis.grid(True, axis="y", linestyle="--", linewidth=0.7, color=PAPER_GRID_COLOR)
    axis.grid(False, axis="x")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    if y_limits is not None:
        axis.set_ylim(*y_limits)


def plot_epoch_pair(aggregated, models, figures_dir, stem, title):
    setup_paper_plot_style()
    figure, axes = plt.subplots(1, 2, figsize=PAPER_EPOCH_PANEL_FIGSIZE)
    for index, model in enumerate(models):
        draw_epoch_curve(axes[0], aggregated, model, "val_acc", index)
        draw_epoch_curve(axes[1], aggregated, model, "val_loss", index)
    finish_epoch_comparison_axis(
        axes[0],
        "val_acc",
        "(a) Validation accuracy",
        epoch_axis_limits(aggregated, models, "val_acc"),
    )
    finish_epoch_comparison_axis(
        axes[1],
        "val_loss",
        "(b) Validation loss",
        epoch_axis_limits(aggregated, models, "val_loss"),
    )
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=min(3, len(models)),
        frameon=False,
    )
    figure.suptitle(title, fontsize=12, y=1.01)
    paths = save_figure_pair(figure, figures_dir / f"{stem}.png")
    plt.close(figure)
    return paths


def plot_patch_epoch_facets(aggregated, figures_dir, metric):
    setup_paper_plot_style()
    figure, axes = plt.subplots(2, 3, figsize=PAPER_FACET_FIGSIZE, sharey=True)
    for family_index, ((family_label, suffix), axis) in enumerate(
        zip(PATCH_FAMILIES, axes.flat)
    ):
        for order_index, (order, order_label) in enumerate(PATCH_ORDERS):
            model = patch_model_name(order, suffix)
            draw_epoch_curve(
                axis,
                aggregated,
                model,
                metric,
                family_index,
                label=order_label,
                style_override=get_assignment_style(order),
            )
        finish_epoch_comparison_axis(axis, metric, family_label)
    axes.flat[-1].axis("off")
    patch_models = [
        patch_model_name(order, suffix)
        for _, suffix in PATCH_FAMILIES
        for order, _ in PATCH_ORDERS
    ]
    y_limits = epoch_axis_limits(aggregated, patch_models, metric)
    for axis in axes.flat[:-1]:
        axis.set_ylim(*y_limits)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=4,
        frameon=False,
    )
    metric_label = "validation accuracy" if metric == "val_acc" else "validation loss"
    figure.suptitle(
        f"Patch-to-position assignment: {metric_label} across epochs",
        fontsize=12,
        y=1.01,
    )
    stem = f"patch_assignment_{metric}_epoch"
    paths = save_figure_pair(figure, figures_dir / f"{stem}.png")
    plt.close(figure)
    return paths


def write_rows(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def model_color(model, index=0):
    return get_model_style(model, index)["color"]


def clean_axis(axis, grid_axis="x"):
    axis.grid(True, axis=grid_axis, linestyle="--", linewidth=0.7, color=PAPER_GRID_COLOR)
    axis.grid(False, axis="y" if grid_axis == "x" else "x")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def plot_basic_pe(rows, figures_dir, seeds):
    setup_paper_plot_style()
    figure, axis = plt.subplots(figsize=PAPER_FIGSIZE)
    offsets = np.linspace(-0.16, 0.16, len(seeds))
    all_values = []

    for index, model in enumerate(BASIC_MODELS):
        values = np.array([100.0 * row["test_acc"] for row in model_rows(rows, model, seeds)])
        color = model_color(model, index)
        all_values.extend(values.tolist())
        axis.scatter(index + offsets, values, s=28, color=color, alpha=PAPER_POINT_ALPHA, edgecolors="none", zorder=2)
        axis.errorbar(
            index, values.mean(), yerr=ci95_half_width(values.tolist()), fmt="D", markersize=6.5,
            color=color, markeredgecolor=PAPER_TEXT_COLOR, markeredgewidth=0.8,
            ecolor=PAPER_TEXT_COLOR, elinewidth=1.5, capsize=4, zorder=3,
        )

    axis.set_xticks(
        range(len(BASIC_MODELS)),
        [get_model_label(model) for model in BASIC_MODELS],
        rotation=22,
        ha="right",
    )
    axis.set_ylabel("Selected-checkpoint test accuracy (%)")
    axis.set_ylim(np.floor(min(all_values) - 0.8), np.ceil(max(all_values) + 0.8))
    axis.set_title("Basic positional encoding comparison", pad=10)
    clean_axis(axis, grid_axis="y")
    axis.legend(
        handles=[
            Line2D([0], [0], marker="o", linestyle="", color=PAPER_MUTED_COLOR, alpha=PAPER_POINT_ALPHA, label="Individual seed"),
            Line2D([0], [0], marker="D", linestyle="-", color=PAPER_TEXT_COLOR, label="Mean ± 95% CI"),
        ],
        loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=2, frameon=False,
    )
    paths = save_figure_pair(figure, figures_dir / "basic_pe_comparison.png")
    plt.close(figure)
    return paths


def build_shift_tables(rows, seeds):
    paired = []
    summary = []
    for label, unshifted, shifted in SHIFT_CONTRASTS:
        left = {row["seed"]: row for row in model_rows(rows, unshifted, seeds)}
        right = {row["seed"]: row for row in model_rows(rows, shifted, seeds)}
        deltas = []
        for seed in seeds:
            delta = 100.0 * (right[seed]["test_acc"] - left[seed]["test_acc"])
            deltas.append(delta)
            paired.append(
                {
                    "contrast": label,
                    "seed": seed,
                    "unshifted_model": unshifted,
                    "shifted_model": shifted,
                    "unshifted_test_acc": left[seed]["test_acc"],
                    "shifted_test_acc": right[seed]["test_acc"],
                    "delta_shifted_minus_unshifted_pp": delta,
                }
            )
        summary.append(
            {
                "contrast": label,
                "unshifted_model": unshifted,
                "shifted_model": shifted,
                "num_seeds": len(deltas),
                "mean_delta_pp": statistics.mean(deltas),
                "sd_delta_pp": sample_sd(deltas),
                "ci95_half_width_delta_pp": ci95_half_width(deltas),
                "min_delta_pp": min(deltas),
                "max_delta_pp": max(deltas),
            }
        )
    return paired, summary


def plot_shift_effect(paired, summary, figures_dir, seeds):
    setup_paper_plot_style()
    figure, axis = plt.subplots(figsize=PAPER_FIGSIZE)
    offsets = np.linspace(-0.12, 0.12, len(seeds))
    summary_lookup = {row["contrast"]: row for row in summary}

    for index, (label, _, shifted) in enumerate(SHIFT_CONTRASTS):
        values = sorted([row for row in paired if row["contrast"] == label], key=lambda row: row["seed"])
        deltas = np.array([row["delta_shifted_minus_unshifted_pp"] for row in values])
        aggregate = summary_lookup[label]
        color = model_color(shifted, index)
        axis.scatter(index + offsets, deltas, s=34, color=color, alpha=PAPER_POINT_ALPHA, edgecolors="none", zorder=2)
        axis.errorbar(
            index, aggregate["mean_delta_pp"], yerr=aggregate["ci95_half_width_delta_pp"], fmt="D",
            markersize=7, color=color, markeredgecolor=PAPER_TEXT_COLOR, markeredgewidth=0.8,
            ecolor=PAPER_TEXT_COLOR, elinewidth=1.5, capsize=5, zorder=3,
        )

    axis.axhline(0.0, color=PAPER_MUTED_COLOR, linewidth=1.2)
    axis.set_xticks(range(len(SHIFT_CONTRASTS)), [item[0] for item in SHIFT_CONTRASTS])
    axis.set_ylabel("Shifted − unshifted test accuracy (percentage points)")
    axis.set_title("Paired effect of wavelength shifting", pad=10)
    clean_axis(axis, grid_axis="y")
    axis.legend(
        handles=[
            Line2D([0], [0], marker="o", linestyle="", color=PAPER_MUTED_COLOR, alpha=PAPER_POINT_ALPHA, label="Seed-level paired difference"),
            Line2D([0], [0], marker="D", linestyle="-", color=PAPER_TEXT_COLOR, label="Mean ± 95% CI"),
        ],
        loc="best", frameon=False,
    )
    paths = save_figure_pair(figure, figures_dir / "shift_paired_effect.png")
    plt.close(figure)
    return paths


def build_patch_tables(rows, seeds):
    per_seed = []
    for family_label, suffix in PATCH_FAMILIES:
        reference_model = patch_model_name("normal_row", suffix)
        reference = {row["seed"]: row for row in model_rows(rows, reference_model, seeds)}
        for order, order_label in PATCH_ORDERS:
            model = patch_model_name(order, suffix)
            current = {row["seed"]: row for row in model_rows(rows, model, seeds)}
            for seed in seeds:
                per_seed.append(
                    {
                        "family": family_label,
                        "suffix": suffix,
                        "order": order,
                        "order_label": order_label,
                        "model": model,
                        "seed": seed,
                        "test_acc": current[seed]["test_acc"],
                        "delta_vs_row_major_pp": 100.0 * (current[seed]["test_acc"] - reference[seed]["test_acc"]),
                    }
                )

    summary = []
    for family_label, suffix in PATCH_FAMILIES:
        for order, order_label in PATCH_ORDERS:
            subset = [row for row in per_seed if row["family"] == family_label and row["order"] == order]
            test_acc = [row["test_acc"] for row in subset]
            deltas = [row["delta_vs_row_major_pp"] for row in subset]
            summary.append(
                {
                    "family": family_label,
                    "suffix": suffix,
                    "order": order,
                    "order_label": order_label,
                    "model": patch_model_name(order, suffix),
                    "num_seeds": len(subset),
                    "mean_test_acc": statistics.mean(test_acc),
                    "sd_test_acc": sample_sd(test_acc),
                    "mean_delta_vs_row_major_pp": statistics.mean(deltas),
                    "sd_delta_vs_row_major_pp": sample_sd(deltas),
                    "ci95_half_width_delta_vs_row_major_pp": ci95_half_width(deltas),
                }
            )
    return per_seed, summary


def plot_patch_deltas(per_seed, summary, figures_dir, seeds):
    setup_paper_plot_style()
    figure, axis = plt.subplots(figsize=PAPER_TALL_FIGSIZE)
    plotted_orders = PATCH_ORDERS[1:]
    family_offsets = np.linspace(-0.28, 0.28, len(PATCH_FAMILIES))
    seed_offsets = np.linspace(-0.045, 0.045, len(seeds))

    for family_index, (family_label, suffix) in enumerate(PATCH_FAMILIES):
        base_model = patch_model_name("normal_row", suffix)
        style = get_model_style(base_model, family_index)
        mean_points = []
        x_points = []
        for order_index, (order, _) in enumerate(plotted_orders):
            x = order_index + family_offsets[family_index]
            values = sorted(
                [row for row in per_seed if row["family"] == family_label and row["order"] == order],
                key=lambda row: row["seed"],
            )
            deltas = np.array([row["delta_vs_row_major_pp"] for row in values])
            aggregate = next(row for row in summary if row["family"] == family_label and row["order"] == order)
            axis.scatter(x + seed_offsets, deltas, s=17, color=style["color"], alpha=0.28, edgecolors="none", zorder=2)
            axis.errorbar(
                x, aggregate["mean_delta_vs_row_major_pp"], yerr=aggregate["ci95_half_width_delta_vs_row_major_pp"],
                fmt="D", markersize=5.5, color=style["color"],
                markeredgecolor=PAPER_TEXT_COLOR, markeredgewidth=0.6, ecolor=style["color"],
                elinewidth=1.2, capsize=3, zorder=3,
            )
            x_points.append(x)
            mean_points.append(aggregate["mean_delta_vs_row_major_pp"])
        axis.plot(x_points, mean_points, color=style["color"], linewidth=1.2, alpha=0.8, zorder=1)

    axis.axhline(0.0, color=PAPER_MUTED_COLOR, linewidth=1.2)
    axis.set_xticks(range(len(plotted_orders)), [label for _, label in plotted_orders])
    axis.set_ylabel("Difference from row-major test accuracy (percentage points)")
    axis.set_title("Interaction between patch-to-position assignment and positional encoding", pad=10)
    clean_axis(axis, grid_axis="y")
    handles = []
    for index, (family_label, suffix) in enumerate(PATCH_FAMILIES):
        base_model = patch_model_name("normal_row", suffix)
        style = get_model_style(base_model, index)
        handles.append(Line2D([0], [0], color=style["color"], marker="D", linewidth=1.3, label=family_label))
    axis.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, frameon=False)
    paths = save_figure_pair(figure, figures_dir / "patch_assignment_paired_deltas.png")
    plt.close(figure)
    return paths


def plot_patch_schematic(figures_dir):
    setup_paper_plot_style()
    figure, axes = plt.subplots(2, 2, figsize=PAPER_TALL_FIGSIZE)
    grid_size = 8
    for axis, (order, order_label) in zip(axes.flat, PATCH_ORDERS):
        physical_indices = build_patch_order(grid_size, order).tolist()
        sequence_positions = np.zeros((grid_size, grid_size), dtype=int)
        for position, physical_index in enumerate(physical_indices):
            row, column = divmod(int(physical_index), grid_size)
            sequence_positions[row, column] = position
        axis.imshow(sequence_positions, cmap="Blues", vmin=0, vmax=grid_size**2 - 1)
        for row in range(grid_size):
            for column in range(grid_size):
                value = sequence_positions[row, column]
                axis.text(column, row, str(value), ha="center", va="center", fontsize=5.5, color="white" if value >= 36 else PAPER_TEXT_COLOR)
        axis.set_title(order_label)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_color(PAPER_GRID_COLOR)
    figure.suptitle("Patch-to-position assignment on an 8 × 8 patch grid", fontsize=12)
    paths = save_figure_pair(figure, figures_dir / "patch_assignment_schematic.png")
    plt.close(figure)
    return paths


def count_parameters(model, config):
    instance, _ = EXPERIMENT_REGISTRY[model].build_model(Namespace(**config))
    return sum(parameter.numel() for parameter in instance.parameters() if parameter.requires_grad)


def build_fusion_tables(rows, configs, seeds):
    per_seed = selected_rows(rows, FUSION_MODELS, seeds)
    summary = summarise(per_seed, FUSION_MODELS)
    for item in summary:
        count = count_parameters(item["model"], configs[(item["model"], seeds[0])])
        item["parameter_count"] = count
        item["parameter_count_millions"] = count / 1_000_000.0
        item["model_group"] = "Single encoder" if item["model"] in FUSION_MODELS[:3] else "Dual encoder fusion"
    return per_seed, summary


def plot_fusion_capacity(per_seed, summary, figures_dir, seeds):
    setup_paper_plot_style()
    figure, (accuracy_axis, parameter_axis) = plt.subplots(
        2, 1, figsize=PAPER_TALL_FIGSIZE, sharex=True,
        gridspec_kw={"height_ratios": [1.5, 0.8]},
    )
    offsets = np.linspace(-0.15, 0.15, len(seeds))
    lookup = {row["model"]: row for row in summary}
    all_values = []

    for index, model in enumerate(FUSION_MODELS):
        values = np.array([100.0 * row["test_acc"] for row in model_rows(per_seed, model, seeds)])
        aggregate = lookup[model]
        color = model_color(model, index)
        all_values.extend(values.tolist())
        accuracy_axis.scatter(index + offsets, values, s=23, color=color, alpha=PAPER_POINT_ALPHA, edgecolors="none", zorder=2)
        accuracy_axis.errorbar(
            index, 100.0 * aggregate["mean_test_acc"],
            yerr=100.0 * aggregate["ci95_half_width_test_acc"], fmt="D", markersize=6,
            color=color, markeredgecolor=PAPER_TEXT_COLOR, markeredgewidth=0.7,
            ecolor=PAPER_TEXT_COLOR, elinewidth=1.4, capsize=4, zorder=3,
        )
        parameter_value = aggregate["parameter_count_millions"]
        parameter_axis.bar(index, parameter_value, color=color, alpha=0.82, width=0.62)
        parameter_axis.text(index, parameter_value + 0.025, f"{parameter_value:.2f}M", ha="center", va="bottom", fontsize=7.5, color=PAPER_TEXT_COLOR)

    accuracy_axis.set_ylabel("Selected-checkpoint test accuracy (%)")
    accuracy_axis.set_title("(a) Performance across seeds", pad=9)
    accuracy_axis.set_ylim(np.floor(min(all_values) - 0.5), np.ceil(max(all_values) + 0.5))
    clean_axis(accuracy_axis, grid_axis="y")
    accuracy_axis.legend(
        handles=[
            Line2D([0], [0], marker="o", linestyle="", color=PAPER_MUTED_COLOR, alpha=PAPER_POINT_ALPHA, label="Individual seed"),
            Line2D([0], [0], marker="D", linestyle="-", color=PAPER_TEXT_COLOR, label="Mean ± 95% CI"),
        ],
        loc="best",
        frameon=False,
    )

    parameter_axis.set_ylabel("Trainable parameters (millions)")
    parameter_axis.set_title("(b) Model capacity", pad=9)
    parameter_axis.set_ylim(0.0, max(row["parameter_count_millions"] for row in summary) + 0.25)
    parameter_axis.set_xticks(
        range(len(FUSION_MODELS)),
        [get_model_label(model) for model in FUSION_MODELS],
        rotation=26,
        ha="right",
    )
    clean_axis(parameter_axis, grid_axis="y")
    for axis in (accuracy_axis, parameter_axis):
        axis.axvline(2.5, color=PAPER_GRID_COLOR, linewidth=1.1)
    figure.suptitle("Row/column fusion comparison with capacity context", fontsize=12, y=1.01)
    paths = save_figure_pair(figure, figures_dir / "fusion_capacity_comparison.png")
    plt.close(figure)
    return paths


def write_captions(path, seeds, signature):
    seed_text = ", ".join(str(seed) for seed in seeds)
    protocol = (
        f"All results use training seeds {seed_text}, split seed {signature['split_seed']}, "
        f"and checkpoints selected by {signature['early_stopping_metric']}; the held-out test "
        "set is evaluated once after checkpoint selection."
    )
    content = f"""# Draft figure captions

## Epoch-based comparison figures

Multi-seed validation accuracy and validation loss across training epochs. Distinct colours and line patterns identify models; line markers are intentionally omitted because they do not encode an additional quantity. Lines show the mean and shaded bands show one sample standard deviation. For each model, curves stop at the last epoch for which all five runs still contain observations; later epochs are not averaged over a shrinking seed subset. These figures describe optimisation and validation behaviour, not held-out test performance. {protocol}

## Basic positional encoding comparison

Selected-checkpoint CIFAR-10 test accuracy for reference, axis-specific, and two-axis positional encoding conditions. Translucent circles show individual runs; diamonds and error bars show the mean and 95% t confidence interval. Marker shape distinguishes raw runs from their summary and does not encode significance or checkpoint choice. {protocol}

## Paired effect of wavelength shifting

Seed-matched change in selected-checkpoint CIFAR-10 test accuracy after applying the shifted construction to the additive and multiplicative encodings. Positive values favour the shifted variant; the horizontal line marks zero change. Circles show paired seed differences; diamonds and error bars show the mean and 95% t confidence interval. {protocol}

## Patch-to-position assignment

Seed-matched change in selected-checkpoint CIFAR-10 test accuracy relative to row-major assignment. Each colour denotes one positional-encoding family; circles show paired seed differences and diamonds show the mean with a 95% t confidence interval. The comparison changes the mapping between physical patches and sequence-indexed positional vectors; it should not be described as a limitation of global self-attention under token permutation alone. {protocol}

## Patch-assignment schematic

The four implemented mappings from physical locations on an 8 × 8 patch grid to token sequence positions: row-major, column-major, serpentine rows, and serpentine columns. Cell values denote sequence positions.

## Fusion and capacity

Selected-checkpoint CIFAR-10 test accuracy and trainable parameter count for single-encoder references and dual-encoder row/column fusion models. Translucent circles show individual runs; diamonds and error bars show the mean and 95% t confidence interval. Fusion models are not parameter matched to the single-encoder references, so performance differences cannot be attributed to the fusion operator alone. {protocol}
"""
    path.write_text(content, encoding="utf-8")


def write_manifest(path, args, source_count, signature, figures, artifacts):
    payload = {
        "source_experiment": args.experiment_name,
        "source_summary_count": source_count,
        "selected_test_protocol": "selected_checkpoint_only",
        "seeds": list(args.seeds),
        "shared_config": signature,
        "figure_groups": {
            "basic_pe": list(BASIC_MODELS),
            "shift": [
                {"label": label, "unshifted": unshifted, "shifted": shifted}
                for label, unshifted, shifted in SHIFT_CONTRASTS
            ],
            "patch_assignment": {
                "families": [label for label, _ in PATCH_FAMILIES],
                "orders": [order for order, _ in PATCH_ORDERS],
                "reference_order": "normal_row",
            },
            "fusion": list(FUSION_MODELS),
        },
        "figures": {key: {name: str(value) for name, value in paths.items()} for key, paths in figures.items()},
        "data_artifacts": {key: str(value) for key, value in artifacts.items()},
        "notes": [
            "All plotted test metrics come from summary['selected_model'].",
            "Epoch curves contain validation metrics only; no per-epoch test metric is plotted.",
            "Each epoch curve stops where fewer than all requested seeds remain.",
            "Selected-test figures show individual seeds and 95% t confidence intervals; epoch bands show one sample standard deviation.",
            "Epoch curves use colour and line pattern without point markers; summary plots use circles for runs and diamonds for means.",
            "The patch-assignment result plot uses paired deltas relative to row-major.",
            "Fusion results include parameter counts because the dual-encoder models are not capacity matched.",
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main():
    args = parse_args()
    metrics_dir = args.results_dir / args.experiment_name / "metrics"
    report_dir = args.results_dir / args.experiment_name / "reports" / args.report_name
    figures_dir = report_dir / "figures"
    report_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    rows, configs, signature, source_count = load_selected_results(metrics_dir, args.seeds)
    histories = load_epoch_history_rows(rows, required_models(), args.seeds)
    epoch_summary = aggregate_epoch_history(
        histories, sorted(required_models()), args.seeds
    )
    basic_per_seed = selected_rows(rows, BASIC_MODELS, args.seeds)
    basic_summary = summarise(basic_per_seed, BASIC_MODELS)
    shift_per_seed, shift_summary = build_shift_tables(rows, args.seeds)
    patch_per_seed, patch_summary = build_patch_tables(rows, args.seeds)
    fusion_per_seed, fusion_summary = build_fusion_tables(rows, configs, args.seeds)

    artifacts = {
        "basic_per_seed": report_dir / "basic_pe_per_seed.csv",
        "basic_summary": report_dir / "basic_pe_summary.csv",
        "shift_per_seed": report_dir / "shift_paired_deltas.csv",
        "shift_summary": report_dir / "shift_summary.csv",
        "patch_per_seed": report_dir / "patch_assignment_per_seed.csv",
        "patch_summary": report_dir / "patch_assignment_summary.csv",
        "fusion_per_seed": report_dir / "fusion_per_seed.csv",
        "fusion_summary": report_dir / "fusion_summary.csv",
        "epoch_summary": report_dir / "epoch_curve_summary.csv",
    }
    for key, content in (
        ("basic_per_seed", basic_per_seed),
        ("basic_summary", basic_summary),
        ("shift_per_seed", shift_per_seed),
        ("shift_summary", shift_summary),
        ("patch_per_seed", patch_per_seed),
        ("patch_summary", patch_summary),
        ("fusion_per_seed", fusion_per_seed),
        ("fusion_summary", fusion_summary),
        ("epoch_summary", epoch_summary),
    ):
        write_rows(artifacts[key], content)

    figures = {
        "basic_training_dynamics": plot_epoch_pair(
            epoch_summary,
            BASIC_MODELS,
            figures_dir,
            "basic_pe_validation_dynamics",
            "Basic positional encoding: validation dynamics",
        ),
        "shift_training_dynamics": plot_epoch_pair(
            epoch_summary,
            tuple(model for _, pair_a, pair_b in SHIFT_CONTRASTS for model in (pair_a, pair_b)),
            figures_dir,
            "shift_validation_dynamics",
            "Shifted and unshifted encodings: validation dynamics",
        ),
        "patch_accuracy_dynamics": plot_patch_epoch_facets(
            epoch_summary, figures_dir, "val_acc"
        ),
        "patch_loss_dynamics": plot_patch_epoch_facets(
            epoch_summary, figures_dir, "val_loss"
        ),
        "fusion_training_dynamics": plot_epoch_pair(
            epoch_summary,
            ("vit_learnable_position",) + FUSION_MODELS[3:],
            figures_dir,
            "fusion_validation_dynamics",
            "Row/column fusion: validation dynamics",
        ),
        "basic_pe": plot_basic_pe(rows, figures_dir, args.seeds),
        "shift": plot_shift_effect(shift_per_seed, shift_summary, figures_dir, args.seeds),
        "patch_assignment": plot_patch_deltas(patch_per_seed, patch_summary, figures_dir, args.seeds),
        "patch_schematic": plot_patch_schematic(figures_dir),
        "fusion_capacity": plot_fusion_capacity(fusion_per_seed, fusion_summary, figures_dir, args.seeds),
    }

    captions_path = report_dir / "figure_captions.md"
    manifest_path = report_dir / "thesis_figures_manifest.json"
    write_captions(captions_path, args.seeds, signature)
    artifacts["figure_captions"] = captions_path
    write_manifest(manifest_path, args, source_count, signature, figures, artifacts)

    print(f"Report directory: {report_dir}")
    print(f"Source summaries checked: {source_count}")
    for name, paths in figures.items():
        print(f"Figure {name}: {paths['png']} | {paths['pdf']}")
    for name, path in artifacts.items():
        print(f"Artifact {name}: {path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
