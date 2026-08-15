import argparse
import csv
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

import generate_final_test_figures as selected_test
from paper_plotting import (
    PAPER_GRID_COLOR,
    PAPER_MUTED_COLOR,
    PAPER_TEXT_COLOR,
    get_assignment_style,
    get_model_label,
    get_model_style,
    save_figure_pair,
    setup_paper_plot_style,
)


DEFAULT_REPORT = "thesis_final_evidence_figures_v1"
CORE_GROUPS = (
    (
        "(a) Reference and axis-specific conditions",
        (
            "vit_baseline",
            "vit_learnable_position",
            "vit_row_sinusoidal",
            "vit_col_sinusoidal",
            "vit_radial_sinusoidal",
        ),
    ),
    (
        "(b) Combined row/column conditions",
        (
            "vit_additive_sinusoidal",
            "vit_additive_sinusoidal_shifted",
            "vit_multiplicative_sinusoidal",
            "vit_multiplicative_sinusoidal_shifted",
        ),
    ),
)
SHIFT_MODELS = (
    "vit_additive_sinusoidal",
    "vit_additive_sinusoidal_shifted",
    "vit_multiplicative_sinusoidal",
    "vit_multiplicative_sinusoidal_shifted",
)
PATCH_EPOCH_FAMILIES = (
    ("Learnable PE", "learnable_position"),
    ("Row-wise PE", "row_sinusoidal"),
    ("Column-wise PE", "col_sinusoidal"),
    ("Multiplicative PE", "multiplicative_sinusoidal"),
)
FUSION_GROUPS = (
    (
        "(a) Aggregation-based fusion",
        (
            "vit_learnable_position",
            "vit_normal_col_learnable_multiplicative_sinusoidal",
            "vit_row_col_mean_fusion",
            "vit_row_col_mean_mlp_fusion",
            "vit_row_col_latent_fusion",
        ),
    ),
    (
        "(b) Cross-attention fusion",
        (
            "vit_learnable_position",
            "vit_normal_col_learnable_multiplicative_sinusoidal",
            "vit_row_col_cross_attention_fusion",
            "vit_row_col_cross_attention_mlp_head_fusion",
        ),
    ),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate thesis validation-epoch figures, selected-test tables, "
            "and test-only auxiliary analyses."
        )
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--report-name", default=DEFAULT_REPORT)
    parser.add_argument(
        "--main-experiment", default=selected_test.MAIN_EXPERIMENT
    )
    parser.add_argument(
        "--cifar100-experiment", default=selected_test.CIFAR100_EXPERIMENT
    )
    return parser.parse_args()


def metrics_path(summary_path):
    path = Path(summary_path)
    suffix = "_summary.json"
    if not path.name.endswith(suffix):
        raise ValueError(f"Unexpected summary filename: {path}")
    result = path.with_name(path.name[: -len(suffix)] + "_metrics.csv")
    if not result.exists():
        raise FileNotFoundError(result)
    return result


def load_metric_series(summary_row, metric):
    values = {}
    with metrics_path(summary_row["summary_path"]).open(
        newline="", encoding="utf-8"
    ) as handle:
        reader = csv.DictReader(handle)
        if metric not in (reader.fieldnames or []):
            raise ValueError(f"Missing {metric} in {metrics_path(summary_row['summary_path'])}")
        for row in reader:
            values[int(row["epoch"])] = float(row[metric])
    if not values:
        raise ValueError(f"No epoch values for {summary_row['model']}, seed {summary_row['seed']}")
    return values


def aggregate_epoch_curves(run_specs, metric):
    """Aggregate each condition/series over its shared five-seed epoch range."""
    grouped = defaultdict(list)
    for spec in run_specs:
        grouped[(spec["condition"], spec["series"])].append(spec)

    scale = 100.0 if "acc" in metric or "recall" in metric else 1.0
    output = []
    for (condition, series), specs in grouped.items():
        observed_seeds = {spec["seed"] for spec in specs}
        if observed_seeds != set(selected_test.SEEDS):
            raise ValueError(
                f"{condition}/{series}: found seeds {sorted(observed_seeds)}, "
                f"expected {list(selected_test.SEEDS)}"
            )
        by_seed = {
            spec["seed"]: load_metric_series(spec["row"], metric) for spec in specs
        }
        first_epoch = max(min(values) for values in by_seed.values())
        last_epoch = min(max(values) for values in by_seed.values())
        if first_epoch > last_epoch:
            raise ValueError(f"No shared epoch range for {condition}/{series}")
        for epoch in range(first_epoch, last_epoch + 1):
            if not all(epoch in values for values in by_seed.values()):
                raise ValueError(f"Epoch gap for {condition}/{series} at epoch {epoch}")
            values = [scale * by_seed[seed][epoch] for seed in selected_test.SEEDS]
            mean = statistics.mean(values)
            sd = statistics.stdev(values)
            ci = selected_test.T_CRITICAL_95_DF4 * sd / math.sqrt(len(values))
            output.append(
                {
                    "condition": condition,
                    "series": series,
                    "metric": metric,
                    "epoch": epoch,
                    "num_seeds": len(values),
                    "mean": mean,
                    "sample_sd": sd,
                    "ci95_half_width": ci,
                    "ci95_lower": mean - ci,
                    "ci95_upper": mean + ci,
                    "common_first_epoch": first_epoch,
                    "common_last_epoch": last_epoch,
                }
            )
    return output


def specs_for_models(rows, models, condition="default"):
    specs = []
    for model in models:
        for row in selected_test.group_model(rows, model):
            specs.append(
                {
                    "condition": str(condition),
                    "series": model,
                    "seed": row["seed"],
                    "row": row,
                }
            )
    return specs


def rows_for_curve(curves, condition, series):
    return sorted(
        [
            row
            for row in curves
            if row["condition"] == str(condition) and row["series"] == series
        ],
        key=lambda row: row["epoch"],
    )


def clean_epoch_axis(axis, metric, title):
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Validation accuracy (%)" if metric == "val_acc" else "Validation loss")
    axis.set_title(title, pad=9)
    axis.grid(True, axis="y", color=PAPER_GRID_COLOR, linestyle="--", linewidth=0.7)
    axis.grid(False, axis="x")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def pad_y_limits(axis, lower_values, upper_values, lower_bound=None):
    low = min(lower_values)
    high = max(upper_values)
    span = max(high - low, 1e-6)
    padding = max(0.08 * span, 0.25 if high > 10 else 0.015)
    low -= padding
    if lower_bound is not None:
        low = max(lower_bound, low)
    axis.set_ylim(low, high + padding)


def plot_model_curve(axis, curves, condition, model, index, metric, label=None):
    points = rows_for_curve(curves, condition, model)
    if not points:
        raise ValueError(f"No curve for {condition}/{model}/{metric}")
    style = get_model_style(model, index)
    epochs = np.array([row["epoch"] for row in points])
    means = np.array([row["mean"] for row in points])
    lower = np.array([row["ci95_lower"] for row in points])
    upper = np.array([row["ci95_upper"] for row in points])
    axis.plot(
        epochs,
        means,
        color=style["color"],
        linestyle=style["linestyle"],
        linewidth=1.8,
        label=label or get_model_label(model),
    )
    axis.fill_between(epochs, lower, upper, color=style["color"], alpha=0.14, linewidth=0)
    return lower.tolist(), upper.tolist()


def plot_core_epoch(curves, figures_dir):
    setup_paper_plot_style()
    figure, axes = plt.subplots(1, 2, figsize=(12.8, 5.0), sharey=True)
    for axis, (title, models) in zip(axes, CORE_GROUPS):
        lowers, uppers = [], []
        for index, model in enumerate(models):
            low, high = plot_model_curve(axis, curves, "core", model, index, "val_acc")
            lowers.extend(low)
            uppers.extend(high)
        clean_epoch_axis(axis, "val_acc", title)
        pad_y_limits(axis, lowers, uppers, lower_bound=0.0)
        axis.legend(frameon=False, loc="lower right")
    figure.suptitle("CIFAR-10 core positional encodings: validation trajectories", y=1.01)
    paths = save_figure_pair(
        figure, figures_dir / "core_pe_validation_accuracy_epoch.png"
    )
    plt.close(figure)
    return paths


def plot_shift_epoch(curves, figures_dir):
    setup_paper_plot_style()
    figure, axis = plt.subplots(figsize=(8.6, 5.1))
    lowers, uppers = [], []
    for index, model in enumerate(SHIFT_MODELS):
        low, high = plot_model_curve(axis, curves, "shift", model, index, "val_acc")
        lowers.extend(low)
        uppers.extend(high)
    clean_epoch_axis(axis, "val_acc", "Additive and multiplicative shifted variants")
    pad_y_limits(axis, lowers, uppers, lower_bound=0.0)
    axis.legend(frameon=False, loc="lower right", ncol=2)
    paths = save_figure_pair(
        figure, figures_dir / "shifted_pe_validation_accuracy_epoch.png"
    )
    plt.close(figure)
    return paths


def plot_patch_epoch(curves, figures_dir):
    setup_paper_plot_style()
    figure, axes = plt.subplots(2, 2, figsize=(12.2, 8.0), sharey=True)
    all_lowers, all_uppers = [], []
    for axis, (family, suffix) in zip(axes.flat, PATCH_EPOCH_FAMILIES):
        for order, _ in selected_test.PATCH_ORDERS:
            model = selected_test.patch_model_name(order, suffix)
            points = rows_for_curve(curves, family, model)
            style = get_assignment_style(order)
            epochs = np.array([row["epoch"] for row in points])
            means = np.array([row["mean"] for row in points])
            lower = np.array([row["ci95_lower"] for row in points])
            upper = np.array([row["ci95_upper"] for row in points])
            axis.plot(
                epochs,
                means,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=1.8,
                label=style["label"],
            )
            axis.fill_between(
                epochs, lower, upper, color=style["color"], alpha=0.14, linewidth=0
            )
            all_lowers.extend(lower.tolist())
            all_uppers.extend(upper.tolist())
        clean_epoch_axis(axis, "val_acc", family)
    shared_limits = (
        max(0.0, min(all_lowers) - 0.08 * (max(all_uppers) - min(all_lowers))),
        max(all_uppers) + 0.08 * (max(all_uppers) - min(all_lowers)),
    )
    for axis in axes.flat:
        axis.set_ylim(*shared_limits)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.01)
    )
    figure.suptitle("Patch-to-position assignment: validation trajectories", y=1.01)
    figure.subplots_adjust(bottom=0.12, hspace=0.28, wspace=0.16)
    paths = save_figure_pair(
        figure, figures_dir / "patch_assignment_validation_accuracy_epoch.png"
    )
    plt.close(figure)
    return paths


def plot_low_data_epoch(curves, figures_dir):
    setup_paper_plot_style()
    figure, axes = plt.subplots(2, 2, figsize=(12.2, 8.0), sharey=True)
    sizes = ((1000, "1k"), (5000, "5k"), (10000, "10k"), (45000, "Full (45k)"))
    all_lowers, all_uppers = [], []
    for axis, (size, label) in zip(axes.flat, sizes):
        for index, model in enumerate(selected_test.LOW_DATA_MODELS):
            low, high = plot_model_curve(
                axis, curves, size, model, index, "val_acc"
            )
            all_lowers.extend(low)
            all_uppers.extend(high)
        clean_epoch_axis(axis, "val_acc", f"{label} training examples")
    span = max(all_uppers) - min(all_lowers)
    shared_limits = (
        max(0.0, min(all_lowers) - 0.08 * span),
        max(all_uppers) + 0.08 * span,
    )
    for axis in axes.flat:
        axis.set_ylim(*shared_limits)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.01)
    )
    figure.suptitle("CIFAR-10 training-set size: validation trajectories", y=1.01)
    figure.subplots_adjust(bottom=0.14, hspace=0.28, wspace=0.16)
    paths = save_figure_pair(
        figure, figures_dir / "low_data_validation_accuracy_epoch.png"
    )
    plt.close(figure)
    return paths


def plot_cifar100_epoch(acc_curves, loss_curves, figures_dir):
    setup_paper_plot_style()
    figure, axes = plt.subplots(1, 2, figsize=(12.2, 5.0))
    for axis, curves, metric, title in (
        (axes[0], acc_curves, "val_acc", "(a) Validation accuracy"),
        (axes[1], loss_curves, "val_loss", "(b) Validation loss"),
    ):
        lowers, uppers = [], []
        for index, model in enumerate(selected_test.CIFAR100_MODELS):
            low, high = plot_model_curve(axis, curves, "cifar100", model, index, metric)
            lowers.extend(low)
            uppers.extend(high)
        clean_epoch_axis(axis, metric, title)
        pad_y_limits(axis, lowers, uppers, lower_bound=0.0)
        axis.legend(frameon=False, loc="best")
    figure.suptitle("CIFAR-100 validation trajectories", y=1.01)
    paths = save_figure_pair(
        figure, figures_dir / "cifar100_validation_accuracy_loss_epoch.png"
    )
    plt.close(figure)
    return paths


def plot_fusion_epoch(curves, figures_dir):
    setup_paper_plot_style()
    figure, axes = plt.subplots(1, 2, figsize=(13.0, 5.0), sharey=True)
    all_lowers, all_uppers = [], []
    for axis, (title, models) in zip(axes, FUSION_GROUPS):
        for index, model in enumerate(models):
            low, high = plot_model_curve(axis, curves, "fusion", model, index, "val_acc")
            all_lowers.extend(low)
            all_uppers.extend(high)
        clean_epoch_axis(axis, "val_acc", title)
        axis.legend(frameon=False, loc="lower right", fontsize=8)
    span = max(all_uppers) - min(all_lowers)
    shared_limits = (
        max(0.0, min(all_lowers) - 0.08 * span),
        max(all_uppers) + 0.08 * span,
    )
    for axis in axes:
        axis.set_ylim(*shared_limits)
    figure.suptitle("Single-branch references and dual-branch fusion: validation trajectories", y=1.01)
    paths = save_figure_pair(
        figure, figures_dir / "fusion_validation_accuracy_epoch.png"
    )
    plt.close(figure)
    return paths


def write_selected_test_table(
    report_dir,
    stem,
    title,
    summaries,
    condition_key=None,
    include_parameters=False,
):
    csv_path = report_dir / f"{stem}.csv"
    selected_test.write_csv(csv_path, summaries)
    md_path = report_dir / f"{stem}.md"
    condition_header = "| Condition " if condition_key else ""
    condition_rule = "|---" if condition_key else ""
    parameter_header = " Trainable parameters |" if include_parameters else ""
    parameter_rule = "---:|" if include_parameters else ""
    lines = [
        f"# {title}",
        "",
        "Metrics use the checkpoint selected by validation. Values are means over seeds 42--46; ± is the 95% t confidence-interval half-width (df = 4).",
        "",
        f"{condition_header}| Variant |{parameter_header} Test accuracy (%) | Test loss |",
        f"{condition_rule}|---|{parameter_rule}---:|---:|",
    ]
    for item in summaries:
        prefix = f"| {item[condition_key]} " if condition_key else ""
        parameter_value = (
            f"{item['trainable_parameters']:,} | " if include_parameters else ""
        )
        lines.append(
            f"{prefix}| {item['model_label']} | {parameter_value}"
            f"{item['mean_test_acc_pct']:.3f} ± {item['ci95_half_width_test_acc_pp']:.3f} | "
            f"{item['mean_test_loss']:.4f} ± {item['ci95_half_width_test_loss']:.4f} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"csv": csv_path, "markdown": md_path}


def plot_shift_aux(per_seed, summaries, figures_dir):
    setup_paper_plot_style()
    figure, axis = plt.subplots(figsize=(7.8, 4.7))
    offsets = np.linspace(-0.10, 0.10, len(selected_test.SEEDS))
    colors = (
        get_model_style("vit_additive_sinusoidal_shifted", 0)["color"],
        get_model_style("vit_multiplicative_sinusoidal_shifted", 1)["color"],
    )
    bounds = []
    labels = ("Shifted additive\nminus additive", "Shifted multiplicative\nminus multiplicative")
    for index, ((contrast, _, _), label, color) in enumerate(
        zip(selected_test.SHIFT_CONTRASTS, labels, colors)
    ):
        values = [
            row["paired_test_acc_difference_pp"]
            for row in per_seed
            if row["contrast"] == contrast
        ]
        axis.scatter(index + offsets, values, color=color, s=26, alpha=0.34, edgecolors="none")
        item = next(row for row in summaries if row["contrast"] == contrast)
        mean = item["mean_paired_test_acc_difference_pp"]
        ci = item["ci95_half_width_paired_test_acc_difference_pp"]
        bounds.extend((mean - ci, mean + ci, *values))
        axis.errorbar(index, mean, yerr=ci, fmt="D", color=color, capsize=5, markersize=6)
    axis.axhline(0.0, color=PAPER_MUTED_COLOR, linewidth=1.0)
    axis.set_xticks((0, 1), labels)
    axis.set_ylabel("Paired test-accuracy difference (percentage points)")
    axis.set_title("Seed-matched effects of shifted positional encodings", pad=9)
    selected_test.clean_axis(axis)
    axis.set_ylim(*selected_test.padded_limits(bounds))
    axis.legend(
        handles=[
            Line2D([0], [0], marker="o", linestyle="", color=PAPER_MUTED_COLOR, alpha=0.4, label="Seed-level difference"),
            Line2D([0], [0], marker="D", linestyle="", color=PAPER_TEXT_COLOR, label="Mean ± paired 95% CI"),
        ],
        frameon=False,
        loc="best",
    )
    paths = save_figure_pair(figure, figures_dir / "shifted_pe_paired_test_effects.png")
    plt.close(figure)
    return paths


def plot_per_class_aux(summaries, figures_dir):
    setup_paper_plot_style()
    figure, axes = plt.subplots(2, 1, figsize=(10.2, 7.0), sharex=True)
    colors = (
        get_model_style("vit_learnable_position", 0)["color"],
        get_model_style("vit_multiplicative_sinusoidal_shifted", 1)["color"],
    )
    first_contrast = selected_test.PER_CLASS_CONTRASTS[0][0]
    class_names = [
        row["class_name"] for row in summaries if row["contrast"] == first_contrast
    ]
    x = np.arange(len(class_names))
    for axis, (contrast, _, _), color in zip(
        axes, selected_test.PER_CLASS_CONTRASTS, colors
    ):
        group = [row for row in summaries if row["contrast"] == contrast]
        means = [row["mean_paired_recall_difference_pp"] for row in group]
        cis = [row["ci95_half_width_paired_recall_difference_pp"] for row in group]
        axis.errorbar(
            x, means, yerr=cis, fmt="D", color=color, capsize=3.5, markersize=5.5, linewidth=1.2
        )
        axis.axhline(0.0, color=PAPER_MUTED_COLOR, linewidth=0.9)
        axis.set_ylabel("Recall difference\n(percentage points)")
        axis.set_title(contrast, pad=7)
        selected_test.clean_axis(axis)
    axes[-1].set_xticks(x, class_names, rotation=25, ha="right")
    axes[-1].set_xlabel("CIFAR-10 class")
    figure.text(
        0.5,
        0.01,
        "Diamonds show mean paired recall differences; bars show paired 95% t confidence intervals.",
        ha="center",
        color=PAPER_MUTED_COLOR,
        fontsize=8.5,
    )
    figure.subplots_adjust(bottom=0.19, hspace=0.31)
    paths = save_figure_pair(
        figure, figures_dir / "per_class_recall_paired_differences.png"
    )
    plt.close(figure)
    return paths


def write_captions(report_dir):
    text = """# Draft figure captions

## Main validation-epoch figures

### Core positional encodings

Mean CIFAR-10 validation accuracy over seeds 42--46 for nine positional-encoding conditions. Shaded regions are pointwise 95% t confidence intervals. Each series ends at the last epoch available for all five seeds of that condition. Final comparisons use the selected-checkpoint test table rather than the validation trajectories.

### Shifted positional encodings

Mean CIFAR-10 validation accuracy for additive, shifted additive, multiplicative and shifted multiplicative positional encodings. Shaded regions show pointwise 95% t confidence intervals over five seeds.

### Patch-to-position assignment

Mean CIFAR-10 validation accuracy under four mappings between physical patches and assigned positional vectors. Each panel fixes the positional-encoding family; shaded regions show pointwise 95% t confidence intervals over five seeds.

### Training-set size

Mean CIFAR-10 validation accuracy for four positional-encoding conditions trained with 1,000, 5,000, 10,000 or 45,000 training examples. All four panels use the aligned learning-rate protocol. Within a seed, variants share a sampled subset; across seeds, subset composition and stochastic training both vary.

### CIFAR-100

Mean CIFAR-100 validation accuracy and loss for four prespecified positional-encoding conditions. Shaded regions show pointwise 95% t confidence intervals over five seeds. Dataset-level ranking is determined from the selected-checkpoint test table.

### Fusion

Mean CIFAR-10 validation accuracy for single-branch references and dual-branch fusion variants. The panels separate aggregation-based and cross-attention fusion for legibility. Shaded regions show pointwise 95% t confidence intervals over five seeds; the architectures are not parameter matched.

## Auxiliary selected-test figures

### Shifted paired effects

Seed-matched selected-test accuracy differences for shifted minus unshifted positional encodings. Faint points are the five paired differences; diamonds and error bars show the mean and paired 95% t confidence interval.

### Patch-assignment heatmap

Mean seed-matched selected-test accuracy change for each patch-to-position assignment relative to normal_row within the same positional-encoding family. Values are percentage points and the colour scale is centred at zero.

### Fusion accuracy-capacity trade-off

Selected-checkpoint CIFAR-10 test accuracy against trainable parameter count for the learnable, hybrid and five fusion variants. Fusion architectures are not parameter matched to the compact single-branch backbone.

### Per-class recall differences

Mean seed-matched CIFAR-10 per-class recall differences with paired 95% t confidence intervals. No individual-seed points are shown. The plotted metric is recall, including where historical result files used the field name per_class_accuracy.
"""
    path = report_dir / "figure_captions.md"
    path.write_text(text, encoding="utf-8")
    return path


def main():
    args = parse_args()
    report_dir = args.results_dir / "reports" / args.report_name
    figures_dir = report_dir / "figures"
    report_dir.mkdir(parents=True, exist_ok=True)

    patch_models = tuple(
        selected_test.patch_model_name(order, suffix)
        for _, suffix in selected_test.PATCH_FAMILIES
        for order, _ in selected_test.PATCH_ORDERS
    )
    main_models = tuple(
        dict.fromkeys(
            selected_test.CORE_MODELS + patch_models + selected_test.FUSION_MODELS
        )
    )
    main_rows, main_configs, main_source_count = selected_test.load_experiment(
        args.results_dir, args.main_experiment, main_models
    )
    audit = selected_test.validate_uniform_config(args.main_experiment, main_configs)

    low_rows_by_size = {}
    low_configs_by_size = {}
    low_source_counts = {}
    for size, experiment in selected_test.LOW_DATA_EXPERIMENTS.items():
        rows, configs, count = selected_test.load_experiment(
            args.results_dir, experiment, selected_test.LOW_DATA_MODELS
        )
        low_rows_by_size[size] = rows
        low_configs_by_size[size] = configs
        low_source_counts[str(size)] = count
        audit.extend(selected_test.validate_uniform_config(experiment, configs))
    audit.extend(selected_test.validate_full_low_alignment(main_configs, low_configs_by_size))

    c100_rows, c100_configs, c100_source_count = selected_test.load_experiment(
        args.results_dir, args.cifar100_experiment, selected_test.CIFAR100_MODELS
    )
    audit.extend(selected_test.validate_uniform_config(args.cifar100_experiment, c100_configs))
    selected_test.write_csv(report_dir / "configuration_alignment_audit.csv", audit)

    core_rows = [row for row in main_rows if row["model"] in selected_test.CORE_MODELS]
    core_summary = selected_test.summarise_models(
        core_rows, selected_test.CORE_MODELS, include_parameters=True
    )
    core_table = write_selected_test_table(
        report_dir,
        "table_6_core_pe_selected_test",
        "Table 6: core positional-encoding selected-test results",
        core_summary,
    )
    selected_test.write_csv(
        report_dir / "core_pe_selected_test_per_seed.csv",
        selected_test.per_seed_export(core_rows),
    )

    low_rows, low_summary = selected_test.build_low_data(main_rows, low_rows_by_size)
    for item in low_summary:
        item["training_size_display"] = item["training_size_label"]
    low_table = write_selected_test_table(
        report_dir,
        "low_data_selected_test_table",
        "Low-data CIFAR-10 selected-test results",
        low_summary,
        condition_key="training_size_display",
    )
    selected_test.write_csv(
        report_dir / "low_data_selected_test_per_seed.csv",
        selected_test.per_seed_export(low_rows),
    )

    c100_summary = selected_test.summarise_models(
        c100_rows, selected_test.CIFAR100_MODELS, include_parameters=True
    )
    c100_table = write_selected_test_table(
        report_dir,
        "cifar100_selected_test_table",
        "CIFAR-100 selected-test results",
        c100_summary,
    )
    selected_test.write_csv(
        report_dir / "cifar100_selected_test_per_seed.csv",
        selected_test.per_seed_export(c100_rows),
    )

    fusion_rows = [row for row in main_rows if row["model"] in selected_test.FUSION_MODELS]
    fusion_summary = selected_test.summarise_models(
        fusion_rows, selected_test.FUSION_MODELS, include_parameters=True
    )
    fusion_table = write_selected_test_table(
        report_dir,
        "fusion_selected_test_and_parameters_table",
        "Fusion selected-test results and capacity",
        fusion_summary,
        include_parameters=True,
    )

    core_specs = specs_for_models(core_rows, selected_test.CORE_MODELS, "core")
    core_curves = aggregate_epoch_curves(core_specs, "val_acc")
    shift_curves = aggregate_epoch_curves(
        specs_for_models(main_rows, SHIFT_MODELS, "shift"), "val_acc"
    )

    patch_specs = []
    for family, suffix in PATCH_EPOCH_FAMILIES:
        for order, _ in selected_test.PATCH_ORDERS:
            model = selected_test.patch_model_name(order, suffix)
            patch_specs.extend(specs_for_models(main_rows, (model,), family))
    patch_curves = aggregate_epoch_curves(patch_specs, "val_acc")

    low_specs = []
    for size, rows in low_rows_by_size.items():
        low_specs.extend(specs_for_models(rows, selected_test.LOW_DATA_MODELS, size))
    low_specs.extend(specs_for_models(main_rows, selected_test.LOW_DATA_MODELS, 45000))
    low_curves = aggregate_epoch_curves(low_specs, "val_acc")

    c100_specs = specs_for_models(
        c100_rows, selected_test.CIFAR100_MODELS, "cifar100"
    )
    c100_acc_curves = aggregate_epoch_curves(c100_specs, "val_acc")
    c100_loss_curves = aggregate_epoch_curves(c100_specs, "val_loss")
    fusion_curves = aggregate_epoch_curves(
        specs_for_models(main_rows, selected_test.FUSION_MODELS, "fusion"), "val_acc"
    )

    epoch_sources = {
        "core": core_curves,
        "shifted": shift_curves,
        "patch_assignment": patch_curves,
        "low_data": low_curves,
        "cifar100_accuracy": c100_acc_curves,
        "cifar100_loss": c100_loss_curves,
        "fusion": fusion_curves,
    }
    for name, rows in epoch_sources.items():
        selected_test.write_csv(report_dir / f"{name}_validation_epoch_summary.csv", rows)

    patch_per_seed, patch_summary = selected_test.build_patch_rows(main_rows)
    selected_test.write_csv(
        report_dir / "patch_assignment_test_accuracy_per_seed.csv", patch_per_seed
    )
    selected_test.write_csv(
        report_dir / "patch_assignment_test_accuracy_summary.csv", patch_summary
    )
    shift_per_seed, shift_summary = selected_test.build_shift_rows(main_rows)
    selected_test.write_csv(
        report_dir / "shifted_pe_paired_test_effects_per_seed.csv", shift_per_seed
    )
    selected_test.write_csv(
        report_dir / "shifted_pe_paired_test_effects_summary.csv", shift_summary
    )
    per_class_seed, per_class_summary = selected_test.build_per_class_rows(main_rows)
    selected_test.write_csv(
        report_dir / "per_class_recall_paired_differences_per_seed.csv", per_class_seed
    )
    selected_test.write_csv(
        report_dir / "per_class_recall_paired_differences_summary.csv", per_class_summary
    )

    figures = {
        "main_core_pe_validation_accuracy_epoch": plot_core_epoch(core_curves, figures_dir),
        "main_shifted_pe_validation_accuracy_epoch": plot_shift_epoch(shift_curves, figures_dir),
        "main_patch_assignment_validation_accuracy_epoch": plot_patch_epoch(patch_curves, figures_dir),
        "main_low_data_validation_accuracy_epoch": plot_low_data_epoch(low_curves, figures_dir),
        "main_cifar100_validation_accuracy_loss_epoch": plot_cifar100_epoch(c100_acc_curves, c100_loss_curves, figures_dir),
        "main_fusion_validation_accuracy_epoch": plot_fusion_epoch(fusion_curves, figures_dir),
        "aux_patch_assignment_test_accuracy_delta_heatmap": selected_test.plot_patch_heatmap(patch_summary, figures_dir),
        "aux_fusion_test_accuracy_vs_parameters": selected_test.plot_fusion(fusion_rows, fusion_summary, figures_dir),
        "aux_shifted_pe_paired_test_effects": plot_shift_aux(shift_per_seed, shift_summary, figures_dir),
        "aux_per_class_recall_paired_differences": plot_per_class_aux(per_class_summary, figures_dir),
    }
    captions_path = write_captions(report_dir)

    manifest = {
        "report_name": args.report_name,
        "evidence_rule": {
            "epoch_figures": "validation metrics only; training dynamics, convergence, and stability",
            "final_comparison": "selected-checkpoint test accuracy and loss tables",
            "test_over_epoch": False,
        },
        "seeds": list(selected_test.SEEDS),
        "ci_definition": "mean +/- 2.7764451051977987 * sample_sd / sqrt(5)",
        "epoch_range_rule": "Each condition is plotted only through the final epoch present for all five seeds.",
        "protocol_alignment_gate": "passed",
        "main_experiment": args.main_experiment,
        "low_data_experiments": selected_test.LOW_DATA_EXPERIMENTS,
        "cifar100_experiment": args.cifar100_experiment,
        "source_summary_counts": {
            "main_directory_total": main_source_count,
            "low_data_directory_totals": low_source_counts,
            "cifar100_directory_total": c100_source_count,
        },
        "tables": {
            "table_6_core": {key: str(value) for key, value in core_table.items()},
            "low_data": {key: str(value) for key, value in low_table.items()},
            "cifar100": {key: str(value) for key, value in c100_table.items()},
            "fusion": {key: str(value) for key, value in fusion_table.items()},
        },
        "figures": {
            name: {key: str(value) for key, value in paths.items()}
            for name, paths in figures.items()
        },
        "captions": str(captions_path),
        "notes": [
            "Validation epoch curves do not determine final model ranking.",
            "All test metrics come from summary['selected_model'] only.",
            "Paired intervals are computed from seed-level paired differences.",
            "Low-data variants share a subset within seed; across seeds both subset and training randomness vary.",
            "Fusion architectures are not parameter matched.",
            "Per-class accuracy in historical files is presented as recall.",
        ],
    }
    manifest_path = report_dir / "thesis_evidence_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Report directory: {report_dir}")
    print("Protocol alignment gate: PASSED")
    print("Main validation-epoch figures: 6")
    print("Auxiliary selected-test figures: 4")
    print("Selected-test tables: core, low-data, CIFAR-100, fusion")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
