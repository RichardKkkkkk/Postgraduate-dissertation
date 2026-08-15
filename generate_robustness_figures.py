import argparse
import csv
import json
import os
import statistics
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("results/matplotlib_cache")))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from paper_plotting import (
    PAPER_FIGSIZE,
    PAPER_GRID_COLOR,
    PAPER_MUTED_COLOR,
    PAPER_POINT_ALPHA,
    PAPER_THREE_PANEL_FIGSIZE,
    PAPER_TEXT_COLOR,
    get_model_label,
    get_model_style,
    save_figure_pair,
    setup_paper_plot_style,
)


DEFAULT_SEEDS = (42, 43, 44, 45, 46)
T_CRITICAL_95_DF4 = 2.7764451051977987

LOW_DATA_EXPERIMENTS = {
    1000: "cifar10_low_data_1k_4models_5seeds",
    5000: "cifar10_low_data_5k_4models_5seeds",
    10000: "cifar10_low_data_10k_4models_5seeds",
}

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

SHARED_CONFIG_FIELDS = (
    "dataset",
    "epochs",
    "batch_size",
    "lr",
    "weight_decay",
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
        description=(
            "Generate thesis-facing low-data and CIFAR-100 figures from completed "
            "selected-checkpoint multi-seed experiments."
        )
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--report-name", default="thesis_robustness_figures_v2")
    parser.add_argument(
        "--cifar100-experiment", default="cifar100_4models_5seeds"
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS)
    )
    return parser.parse_args()


def sample_sd(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


def ci95_half_width(values):
    if len(values) != 5:
        raise ValueError(
            "This thesis report currently defines a 95% t interval only for five seeds."
        )
    return T_CRITICAL_95_DF4 * sample_sd(values) / np.sqrt(len(values))


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def config_signature(config):
    return {field: config.get(field) for field in SHARED_CONFIG_FIELDS}


def load_experiment(results_dir, experiment_name, models, seeds):
    metrics_dir = results_dir / experiment_name / "metrics"
    paths = sorted(metrics_dir.glob("*/*_summary.json"))
    if not paths:
        raise FileNotFoundError(f"No summary JSON files found under {metrics_dir}")

    expected_models = set(models)
    expected_seeds = set(seeds)
    rows = []
    configs = {}

    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        config = payload["config"]
        model = config["model"]
        seed = int(config["seed"])
        if model not in expected_models or seed not in expected_seeds:
            continue
        key = (model, seed)
        if key in configs:
            raise ValueError(
                f"Duplicate summary for {experiment_name}, {model}, seed {seed}"
            )
        selected = payload["selected_model"]
        protocol = payload.get("test_evaluation_protocol")
        if protocol != "selected_checkpoint_only":
            raise ValueError(f"Unexpected test protocol in {path}: {protocol}")
        metrics_path = path.with_name(path.name.replace("_summary.json", "_metrics.csv"))
        if not metrics_path.exists():
            raise FileNotFoundError(f"Missing per-epoch metrics CSV: {metrics_path}")
        configs[key] = config
        rows.append(
            {
                "experiment": experiment_name,
                "dataset": config["dataset"],
                "train_subset": config.get("train_subset"),
                "model": model,
                "model_label": get_model_label(model),
                "seed": seed,
                "selected_epoch": int(selected["epoch"]),
                "test_acc": float(selected["test_acc"]),
                "test_loss": float(selected["test_loss"]),
                "test_macro_f1": float(selected["test_macro_f1"]),
                "protocol": protocol,
                "summary_path": str(path),
                "metrics_path": str(metrics_path),
            }
        )

    for model in models:
        observed = {row["seed"] for row in rows if row["model"] == model}
        if observed != expected_seeds:
            raise ValueError(
                f"{experiment_name}: {model} has seeds {sorted(observed)}, "
                f"expected {sorted(expected_seeds)}"
            )

    signatures = {
        json.dumps(config_signature(config), sort_keys=True)
        for config in configs.values()
    }
    if len(signatures) != 1:
        raise ValueError(f"Non-uniform configuration within {experiment_name}")
    signature = config_signature(next(iter(configs.values())))
    return rows, signature, len(paths)


def validate_low_data_signatures(signatures):
    reference_size = min(signatures)
    reference = signatures[reference_size]
    for train_size, signature in signatures.items():
        if signature != reference:
            raise ValueError(
                f"Low-data config mismatch between {reference_size} and {train_size}: "
                f"{reference} != {signature}"
            )
    if reference["dataset"] != "cifar10":
        raise ValueError(f"Expected CIFAR-10 low-data runs, found {reference['dataset']}")
    return reference


def summarise(rows, group_fields):
    grouped = defaultdict(list)
    for row in rows:
        key = tuple(row[field] for field in group_fields)
        grouped[key].append(row)

    summaries = []
    for key, group in grouped.items():
        group = sorted(group, key=lambda row: row["seed"])
        acc = [100.0 * row["test_acc"] for row in group]
        loss = [row["test_loss"] for row in group]
        f1 = [100.0 * row["test_macro_f1"] for row in group]
        item = {field: value for field, value in zip(group_fields, key)}
        item.update(
            {
                "model_label": group[0]["model_label"],
                "num_seeds": len(group),
                "mean_test_acc_pct": statistics.mean(acc),
                "sd_test_acc_pp": sample_sd(acc),
                "ci95_half_width_test_acc_pp": ci95_half_width(acc),
                "mean_test_loss": statistics.mean(loss),
                "sd_test_loss": sample_sd(loss),
                "ci95_half_width_test_loss": ci95_half_width(loss),
                "mean_test_macro_f1_pct": statistics.mean(f1),
                "sd_test_macro_f1_pp": sample_sd(f1),
                "ci95_half_width_test_macro_f1_pp": ci95_half_width(f1),
                "mean_selected_epoch": statistics.mean(
                    row["selected_epoch"] for row in group
                ),
            }
        )
        summaries.append(item)
    return summaries


def export_per_seed_rows(rows, include_train_size=False):
    output = []
    for row in rows:
        item = {
            "experiment": row["experiment"],
            "dataset": row["dataset"],
        }
        if include_train_size:
            item["train_size"] = row["train_size"]
        item.update(
            {
                "model": row["model"],
                "model_label": row["model_label"],
                "seed": row["seed"],
                "selected_epoch": row["selected_epoch"],
                "test_acc": row["test_acc"],
                "test_acc_pct": 100.0 * row["test_acc"],
                "test_loss": row["test_loss"],
                "test_macro_f1": row["test_macro_f1"],
                "test_macro_f1_pct": 100.0 * row["test_macro_f1"],
                "protocol": row["protocol"],
                "summary_path": row["summary_path"],
            }
        )
        output.append(item)
    return output


def model_rows(rows, model, train_size=None):
    selected = [row for row in rows if row["model"] == model]
    if train_size is not None:
        selected = [row for row in selected if row["train_size"] == train_size]
    return sorted(selected, key=lambda row: row["seed"])


def summary_row(summaries, model, train_size=None):
    matches = [row for row in summaries if row["model"] == model]
    if train_size is not None:
        matches = [row for row in matches if row["train_size"] == train_size]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one summary for model={model}, train_size={train_size}; "
            f"found {len(matches)}"
        )
    return matches[0]


def clean_axis(axis, grid_axis="y"):
    axis.grid(
        True,
        axis=grid_axis,
        linestyle="--",
        linewidth=0.7,
        color=PAPER_GRID_COLOR,
        alpha=0.9,
    )
    axis.grid(False, axis="x" if grid_axis == "y" else "y")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def build_low_data_paired_rows(rows):
    paired = []
    for train_size in sorted(LOW_DATA_EXPERIMENTS):
        learned = {
            row["seed"]: row
            for row in model_rows(rows, "vit_learnable_position", train_size)
        }
        fixed = {
            row["seed"]: row
            for row in model_rows(
                rows, "vit_multiplicative_sinusoidal_shifted", train_size
            )
        }
        for seed in DEFAULT_SEEDS:
            paired.append(
                {
                    "train_size": train_size,
                    "seed": seed,
                    "learnable_test_acc_pct": 100.0 * learned[seed]["test_acc"],
                    "multi_shift_test_acc_pct": 100.0 * fixed[seed]["test_acc"],
                    "learnable_minus_multi_shift_pp": 100.0
                    * (learned[seed]["test_acc"] - fixed[seed]["test_acc"]),
                }
            )
    return paired


def summarise_paired_rows(rows):
    summaries = []
    for train_size in sorted(LOW_DATA_EXPERIMENTS):
        values = [
            row["learnable_minus_multi_shift_pp"]
            for row in rows
            if row["train_size"] == train_size
        ]
        summaries.append(
            {
                "train_size": train_size,
                "num_seeds": len(values),
                "mean_learnable_minus_multi_shift_pp": statistics.mean(values),
                "sd_learnable_minus_multi_shift_pp": sample_sd(values),
                "ci95_half_width_pp": ci95_half_width(values),
                "positive_seed_count": sum(value > 0 for value in values),
                "negative_seed_count": sum(value < 0 for value in values),
            }
        )
    return summaries


def plot_cifar100_metric(rows, summaries, figures_dir, metric):
    setup_paper_plot_style()
    figure, axis = plt.subplots(figsize=PAPER_FIGSIZE)
    seed_offsets = np.linspace(-0.15, 0.15, 5)

    if metric == "test_acc":
        raw_key = "test_acc"
        mean_key = "mean_test_acc_pct"
        ci_key = "ci95_half_width_test_acc_pp"
        scale = 100.0
        ylabel = "Selected-checkpoint test accuracy (%)"
        title = "CIFAR-100 positional-encoding comparison"
        stem = "cifar100_test_accuracy"
    elif metric == "test_loss":
        raw_key = "test_loss"
        mean_key = "mean_test_loss"
        ci_key = "ci95_half_width_test_loss"
        scale = 1.0
        ylabel = "Selected-checkpoint test loss"
        title = "CIFAR-100 test-loss comparison"
        stem = "cifar100_test_loss"
    else:
        raise ValueError(f"Unsupported CIFAR-100 metric: {metric}")

    all_values = []
    for index, model in enumerate(CIFAR100_MODELS):
        style = get_model_style(model, index)
        group = model_rows(rows, model)
        values = np.array([scale * row[raw_key] for row in group])
        aggregate = summary_row(summaries, model)
        all_values.extend(values.tolist())
        axis.scatter(
            index + seed_offsets,
            values,
            s=30,
            color=style["color"],
            alpha=PAPER_POINT_ALPHA,
            edgecolors="none",
            zorder=2,
        )
        axis.errorbar(
            index,
            aggregate[mean_key],
            yerr=aggregate[ci_key],
            fmt="D",
            markersize=7,
            color=style["color"],
            markeredgecolor=PAPER_TEXT_COLOR,
            markeredgewidth=0.8,
            ecolor=PAPER_TEXT_COLOR,
            elinewidth=1.5,
            capsize=5,
            zorder=3,
        )

    axis.set_xticks(
        range(len(CIFAR100_MODELS)),
        [get_model_label(model) for model in CIFAR100_MODELS],
        rotation=20,
        ha="right",
    )
    axis.set_ylabel(ylabel)
    axis.set_title(title, pad=10)
    clean_axis(axis, grid_axis="y")
    padding = 0.7 if metric == "test_acc" else 0.15
    axis.set_ylim(min(all_values) - padding, max(all_values) + padding)
    axis.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                color=PAPER_MUTED_COLOR,
                alpha=PAPER_POINT_ALPHA,
                label="Individual seed",
            ),
            Line2D(
                [0],
                [0],
                marker="D",
                linestyle="-",
                color=PAPER_TEXT_COLOR,
                label="Mean ± 95% CI",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.25),
        ncol=2,
        frameon=False,
    )
    paths = save_figure_pair(figure, figures_dir / f"{stem}.png")
    plt.close(figure)
    return paths


def load_epoch_rows(rows, condition):
    histories = []
    for row in rows:
        with Path(row["metrics_path"]).open(newline="", encoding="utf-8") as handle:
            for raw in csv.DictReader(handle):
                epoch = int(float(raw["epoch"]))
                for metric in EPOCH_METRICS:
                    if raw.get(metric) in (None, ""):
                        continue
                    histories.append(
                        {
                            "condition": condition,
                            "model": row["model"],
                            "model_label": row["model_label"],
                            "seed": row["seed"],
                            "epoch": epoch,
                            "metric": metric,
                            "value": float(raw[metric]),
                            "metrics_path": row["metrics_path"],
                        }
                    )
    return histories


def aggregate_epoch_rows(histories, models, seeds):
    output = []
    condition = histories[0]["condition"]
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
                if len(values) != len(seeds):
                    continue
                output.append(
                    {
                        "condition": condition,
                        "model": model,
                        "model_label": get_model_label(model),
                        "metric": metric,
                        "epoch": epoch,
                        "num_seeds": len(values),
                        "mean": statistics.mean(values),
                        "sd": sample_sd(values),
                        "min": min(values),
                        "max": max(values),
                    }
                )
    return output


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
    return max(0.0, lower - padding), min(100.0, upper + padding) if metric == "val_acc" else upper + padding


def draw_epoch_metric(axis, aggregated, models, metric, title):
    for model_index, model in enumerate(models):
        style = get_model_style(model, model_index)
        curve = sorted(
            [
                row
                for row in aggregated
                if row["model"] == model and row["metric"] == metric
            ],
            key=lambda row: row["epoch"],
        )
        if not curve:
            raise ValueError(f"No common-seed epoch curve for {model}, {metric}")
        epochs = np.array([row["epoch"] for row in curve])
        scale = 100.0 if metric == "val_acc" else 1.0
        means = scale * np.array([row["mean"] for row in curve])
        deviations = scale * np.array([row["sd"] for row in curve])
        axis.plot(
            epochs,
            means,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=2.0,
            label=get_model_label(model),
        )
        axis.fill_between(
            epochs,
            means - deviations,
            means + deviations,
            color=style["color"],
            alpha=0.10,
            linewidth=0,
        )
    axis.set_xlabel("Epoch")
    axis.set_ylabel(
        "Validation accuracy (%)" if metric == "val_acc" else "Validation loss"
    )
    axis.set_title(title, pad=8)
    clean_axis(axis)
    axis.set_ylim(*epoch_axis_limits(aggregated, models, metric))


def plot_low_data_epoch_facets(epoch_rows, models, figures_dir, metric):
    setup_paper_plot_style()
    figure, axes = plt.subplots(1, 3, figsize=PAPER_THREE_PANEL_FIGSIZE, sharey=True)
    for axis, train_size in zip(axes, sorted(LOW_DATA_EXPERIMENTS)):
        condition = f"cifar10_train_{train_size}"
        subset = [row for row in epoch_rows if row["condition"] == condition]
        draw_epoch_metric(axis, subset, models, metric, f"{train_size // 1000}k training examples")
    shared_limits = epoch_axis_limits(
        [row for row in epoch_rows if row["condition"].startswith("cifar10_train_")],
        models,
        metric,
    )
    for index, axis in enumerate(axes):
        axis.set_ylim(*shared_limits)
        if index > 0:
            axis.set_ylabel("")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=2,
        frameon=False,
    )
    metric_label = "validation accuracy" if metric == "val_acc" else "validation loss"
    figure.suptitle(f"CIFAR-10 reduced-data {metric_label} across epochs", fontsize=12, y=1.01)
    stem = "low_data_validation_accuracy_epoch" if metric == "val_acc" else "low_data_validation_loss_epoch"
    paths = save_figure_pair(figure, figures_dir / f"{stem}.png")
    plt.close(figure)
    return paths


def plot_single_epoch_metric(aggregated, models, figures_dir, metric, stem, title):
    setup_paper_plot_style()
    figure, axis = plt.subplots(figsize=PAPER_FIGSIZE)
    draw_epoch_metric(axis, aggregated, models, metric, title)
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=2,
        frameon=False,
    )
    paths = save_figure_pair(figure, figures_dir / f"{stem}.png")
    plt.close(figure)
    return paths


def write_captions(path, low_signature, cifar100_signature):
    content = f"""# Draft figure captions

## Reduced-data validation accuracy

Five-seed mean CIFAR-10 validation accuracy across epochs for the 1,000-, 5,000- and 10,000-example training conditions. Shaded regions show one sample standard deviation. Distinct colours and stable line patterns identify models; point markers are omitted because they do not encode another variable. Curves stop at the final epoch available for all five seeds. All reduced-data runs use split seed {low_signature['split_seed']}, learning rate {low_signature['lr']} and batch size {low_signature['batch_size']}.

## Reduced-data validation loss

Five-seed mean CIFAR-10 validation loss across epochs for the same reduced-data conditions. Shaded regions show one sample standard deviation, and curves use the same model colours and line patterns as the validation-accuracy figure.

## Reduced-data selected-test tables

Final reduced-data conclusions should use the validation-selected test summaries stored with this report rather than reading values from the epoch curves. The source CSV files include individual test outcomes, means, sample standard deviations, 95% t confidence intervals and the paired learnable-minus-shifted-multiplicative contrast. The existing full-data CIFAR-10 experiment uses a different learning rate and is not treated as a fourth point in a controlled data-size curve.

## CIFAR-100 validation accuracy and loss

Five-seed CIFAR-100 validation accuracy and validation loss across training epochs. Lines show the mean, shaded regions show one sample standard deviation, and curves stop at the last epoch available for all five seeds. Colours and line patterns identify models; no triangle, square or diamond marker is used on the epoch curves. The held-out test set is evaluated only after loading the validation-selected checkpoint.

## CIFAR-100 selected-test comparison

Selected-checkpoint CIFAR-100 test performance for the four pre-selected positional-encoding conditions. The categorical x-axis identifies the model and the numerical y-axis reports accuracy or loss. Faint circles show seeds 42--46; diamonds and error bars show the mean and 95% t confidence interval. These marker shapes distinguish raw observations from their summary and do not denote significance, checkpoint choice or model complexity. All runs use split seed {cifar100_signature['split_seed']}, learning rate {cifar100_signature['lr']}, batch size {cifar100_signature['batch_size']} and validation-accuracy checkpoint selection.
"""
    path.write_text(content, encoding="utf-8")


def main():
    args = parse_args()
    if tuple(args.seeds) != DEFAULT_SEEDS:
        raise ValueError(
            f"This frozen report expects seeds {list(DEFAULT_SEEDS)}, received {args.seeds}"
        )

    report_dir = args.results_dir / "reports" / args.report_name
    figures_dir = report_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    low_rows = []
    low_signatures = {}
    source_counts = {}
    epoch_summaries = []

    for train_size, experiment_name in LOW_DATA_EXPERIMENTS.items():
        rows, signature, source_count = load_experiment(
            args.results_dir,
            experiment_name,
            LOW_DATA_MODELS,
            args.seeds,
        )
        observed_subsets = {row["train_subset"] for row in rows}
        if observed_subsets != {train_size}:
            raise ValueError(
                f"{experiment_name} contains train subsets {observed_subsets}; "
                f"expected {train_size}"
            )
        for row in rows:
            row["train_size"] = train_size
        low_rows.extend(rows)
        low_signatures[train_size] = signature
        source_counts[experiment_name] = source_count

        condition = f"cifar10_train_{train_size}"
        histories = load_epoch_rows(rows, condition)
        aggregated = aggregate_epoch_rows(histories, LOW_DATA_MODELS, args.seeds)
        epoch_summaries.extend(aggregated)

    low_signature = validate_low_data_signatures(low_signatures)
    low_summaries = summarise(low_rows, ("train_size", "model"))
    low_paired = build_low_data_paired_rows(low_rows)
    low_paired_summary = summarise_paired_rows(low_paired)

    cifar100_rows, cifar100_signature, cifar100_source_count = load_experiment(
        args.results_dir,
        args.cifar100_experiment,
        CIFAR100_MODELS,
        args.seeds,
    )
    if cifar100_signature["dataset"] != "cifar100":
        raise ValueError(
            f"Expected CIFAR-100 experiment, found {cifar100_signature['dataset']}"
        )
    source_counts[args.cifar100_experiment] = cifar100_source_count
    cifar100_summaries = summarise(cifar100_rows, ("model",))
    cifar100_histories = load_epoch_rows(cifar100_rows, "cifar100_full")
    cifar100_epoch_summary = aggregate_epoch_rows(
        cifar100_histories, CIFAR100_MODELS, args.seeds
    )
    epoch_summaries.extend(cifar100_epoch_summary)

    artifacts = {
        "low_data_per_seed": report_dir / "low_data_per_seed.csv",
        "low_data_summary": report_dir / "low_data_summary_with_ci.csv",
        "low_data_paired": report_dir / "low_data_learnable_vs_multi_shift_per_seed.csv",
        "low_data_paired_summary": report_dir
        / "low_data_learnable_vs_multi_shift_summary.csv",
        "cifar100_per_seed": report_dir / "cifar100_per_seed.csv",
        "cifar100_summary": report_dir / "cifar100_summary_with_ci.csv",
        "epoch_summary": report_dir / "validation_epoch_summary.csv",
        "captions": report_dir / "figure_captions.md",
    }
    write_rows(artifacts["low_data_per_seed"], export_per_seed_rows(low_rows, True))
    write_rows(
        artifacts["low_data_summary"],
        sorted(low_summaries, key=lambda row: (row["train_size"], LOW_DATA_MODELS.index(row["model"]))),
    )
    write_rows(artifacts["low_data_paired"], low_paired)
    write_rows(artifacts["low_data_paired_summary"], low_paired_summary)
    write_rows(artifacts["cifar100_per_seed"], export_per_seed_rows(cifar100_rows))
    write_rows(
        artifacts["cifar100_summary"],
        sorted(cifar100_summaries, key=lambda row: CIFAR100_MODELS.index(row["model"])),
    )
    write_rows(artifacts["epoch_summary"], epoch_summaries)
    write_captions(artifacts["captions"], low_signature, cifar100_signature)

    figures = {
        "low_data_validation_accuracy_epoch": plot_low_data_epoch_facets(
            epoch_summaries, LOW_DATA_MODELS, figures_dir, "val_acc"
        ),
        "low_data_validation_loss_epoch": plot_low_data_epoch_facets(
            epoch_summaries, LOW_DATA_MODELS, figures_dir, "val_loss"
        ),
        "cifar100_validation_accuracy_epoch": plot_single_epoch_metric(
            cifar100_epoch_summary,
            CIFAR100_MODELS,
            figures_dir,
            "val_acc",
            "cifar100_validation_accuracy_epoch",
            "CIFAR-100 validation accuracy across epochs",
        ),
        "cifar100_validation_loss_epoch": plot_single_epoch_metric(
            cifar100_epoch_summary,
            CIFAR100_MODELS,
            figures_dir,
            "val_loss",
            "cifar100_validation_loss_epoch",
            "CIFAR-100 validation loss across epochs",
        ),
        "cifar100_test_accuracy": plot_cifar100_metric(
            cifar100_rows, cifar100_summaries, figures_dir, "test_acc"
        ),
        "cifar100_test_loss": plot_cifar100_metric(
            cifar100_rows, cifar100_summaries, figures_dir, "test_loss"
        ),
    }

    manifest_path = report_dir / "robustness_figures_manifest.json"
    manifest = {
        "selected_test_protocol": "selected_checkpoint_only",
        "seeds": list(args.seeds),
        "source_summary_counts": source_counts,
        "selected_low_data_rows": len(low_rows),
        "selected_cifar100_rows": len(cifar100_rows),
        "low_data_models": list(LOW_DATA_MODELS),
        "cifar100_models": list(CIFAR100_MODELS),
        "low_data_config": low_signature,
        "cifar100_config": cifar100_signature,
        "figures": {
            name: {format_name: str(path) for format_name, path in paths.items()}
            for name, paths in figures.items()
        },
        "data_artifacts": {name: str(path) for name, path in artifacts.items()},
        "notes": [
            "Test metrics are read only from summary['selected_model'].",
            "Selected-test error bars show 95% t confidence intervals across five seeds.",
            "Validation-dynamics bands show one sample standard deviation across five seeds.",
            "Epoch curves stop when fewer than all five seeds remain after early stopping.",
            "Epoch curves use distinct colours and stable line patterns; point markers are omitted because they do not encode another variable.",
            "Selected-test plots use faint circles for runs and diamonds for means; marker shape does not denote significance or checkpoint selection.",
            "Reduced-data selected-test and paired results remain in source CSV tables rather than a training-size-axis figure in this report version.",
            "The reduced-data suite uses lr=0.001; the existing CIFAR-10 full-data suite uses lr=0.0003 and is intentionally omitted from the connected trend.",
            "CIFAR-100 is plotted as a within-dataset controlled comparison, not as a direct accuracy-scale comparison with CIFAR-10.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Report directory: {report_dir}")
    print(f"Selected low-data rows: {len(low_rows)}")
    print(f"Selected CIFAR-100 rows: {len(cifar100_rows)}")
    for name, paths in figures.items():
        print(f"Figure {name}: {paths['png']} | {paths['pdf']}")
    for name, path in artifacts.items():
        print(f"Artifact {name}: {path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
