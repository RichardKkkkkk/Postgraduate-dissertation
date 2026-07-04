import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.registry import EXPERIMENT_REGISTRY
from result_paths import build_report_artifact_dirs, resolve_run_artifact_paths


DEFAULT_MODELS = ["vit_baseline", "vit_rope", "vit_rope_2d"]
DEFAULT_METRICS = ["best_val_acc", "test_acc", "macro_f1"]
CURVE_METRICS = [
    "train_loss",
    "val_loss",
    "test_loss",
    "train_acc",
    "val_acc",
    "test_acc",
    "val_macro_f1",
    "test_macro_f1",
]
MODEL_LABELS = {
    "vit_baseline": "ViT Baseline",
    "vit_rope": "ViT RoPE",
    "vit_rope_2d": "ViT RoPE 2D",
    "resnet18_scratch": "ResNet18 Scratch",
    "resnet18_imagenet": "ResNet18 ImageNet",
}
METRIC_LABELS = {
    "best_val_acc": "Best Validation Accuracy",
    "test_acc": "Selected Test Accuracy",
    "macro_f1": "Selected Test Macro F1",
}
CURVE_METRIC_LABELS = {
    "train_loss": "Train Loss",
    "val_loss": "Validation Loss",
    "test_loss": "Test Loss",
    "train_acc": "Train Accuracy",
    "val_acc": "Validation Accuracy",
    "test_acc": "Test Accuracy",
    "val_macro_f1": "Validation Macro F1",
    "test_macro_f1": "Test Macro F1",
}
MODEL_COLORS = {
    "vit_no_pos": "#1d4ed8",
    "vit_baseline": "#ea580c",
    "vit_row_sinusoidal": "#16a34a",
    "vit_col_sinusoidal": "#dc2626",
    "vit_additive_sinusoidal": "#7c3aed",
    "vit_additive_sinusoidal_shifted": "#a16207",
    "vit_multiplicative_sinusoidal": "#db2777",
    "vit_multiplicative_sinusoidal_shifted": "#0f766e",
    "vit_rope": "#2563eb",
    "vit_rope_2d": "#0891b2",
    "resnet18_scratch": "#4b5563",
    "resnet18_imagenet": "#111827",
}
SUMMARY_METRIC_EXTRACTORS = {
    "best_val_acc": lambda summary: float(summary["best_val_acc"]),
    "test_acc": lambda summary: float(summary["selected_model"]["test_acc"]),
    "macro_f1": lambda summary: float(summary["selected_model"]["test_macro_f1"]),
    "best_epoch": lambda summary: int(summary["selected_model"]["epoch"]),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize multi-seed experiment runs into mean/std tables and plots."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(EXPERIMENT_REGISTRY.keys()),
        default=DEFAULT_MODELS,
        help="Models to summarize across seeds.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        required=True,
        help="Seeds to summarize, for example: --seeds 42 43 44",
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument(
        "--run-prefix",
        type=str,
        default=None,
        help="Prefix used by run_seed_sweep.py. Example: cifar10_main",
    )
    parser.add_argument(
        "--report-name",
        type=str,
        default=None,
        help="Output report folder name. If experiment-name is set, it is saved under results/<experiment>/reports/.",
    )
    parser.add_argument(
        "--reference-model",
        choices=tuple(EXPERIMENT_REGISTRY.keys()),
        default="vit_baseline",
        help="Reference model used for delta comparisons.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=tuple(DEFAULT_METRICS),
        default=DEFAULT_METRICS,
        help="Metrics to summarize.",
    )
    return parser.parse_args()


def build_run_name(model_name: str, seed: int, run_prefix: str | None):
    base_name = f"{model_name}_seed{seed}"
    if run_prefix:
        return f"{run_prefix}_{base_name}"
    return base_name


def make_default_report_name(args):
    seed_tag = f"seed{min(args.seeds)}_to_seed{max(args.seeds)}"
    prefix = args.run_prefix or "manual"
    return f"seed_summary_{prefix}_{seed_tag}"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_history(path: Path):
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            parsed = {}
            for key, value in row.items():
                if value is None or value == "":
                    parsed[key] = value
                    continue
                try:
                    parsed[key] = float(value)
                except ValueError:
                    parsed[key] = value
            rows.append(parsed)
    return rows


def safe_mean(values):
    return mean(values) if values else math.nan


def safe_stdev(values):
    return stdev(values) if len(values) > 1 else 0.0


def format_float(value):
    return f"{value:.4f}"


def collect_run_rows(args):
    rows = []
    histories = []
    for model_name in args.models:
        for seed in args.seeds:
            run_name = build_run_name(model_name, seed, args.run_prefix)
            artifact_paths = resolve_run_artifact_paths(
                args.results_dir,
                run_name,
                experiment_name=args.experiment_name,
            )
            summary_path = artifact_paths["summary_path"]
            metrics_path = artifact_paths["metrics_path"]
            if summary_path is None:
                raise FileNotFoundError(f"Missing summary file for run: {run_name}")
            if metrics_path is None:
                raise FileNotFoundError(f"Missing metrics file for run: {run_name}")
            summary = load_json(summary_path)
            history = load_history(metrics_path)
            row = {
                "model": model_name,
                "model_label": MODEL_LABELS.get(model_name, model_name),
                "seed": seed,
                "run_name": run_name,
                "best_val_acc": SUMMARY_METRIC_EXTRACTORS["best_val_acc"](summary),
                "test_acc": SUMMARY_METRIC_EXTRACTORS["test_acc"](summary),
                "macro_f1": SUMMARY_METRIC_EXTRACTORS["macro_f1"](summary),
                "best_epoch": SUMMARY_METRIC_EXTRACTORS["best_epoch"](summary),
            }
            rows.append(row)
            histories.append(
                {
                    "model": model_name,
                    "model_label": MODEL_LABELS.get(model_name, model_name),
                    "seed": seed,
                    "run_name": run_name,
                    "history": history,
                }
            )
    return rows, histories


def summarize_by_model(rows, metrics):
    summary_rows = []
    grouped = {}
    for row in rows:
        grouped.setdefault(row["model"], []).append(row)

    for model_name, model_rows in grouped.items():
        item = {
            "model": model_name,
            "model_label": MODEL_LABELS.get(model_name, model_name),
            "num_seeds": len(model_rows),
        }
        for metric in metrics + ["best_epoch"]:
            values = [float(r[metric]) for r in model_rows]
            item[f"{metric}_mean"] = safe_mean(values)
            item[f"{metric}_std"] = safe_stdev(values)
            item[f"{metric}_min"] = min(values)
            item[f"{metric}_max"] = max(values)
        summary_rows.append(item)
    return summary_rows


def build_delta_rows(summary_rows, reference_model, metrics):
    summary_by_model = {row["model"]: row for row in summary_rows}
    if reference_model not in summary_by_model:
        raise ValueError(f"Reference model {reference_model} is not in the selected models.")
    reference = summary_by_model[reference_model]

    delta_rows = []
    for row in summary_rows:
        delta_row = {
            "model": row["model"],
            "model_label": row["model_label"],
            "reference_model": reference_model,
        }
        for metric in metrics:
            delta_row[f"{metric}_delta_vs_{reference_model}"] = (
                row[f"{metric}_mean"] - reference[f"{metric}_mean"]
            )
        delta_rows.append(delta_row)
    return delta_rows


def determine_seed_winners(rows, metric):
    winners = {}
    grouped = {}
    for row in rows:
        grouped.setdefault(row["seed"], []).append(row)
    for seed, seed_rows in grouped.items():
        winner = max(seed_rows, key=lambda item: item[metric])
        winners[seed] = winner["model"]
    return winners


def build_headline_insights(rows, summary_rows, reference_model):
    insights = []
    summary_by_model = {row["model"]: row for row in summary_rows}
    test_winners = determine_seed_winners(rows, "test_acc")
    win_counts = {}
    for model_name in test_winners.values():
        win_counts[model_name] = win_counts.get(model_name, 0) + 1

    best_model = max(summary_rows, key=lambda item: item["test_acc_mean"])
    insights.append(
        f"{best_model['model_label']} has the highest mean test accuracy: "
        f"{format_float(best_model['test_acc_mean'])} +- {format_float(best_model['test_acc_std'])}."
    )

    reference = summary_by_model.get(reference_model)
    if reference is not None:
        for row in summary_rows:
            if row["model"] == reference_model:
                continue
            delta = row["test_acc_mean"] - reference["test_acc_mean"]
            insights.append(
                f"{row['model_label']} changes mean test accuracy by "
                f"{delta:+.4f} versus {reference['model_label']}."
            )

    if win_counts:
        top_winner = max(win_counts.items(), key=lambda item: item[1])
        insights.append(
            f"{MODEL_LABELS.get(top_winner[0], top_winner[0])} wins "
            f"{top_winner[1]}/{len(test_winners)} seeds on test accuracy."
        )

    return insights


def ensure_report_dirs(results_dir: Path, report_name: str, experiment_name: str | None = None):
    report_paths = build_report_artifact_dirs(
        results_dir=results_dir,
        report_name=report_name,
        experiment_name=experiment_name,
    )
    report_dir = report_paths["report_dir"]
    figures_dir = report_paths["figures_dir"]
    figures_dir.mkdir(parents=True, exist_ok=True)
    return report_dir, figures_dir


def write_csv(path: Path, fieldnames, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def determine_available_curve_metrics(histories):
    available = []
    for metric in CURVE_METRICS:
        if any(
            any(metric in row and isinstance(row.get(metric), (int, float)) for row in item["history"])
            for item in histories
        ):
            available.append(metric)
    return available


def build_epoch_curve_rows(histories, metric):
    grouped = {}
    for item in histories:
        grouped.setdefault(item["model"], []).append(item)

    rows = []
    for model_name, model_histories in grouped.items():
        by_epoch = {}
        for history_item in model_histories:
            for row in history_item["history"]:
                epoch = row.get("epoch")
                value = row.get(metric)
                if not isinstance(epoch, (int, float)) or not isinstance(value, (int, float)):
                    continue
                epoch_index = int(epoch)
                by_epoch.setdefault(epoch_index, []).append(float(value))

        for epoch_index in sorted(by_epoch.keys()):
            values = by_epoch[epoch_index]
            rows.append(
                {
                    "model": model_name,
                    "model_label": MODEL_LABELS.get(model_name, model_name),
                    "epoch": epoch_index,
                    "count": len(values),
                    "mean": safe_mean(values),
                    "std": safe_stdev(values),
                    "min": min(values),
                    "max": max(values),
                }
            )
    return rows


def write_epoch_curve_csv(report_dir: Path, metric: str, rows):
    path = report_dir / f"{metric}_epoch_mean_std.csv"
    fieldnames = ["model", "model_label", "epoch", "count", "mean", "std", "min", "max"]
    write_csv(path, fieldnames, rows)
    return path


def plot_epoch_mean_std(figures_dir: Path, metric: str, rows):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    grouped = {}
    for row in rows:
        grouped.setdefault(row["model"], []).append(row)

    for model_name, model_rows in grouped.items():
        model_rows = sorted(model_rows, key=lambda item: item["epoch"])
        epochs = [item["epoch"] for item in model_rows]
        means = [item["mean"] for item in model_rows]
        stds = [item["std"] for item in model_rows]
        color = MODEL_COLORS.get(model_name, "#2563eb")
        label = MODEL_LABELS.get(model_name, model_name)
        lower = [mean_value - std_value for mean_value, std_value in zip(means, stds)]
        upper = [mean_value + std_value for mean_value, std_value in zip(means, stds)]

        ax.plot(epochs, means, linewidth=2.2, label=label, color=color)
        ax.fill_between(epochs, lower, upper, color=color, alpha=0.16)

    ylabel = CURVE_METRIC_LABELS.get(metric, metric)
    if metric.endswith("acc") or metric.endswith("f1"):
        ax.set_ylabel(f"{ylabel} (mean +/- std)")
    else:
        ax.set_ylabel(f"{ylabel} (mean +/- std)")
    ax.set_xlabel("Epoch")
    ax.set_title(f"{ylabel} Across Epochs")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()

    path = figures_dir / f"{metric}_epoch_mean_std.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def write_overview(
    path: Path,
    args,
    summary_rows,
    delta_rows,
    insights,
    curve_metrics,
):
    metric_headers = args.metrics
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Seed Sweep Summary\n\n")
        handle.write(f"- Seeds: {', '.join(str(seed) for seed in args.seeds)}\n")
        handle.write(f"- Models: {', '.join(args.models)}\n")
        handle.write(f"- Reference model: {args.reference_model}\n\n")
        handle.write(f"- Epoch curve metrics: {', '.join(curve_metrics)}\n\n")

        handle.write("## Headline Insights\n\n")
        for insight in insights:
            handle.write(f"- {insight}\n")
        handle.write("\n")

        handle.write("## Aggregate Table\n\n")
        header = (
            "| Model | "
            + " | ".join(f"{metric} mean +- std" for metric in metric_headers)
            + " |\n"
        )
        separator = "|---" * (len(metric_headers) + 1) + "|\n"
        handle.write(header)
        handle.write(separator)
        for row in summary_rows:
            values = []
            for metric in metric_headers:
                values.append(
                    f"{format_float(row[f'{metric}_mean'])} +- {format_float(row[f'{metric}_std'])}"
                )
            handle.write(f"| {row['model_label']} | " + " | ".join(values) + " |\n")
        handle.write("\n")

        handle.write("## Delta Vs Reference\n\n")
        delta_header = (
            "| Model | "
            + " | ".join(f"{metric} delta" for metric in metric_headers)
            + " |\n"
        )
        handle.write(delta_header)
        handle.write(separator)
        for row in delta_rows:
            values = [f"{row[f'{metric}_delta_vs_{args.reference_model}']:+.4f}" for metric in metric_headers]
            handle.write(f"| {row['model_label']} | " + " | ".join(values) + " |\n")


def main():
    args = parse_args()
    report_name = args.report_name or make_default_report_name(args)
    rows, histories = collect_run_rows(args)
    summary_rows = summarize_by_model(rows, args.metrics)
    delta_rows = build_delta_rows(summary_rows, args.reference_model, args.metrics)
    insights = build_headline_insights(rows, summary_rows, args.reference_model)
    curve_metrics = determine_available_curve_metrics(histories)

    report_dir, figures_dir = ensure_report_dirs(
        args.results_dir,
        report_name,
        experiment_name=args.experiment_name,
    )
    per_seed_csv = report_dir / "per_seed_metrics.csv"
    aggregate_csv = report_dir / "aggregate_summary.csv"
    delta_csv = report_dir / "delta_vs_reference.csv"
    overview_md = report_dir / "overview.md"
    manifest_json = report_dir / "summary_manifest.json"

    write_csv(
        per_seed_csv,
        ["model", "model_label", "seed", "run_name", "best_val_acc", "test_acc", "macro_f1", "best_epoch"],
        rows,
    )

    aggregate_fieldnames = ["model", "model_label", "num_seeds"]
    for metric in args.metrics + ["best_epoch"]:
        aggregate_fieldnames.extend(
            [f"{metric}_mean", f"{metric}_std", f"{metric}_min", f"{metric}_max"]
        )
    write_csv(aggregate_csv, aggregate_fieldnames, summary_rows)

    delta_fieldnames = ["model", "model_label", "reference_model"] + [
        f"{metric}_delta_vs_{args.reference_model}" for metric in args.metrics
    ]
    write_csv(delta_csv, delta_fieldnames, delta_rows)

    epoch_curve_csvs = {}
    epoch_curve_figures = {}
    for metric in curve_metrics:
        epoch_rows = build_epoch_curve_rows(histories, metric)
        epoch_curve_csvs[metric] = str(write_epoch_curve_csv(report_dir, metric, epoch_rows))
        epoch_curve_figures[metric] = str(plot_epoch_mean_std(figures_dir, metric, epoch_rows))

    write_overview(overview_md, args, summary_rows, delta_rows, insights, curve_metrics)

    manifest = {
        "report_name": report_name,
        "seeds": args.seeds,
        "models": args.models,
        "reference_model": args.reference_model,
        "metrics": args.metrics,
        "aggregate_csv": str(aggregate_csv),
        "per_seed_csv": str(per_seed_csv),
        "delta_csv": str(delta_csv),
        "overview_md": str(overview_md),
        "epoch_curve_csvs": epoch_curve_csvs,
        "epoch_curve_figures": epoch_curve_figures,
        "headline_insights": insights,
    }

    with manifest_json.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"Report directory: {report_dir}")
    print(f"Per-seed CSV: {per_seed_csv}")
    print(f"Aggregate CSV: {aggregate_csv}")
    print(f"Delta CSV: {delta_csv}")
    print(f"Overview Markdown: {overview_md}")
    print(f"Manifest JSON: {manifest_json}")
    for key, path in epoch_curve_figures.items():
        print(f"Epoch Figure ({key}): {path}")


if __name__ == "__main__":
    main()
