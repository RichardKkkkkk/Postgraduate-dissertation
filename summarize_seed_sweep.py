import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model_registry import EXPERIMENT_REGISTRY


DEFAULT_MODELS = ["vit_baseline", "vit_rope", "vit_rope_2d"]
DEFAULT_METRICS = ["best_val_acc", "test_acc", "macro_f1"]
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
        help="Output report folder name under results/reports/.",
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


def safe_mean(values):
    return mean(values) if values else math.nan


def safe_stdev(values):
    return stdev(values) if len(values) > 1 else 0.0


def format_float(value):
    return f"{value:.4f}"


def collect_run_rows(args):
    rows = []
    for model_name in args.models:
        for seed in args.seeds:
            run_name = build_run_name(model_name, seed, args.run_prefix)
            summary_path = args.results_dir / "metrics" / f"{run_name}_summary.json"
            if not summary_path.exists():
                raise FileNotFoundError(f"Missing summary file: {summary_path}")
            summary = load_json(summary_path)
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
    return rows


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


def ensure_report_dirs(results_dir: Path, report_name: str):
    report_dir = results_dir / "reports" / report_name
    figures_dir = report_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    return report_dir, figures_dir


def write_csv(path: Path, fieldnames, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_metric_error_bars(figures_dir: Path, summary_rows, metric):
    labels = [row["model_label"] for row in summary_rows]
    means = [row[f"{metric}_mean"] for row in summary_rows]
    stds = [row[f"{metric}_std"] for row in summary_rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(labels))
    ax.bar(x, means, yerr=stds, capsize=6, color=["#2563eb", "#0891b2", "#16a34a", "#f59e0b"][: len(labels)])
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(f"{METRIC_LABELS.get(metric, metric)} Across Seeds")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()

    path = figures_dir / f"{metric}_mean_std.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_seed_lines(figures_dir: Path, rows, metric):
    fig, ax = plt.subplots(figsize=(8, 5))
    grouped = {}
    for row in rows:
        grouped.setdefault(row["model"], []).append(row)

    for model_name, model_rows in grouped.items():
        model_rows = sorted(model_rows, key=lambda item: item["seed"])
        ax.plot(
            [item["seed"] for item in model_rows],
            [item[metric] for item in model_rows],
            marker="o",
            linewidth=2,
            label=MODEL_LABELS.get(model_name, model_name),
        )

    ax.set_xlabel("Seed")
    ax.set_ylabel(metric)
    ax.set_title(f"{METRIC_LABELS.get(metric, metric)} by Seed")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()

    path = figures_dir / f"{metric}_by_seed.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def write_overview(
    path: Path,
    args,
    summary_rows,
    delta_rows,
    insights,
):
    metric_headers = args.metrics
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Seed Sweep Summary\n\n")
        handle.write(f"- Seeds: {', '.join(str(seed) for seed in args.seeds)}\n")
        handle.write(f"- Models: {', '.join(args.models)}\n")
        handle.write(f"- Reference model: {args.reference_model}\n\n")

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
    rows = collect_run_rows(args)
    summary_rows = summarize_by_model(rows, args.metrics)
    delta_rows = build_delta_rows(summary_rows, args.reference_model, args.metrics)
    insights = build_headline_insights(rows, summary_rows, args.reference_model)

    report_dir, figures_dir = ensure_report_dirs(args.results_dir, report_name)
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

    metric_figures = {}
    for metric in args.metrics:
        metric_figures[f"{metric}_mean_std"] = str(plot_metric_error_bars(figures_dir, summary_rows, metric))
        metric_figures[f"{metric}_by_seed"] = str(plot_seed_lines(figures_dir, rows, metric))

    write_overview(overview_md, args, summary_rows, delta_rows, insights)

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
        "figures": metric_figures,
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
    for key, path in metric_figures.items():
        print(f"Figure ({key}): {path}")


if __name__ == "__main__":
    main()
