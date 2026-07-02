import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from result_paths import build_report_artifact_dirs, resolve_run_artifact_paths


DEFAULT_CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

DEFAULT_RUN_SPECS = [
    "vit_baseline=ViT Baseline",
    "vit_rope=ViT RoPE",
    "vit_rope_2d=ViT RoPE 2D",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate per-class comparison tables and plots from saved run summaries."
    )
    parser.add_argument(
        "--run",
        dest="runs",
        action="append",
        default=None,
        help="Run spec in the form run_name or run_name=Display Label. Repeatable.",
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--report-name", type=str, default=None)
    parser.add_argument(
        "--reference-run",
        type=str,
        default=None,
        help="Run name used as the reference for delta plots. Defaults to the first run.",
    )
    return parser.parse_args()


def parse_run_spec(spec: str):
    if "=" in spec:
        run_name, label = spec.split("=", 1)
        return run_name.strip(), label.strip()
    return spec.strip(), spec.strip()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def make_default_report_name(run_specs):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    joined = "_vs_".join(run_name for run_name, _ in run_specs[:3])
    joined = joined[:60]
    return f"per_class_{joined}_{timestamp}"


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


def load_run_payload(results_dir: Path, run_name: str, label: str, experiment_name: str | None = None):
    summary_path = resolve_run_artifact_paths(
        results_dir,
        run_name,
        experiment_name=experiment_name,
    )["summary_path"]
    if summary_path is None:
        raise FileNotFoundError(f"Missing summary artifact for run '{run_name}'.")
    summary = load_json(summary_path)
    selected = summary["selected_model"]
    return {
        "run_name": run_name,
        "label": label,
        "test_acc": float(selected["test_acc"]),
        "macro_f1": float(selected["test_macro_f1"]),
        "per_class_accuracy": [float(value) for value in selected["test_per_class_accuracy"]],
        "per_class_f1": [float(value) for value in selected["test_per_class_f1"]],
        "confusion_csv": selected.get("test_confusion_matrix_csv"),
    }


def load_class_names(results_dir: Path, runs):
    confusion_csv = runs[0].get("confusion_csv")
    if confusion_csv:
        csv_path = Path(confusion_csv)
        if not csv_path.is_absolute():
            csv_path = Path.cwd() / csv_path
        if csv_path.exists():
            with csv_path.open("r", encoding="utf-8") as handle:
                reader = csv.reader(handle)
                header = next(reader, None)
                if header and len(header) > 1:
                    return header[1:]

    num_classes = len(runs[0]["per_class_accuracy"])
    if num_classes == len(DEFAULT_CLASS_NAMES):
        return DEFAULT_CLASS_NAMES
    return [f"class_{index}" for index in range(num_classes)]


def build_metric_rows(class_names, runs, metric_key):
    rows = []
    reference = runs[0]
    for class_index, class_name in enumerate(class_names):
        row = {
            "class_index": class_index,
            "class_name": class_name,
        }
        reference_value = reference[metric_key][class_index]
        row[f"{reference['run_name']}_value"] = reference_value

        for run in runs:
            value = run[metric_key][class_index]
            row[f"{run['run_name']}_value"] = value
            row[f"{run['run_name']}_delta_vs_{reference['run_name']}"] = value - reference_value
        rows.append(row)
    return rows


def write_csv(path: Path, fieldnames, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_grouped_metric(figures_dir: Path, class_names, runs, metric_key, title, filename):
    x = np.arange(len(class_names))
    width = 0.24 if len(runs) >= 3 else 0.32

    fig, ax = plt.subplots(figsize=(12, 5.5))
    colors = ["#2563eb", "#0891b2", "#16a34a", "#f59e0b"]
    for index, run in enumerate(runs):
        offset = (index - (len(runs) - 1) / 2) * width
        values = run[metric_key]
        ax.bar(x + offset, values, width=width, label=run["label"], color=colors[index % len(colors)])

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=25, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel(metric_key)
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()

    path = figures_dir / filename
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_delta_vs_reference(figures_dir: Path, class_names, runs, metric_key, filename):
    reference = runs[0]
    x = np.arange(len(class_names))
    comparison_runs = runs[1:]
    width = 0.32 if len(comparison_runs) == 2 else 0.22

    fig, ax = plt.subplots(figsize=(12, 5.5))
    colors = ["#0891b2", "#16a34a", "#f59e0b"]
    for index, run in enumerate(comparison_runs):
        offset = (index - (len(comparison_runs) - 1) / 2) * width
        deltas = np.array(run[metric_key]) - np.array(reference[metric_key])
        ax.bar(
            x + offset,
            deltas,
            width=width,
            label=f"{run['label']} - {reference['label']}",
            color=colors[index % len(colors)],
        )

    ax.axhline(0.0, color="#0f172a", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=25, ha="right")
    ax.set_ylabel(f"{metric_key} delta")
    ax.set_title(f"Per-Class {metric_key} Delta vs {reference['label']}")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()

    path = figures_dir / filename
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def top_delta_lines(class_names, reference_run, target_run, metric_key, top_k=3):
    deltas = np.array(target_run[metric_key]) - np.array(reference_run[metric_key])
    top_indices = np.argsort(deltas)[::-1][:top_k]
    bottom_indices = np.argsort(deltas)[:top_k]

    positive = [
        f"{class_names[index]} ({deltas[index]:+.4f})"
        for index in top_indices
    ]
    negative = [
        f"{class_names[index]} ({deltas[index]:+.4f})"
        for index in bottom_indices
    ]
    return positive, negative


def weakest_classes(class_names, run, metric_key, top_k=3):
    values = np.array(run[metric_key])
    weakest_indices = np.argsort(values)[:top_k]
    return [
        f"{class_names[index]} ({values[index]:.4f})"
        for index in weakest_indices
    ]


def write_overview(path: Path, class_names, runs, accuracy_rows, f1_rows):
    reference = runs[0]
    best_run = max(runs, key=lambda item: item["test_acc"])

    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Per-Class Comparison Overview\n\n")
        handle.write("## Overall Metrics\n\n")
        for run in runs:
            handle.write(
                f"- {run['label']}: test_acc={run['test_acc']:.4f}, macro_f1={run['macro_f1']:.4f}\n"
            )
        handle.write("\n")

        handle.write("## Headline Takeaways\n\n")
        handle.write(
            f"- Best overall run in this comparison: {best_run['label']} "
            f"(test_acc={best_run['test_acc']:.4f}, macro_f1={best_run['macro_f1']:.4f}).\n"
        )
        handle.write(
            f"- Reference run for class-level deltas: {reference['label']}.\n"
        )
        handle.write("\n")

        handle.write("## Per-Class Accuracy Changes\n\n")
        for run in runs[1:]:
            top_positive, top_negative = top_delta_lines(class_names, reference, run, "per_class_accuracy")
            handle.write(f"### {run['label']} vs {reference['label']}\n\n")
            handle.write(f"- Top improved classes: {', '.join(top_positive)}\n")
            handle.write(f"- Largest drops / weakest changes: {', '.join(top_negative)}\n\n")

        handle.write("## Weakest Classes In The Best Run\n\n")
        weakest = weakest_classes(class_names, best_run, "per_class_accuracy")
        handle.write(f"- Lowest per-class accuracy in {best_run['label']}: {', '.join(weakest)}\n\n")

        handle.write("## Suggested PPT Narrative\n\n")
        handle.write(
            "- Use this page to explain whether performance differences come from a specific class bias rather than only the overall accuracy.\n"
        )
        handle.write(
            "- For directional experiments, focus on whether one model is stronger on the horizontal class while another is stronger on the vertical class.\n"
        )
        handle.write(
            "- If one model collapses to a majority class, the confusion matrix and per-class F1 should make that visible immediately.\n"
        )


def main():
    args = parse_args()
    run_specs = [parse_run_spec(spec) for spec in (args.runs or DEFAULT_RUN_SPECS)]
    report_name = args.report_name or make_default_report_name(run_specs)
    report_dir, figures_dir = ensure_report_dirs(
        args.results_dir,
        report_name,
        experiment_name=args.experiment_name,
    )

    runs = [
        load_run_payload(args.results_dir, run_name, label, experiment_name=args.experiment_name)
        for run_name, label in run_specs
    ]
    class_names = load_class_names(args.results_dir, runs)

    if args.reference_run:
        reference_index = next(
            (index for index, run in enumerate(runs) if run["run_name"] == args.reference_run),
            None,
        )
        if reference_index is None:
            raise ValueError(f"Reference run {args.reference_run} was not found in the selected runs.")
        runs[0], runs[reference_index] = runs[reference_index], runs[0]

    accuracy_rows = build_metric_rows(class_names, runs, "per_class_accuracy")
    f1_rows = build_metric_rows(class_names, runs, "per_class_f1")

    accuracy_csv = report_dir / "per_class_accuracy_comparison.csv"
    f1_csv = report_dir / "per_class_f1_comparison.csv"
    overview_md = report_dir / "overview.md"
    manifest_json = report_dir / "report_manifest.json"

    accuracy_fields = ["class_index", "class_name"]
    f1_fields = ["class_index", "class_name"]
    reference_run_name = runs[0]["run_name"]
    for run in runs:
        accuracy_fields.extend(
            [
                f"{run['run_name']}_value",
                f"{run['run_name']}_delta_vs_{reference_run_name}",
            ]
        )
        f1_fields.extend(
            [
                f"{run['run_name']}_value",
                f"{run['run_name']}_delta_vs_{reference_run_name}",
            ]
        )

    write_csv(accuracy_csv, accuracy_fields, accuracy_rows)
    write_csv(f1_csv, f1_fields, f1_rows)

    accuracy_grouped = plot_grouped_metric(
        figures_dir=figures_dir,
        class_names=class_names,
        runs=runs,
        metric_key="per_class_accuracy",
        title="Per-Class Accuracy Comparison",
        filename="per_class_accuracy_grouped.png",
    )
    f1_grouped = plot_grouped_metric(
        figures_dir=figures_dir,
        class_names=class_names,
        runs=runs,
        metric_key="per_class_f1",
        title="Per-Class F1 Comparison",
        filename="per_class_f1_grouped.png",
    )
    accuracy_delta = plot_delta_vs_reference(
        figures_dir=figures_dir,
        class_names=class_names,
        runs=runs,
        metric_key="per_class_accuracy",
        filename="per_class_accuracy_delta_vs_reference.png",
    )

    write_overview(overview_md, class_names, runs, accuracy_rows, f1_rows)

    manifest = {
        "report_name": report_name,
        "reference_run": reference_run_name,
        "runs": [{"run_name": run["run_name"], "label": run["label"]} for run in runs],
        "accuracy_csv": str(accuracy_csv),
        "f1_csv": str(f1_csv),
        "overview_md": str(overview_md),
        "figures": {
            "per_class_accuracy_grouped": str(accuracy_grouped),
            "per_class_f1_grouped": str(f1_grouped),
            "per_class_accuracy_delta_vs_reference": str(accuracy_delta),
        },
    }
    with manifest_json.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"Report directory: {report_dir}")
    print(f"Accuracy CSV: {accuracy_csv}")
    print(f"F1 CSV: {f1_csv}")
    print(f"Overview Markdown: {overview_md}")
    print(f"Manifest JSON: {manifest_json}")
    print(f"Figure (accuracy grouped): {accuracy_grouped}")
    print(f"Figure (f1 grouped): {f1_grouped}")
    print(f"Figure (accuracy delta): {accuracy_delta}")


if __name__ == "__main__":
    main()
