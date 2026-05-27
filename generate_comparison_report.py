import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from textwrap import shorten

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PRIORITY_METRICS = [
    "test_acc",
    "val_acc",
    "test_loss",
    "val_loss",
    "train_acc",
    "train_loss",
]
SUMMARY_METRIC_PRIORITY = [
    "test_macro_f1",
    "val_macro_f1",
    "test_macro_precision",
    "test_macro_recall",
    "val_macro_precision",
    "val_macro_recall",
]
PER_CLASS_PRIORITY = [
    "test_per_class_accuracy",
    "test_per_class_f1",
    "test_per_class_precision",
    "test_per_class_recall",
]
RUN_INFO_PRIORITY = [
    "model.architecture",
    "model.variant",
    "training.batch_size",
    "training.lr",
    "training.weight_decay",
    "training.epochs",
    "dataset.name",
    "dataset.image_size",
    "image_size",
    "batch_size",
    "lr",
    "weight_decay",
    "epochs",
    "weights",
    "embedding_dropout",
    "attention_dropout",
    "projection_dropout",
    "mlp_dropout",
    "early_stopping_metric",
    "early_stopping_patience",
    "val_ratio",
]
CONFIG_PRIORITY_PREFIXES = ["model.", "training.", "dataset.", "device", "command"]
SLIDE_WIDTH_INCHES = 13.333
SLIDE_HEIGHT_INCHES = 7.5

COLOR_BG = (248, 250, 252)
COLOR_NAVY = (15, 23, 42)
COLOR_BLUE = (37, 99, 235)
COLOR_CYAN = (8, 145, 178)
COLOR_TEXT = (30, 41, 59)
COLOR_MUTED = (100, 116, 139)
COLOR_BORDER = (203, 213, 225)
COLOR_GOOD = (22, 163, 74)
COLOR_BAD = (220, 38, 38)
COLOR_CARD = (255, 255, 255)
COLOR_HEADER_FILL = (226, 232, 240)
FONT_FAMILY = "Aptos"


@dataclass
class RunArtifacts:
    run_name: str
    label: str
    history: list[dict]
    config: dict
    summary: dict
    available_metrics: list[str]
    flat_config: dict
    model_name: str
    device: str
    completed_epochs: int
    selected_epoch: int | None
    selected_metrics: dict[str, float]
    per_class_metrics: dict[str, list[float]]
    confusion_matrix_csv: Path | None
    confusion_matrix_figure: Path | None
    early_stopping: dict | None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a comparison report and meeting-ready PPT for experiment runs."
    )
    parser.add_argument(
        "--run",
        dest="runs",
        action="append",
        required=True,
        help="Run spec in the form run_name or run_name=Display Label. Repeatable.",
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--report-name", type=str, default=None)
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        help="Metrics to compare. Defaults to the intersection of numeric metrics across runs.",
    )
    parser.add_argument(
        "--max-config-rows",
        type=int,
        default=18,
        help="Maximum number of varying config rows to show in the PPT run-info/config sections.",
    )
    parser.add_argument(
        "--skip-ppt",
        action="store_true",
        help="Skip PowerPoint generation and only write plots and summary files.",
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


def flatten_dict(data, prefix=""):
    flat = {}
    for key, value in data.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(flatten_dict(value, full_key))
        else:
            flat[full_key] = value
    return flat


def infer_model_name(config: dict):
    model_cfg = config.get("model", {})
    if isinstance(model_cfg, dict) and model_cfg.get("architecture"):
        return str(model_cfg["architecture"])

    command = str(config.get("command", ""))
    if "train_cnn_cifar10.py" in command:
        return "resnet18"
    if "train_cifar10.py" in command:
        return "vit"
    if config.get("weights") is not None:
        return "resnet18"
    if any(key in config for key in ["embedding_dropout", "attention_dropout", "mlp_dropout"]):
        return "vit"
    return "unknown"


def resolve_optional_path(project_root: Path, value):
    if not value:
        return None
    path = Path(str(value))
    candidates = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.append(project_root / path)
        candidates.append((project_root / path.as_posix()).resolve())
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def extract_selected_metrics(summary: dict):
    selected = summary.get("selected_model")
    metrics = {}
    if not isinstance(selected, dict):
        return metrics
    for key, value in selected.items():
        if isinstance(value, (int, float)) and key != "epoch":
            metrics[key] = float(value)
    return metrics


def extract_per_class_metrics(summary: dict):
    selected = summary.get("selected_model")
    metrics = {}
    if not isinstance(selected, dict):
        return metrics
    for key, value in selected.items():
        if key.startswith("test_per_class_") and isinstance(value, list) and value:
            if all(isinstance(item, (int, float)) for item in value):
                metrics[key] = [float(item) for item in value]
    return metrics


def load_run_artifacts(results_dir: Path, project_root: Path, run_name: str, label: str):
    metrics_dir = results_dir / "metrics"
    history_path = metrics_dir / f"{run_name}_metrics.csv"
    config_path = metrics_dir / f"{run_name}_config.json"
    summary_path = metrics_dir / f"{run_name}_summary.json"

    missing = [path.name for path in [history_path, config_path, summary_path] if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing artifacts for run '{run_name}': {', '.join(missing)}"
        )

    history = load_history(history_path)
    if not history:
        raise ValueError(f"No metric rows found in {history_path}")

    available_metrics = [
        key
        for key, value in history[0].items()
        if key != "epoch" and isinstance(value, (int, float))
    ]
    config = load_json(config_path)
    summary = load_json(summary_path)
    flat_config = flatten_dict(config)
    selected = summary.get("selected_model", {}) if isinstance(summary.get("selected_model"), dict) else {}

    return RunArtifacts(
        run_name=run_name,
        label=label,
        history=history,
        config=config,
        summary=summary,
        available_metrics=available_metrics,
        flat_config=flat_config,
        model_name=infer_model_name(config),
        device=str(config.get("device", "unknown")),
        completed_epochs=int(history[-1]["epoch"]),
        selected_epoch=int(selected["epoch"]) if isinstance(selected.get("epoch"), (int, float)) else None,
        selected_metrics=extract_selected_metrics(summary),
        per_class_metrics=extract_per_class_metrics(summary),
        confusion_matrix_csv=resolve_optional_path(
            project_root, selected.get("test_confusion_matrix_csv")
        ),
        confusion_matrix_figure=resolve_optional_path(
            project_root, selected.get("test_confusion_matrix_figure")
        ),
        early_stopping=summary.get("early_stopping") if isinstance(summary.get("early_stopping"), dict) else None,
    )


def order_metrics(metrics):
    seen = set()
    ordered = []
    for metric in PRIORITY_METRICS:
        if metric in metrics and metric not in seen:
            ordered.append(metric)
            seen.add(metric)
    for metric in metrics:
        if metric not in seen:
            ordered.append(metric)
            seen.add(metric)
    return ordered


def determine_metrics(runs, explicit_metrics=None):
    available_sets = [set(run.available_metrics) for run in runs]
    shared_metrics = set.intersection(*available_sets)
    if explicit_metrics:
        missing = [metric for metric in explicit_metrics if metric not in shared_metrics]
        if missing:
            raise ValueError(
                "These metrics are not present in every run: " + ", ".join(missing)
            )
        return order_metrics(explicit_metrics)
    return order_metrics([metric for metric in runs[0].available_metrics if metric in shared_metrics])


def determine_selected_metric_keys(runs, history_metrics):
    selected_sets = [set(run.selected_metrics.keys()) for run in runs if run.selected_metrics]
    if len(selected_sets) != len(runs):
        return []

    shared = set.intersection(*selected_sets)
    shared = {
        key
        for key in shared
        if key not in history_metrics
        and not key.startswith("test_per_class_")
        and "confusion_matrix" not in key
    }

    def priority(key):
        if key in SUMMARY_METRIC_PRIORITY:
            return SUMMARY_METRIC_PRIORITY.index(key)
        return len(SUMMARY_METRIC_PRIORITY)

    return sorted(shared, key=lambda key: (priority(key), key))


def is_percentage_metric(metric_name: str):
    tokens = ["acc", "accuracy", "precision", "recall", "f1"]
    lowered = metric_name.lower()
    return any(token in lowered for token in tokens)


def is_lower_better(metric_name: str):
    lowered = metric_name.lower()
    lower_keywords = ["loss", "error", "wer", "cer", "perplexity"]
    return any(keyword in lowered for keyword in lower_keywords)


def scale_metric_value(metric_name: str, value):
    if is_percentage_metric(metric_name):
        return value * 100.0
    return value


def metric_display_name(metric_name: str):
    tokens = metric_name.replace("_", " ").split()
    return " ".join(token.upper() if token in {"acc", "auc", "f1"} else token.capitalize() for token in tokens)


def format_metric_value(metric_name: str, value):
    scaled = scale_metric_value(metric_name, value)
    if is_percentage_metric(metric_name):
        return f"{scaled:.2f}%"
    return f"{scaled:.4f}"


def format_delta(metric_name: str, delta):
    if is_percentage_metric(metric_name):
        return f"{delta * 100.0:+.2f} pp"
    return f"{delta:+.4f}"


def compute_metric_summary(history, metric_name: str):
    rows = [row for row in history if metric_name in row]
    values = [row[metric_name] for row in rows]
    if is_lower_better(metric_name):
        best_index = min(range(len(values)), key=values.__getitem__)
    else:
        best_index = max(range(len(values)), key=values.__getitem__)

    final_row = rows[-1]
    best_row = rows[best_index]
    return {
        "final_value": final_row[metric_name],
        "final_epoch": int(final_row["epoch"]),
        "best_value": best_row[metric_name],
        "best_epoch": int(best_row["epoch"]),
    }


def build_summary_rows(runs, metrics, selected_metric_keys):
    rows = []
    for run in runs:
        row = {
            "run_name": run.run_name,
            "label": run.label,
            "model_name": run.model_name,
            "device": run.device,
            "completed_epochs": run.completed_epochs,
            "selected_epoch": run.selected_epoch,
        }
        for metric in metrics:
            metric_summary = compute_metric_summary(run.history, metric)
            row[f"final_{metric}"] = metric_summary["final_value"]
            row[f"best_{metric}"] = metric_summary["best_value"]
            row[f"best_epoch_{metric}"] = metric_summary["best_epoch"]
        for metric_name in selected_metric_keys:
            if metric_name in run.selected_metrics:
                row[f"selected_{metric_name}"] = run.selected_metrics[metric_name]
        rows.append(row)
    return rows


def choose_varying_config_keys(runs, max_rows):
    all_keys = sorted(set().union(*(run.flat_config.keys() for run in runs)))
    varying = []
    for key in all_keys:
        values = [run.flat_config.get(key) for run in runs]
        normalized = [json.dumps(value, sort_keys=True, ensure_ascii=False) for value in values]
        if len(set(normalized)) > 1:
            varying.append(key)

    def priority(key):
        for index, prefix in enumerate(CONFIG_PRIORITY_PREFIXES):
            if key.startswith(prefix):
                return index
        return len(CONFIG_PRIORITY_PREFIXES)

    varying.sort(key=lambda key: (priority(key), key))
    return varying[:max_rows], varying


def stringify_config_value(value):
    if value is None:
        return "None"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value)
    return str(value)


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


def build_metric_insight_rows(summary_rows, metrics, reference_label):
    rows = []
    for metric in metrics:
        if is_lower_better(metric):
            best_row = min(summary_rows, key=lambda row: row[f"final_{metric}"])
        else:
            best_row = max(summary_rows, key=lambda row: row[f"final_{metric}"])
        reference_row = next((row for row in summary_rows if row["label"] == reference_label), summary_rows[0])
        delta = best_row[f"final_{metric}"] - reference_row[f"final_{metric}"]
        rows.append(
            {
                "metric": metric,
                "best_run": best_row["label"],
                "best_value": best_row[f"final_{metric}"],
                "reference_delta": delta,
            }
        )
    return rows


def build_headline_insights(runs, summary_rows, metrics):
    if not metrics:
        return []

    reference_row = summary_rows[0]
    primary_metric = metrics[0]
    if is_lower_better(primary_metric):
        best_row = min(summary_rows, key=lambda row: row[f"final_{primary_metric}"])
    else:
        best_row = max(summary_rows, key=lambda row: row[f"final_{primary_metric}"])

    insights = [
        (
            f"Primary comparison metric: {metric_display_name(primary_metric)}. "
            f"Best final result is {best_row['label']} with "
            f"{format_metric_value(primary_metric, best_row[f'final_{primary_metric}'])}."
        )
    ]

    if best_row["label"] != reference_row["label"]:
        delta = best_row[f"final_{primary_metric}"] - reference_row[f"final_{primary_metric}"]
        insights.append(
            (
                f"Against reference run {reference_row['label']}, {best_row['label']} changes "
                f"{metric_display_name(primary_metric)} by {format_delta(primary_metric, delta)}."
            )
        )

    if "train_acc" in metrics and "test_acc" in metrics:
        gap_rows = []
        for row in summary_rows:
            gap = row["final_train_acc"] - row["final_test_acc"]
            gap_rows.append((gap, row["label"]))
        largest_gap, run_label = max(gap_rows)
        insights.append(
            (
                f"Largest train-to-test accuracy gap appears in {run_label} "
                f"({largest_gap * 100.0:.2f} pp), worth calling out during generalization discussion."
            )
        )

    stopped = []
    for run in runs:
        status = run.early_stopping or {}
        if status.get("enabled"):
            flag = "stopped early" if status.get("stopped_early") else "ran to planned epochs"
            stopped.append(f"{run.label}: {flag}")
    if stopped:
        insights.append("Early stopping summary: " + "; ".join(stopped[:3]))

    return insights[:4]


def build_conclusion_lines(runs, summary_rows, metrics, per_class_metric_keys):
    conclusions = []
    if metrics:
        primary_metric = metrics[0]
        if is_lower_better(primary_metric):
            winner = min(summary_rows, key=lambda row: row[f"final_{primary_metric}"])
        else:
            winner = max(summary_rows, key=lambda row: row[f"final_{primary_metric}"])
        conclusions.append(
            (
                f"Current headline result: {winner['label']} leads on "
                f"{metric_display_name(primary_metric)} with "
                f"{format_metric_value(primary_metric, winner[f'final_{primary_metric}'])}."
            )
        )

    if len(summary_rows) >= 2 and metrics:
        reference = summary_rows[0]
        challenger = summary_rows[1]
        metric = metrics[0]
        delta = challenger[f"final_{metric}"] - reference[f"final_{metric}"]
        conclusions.append(
            (
                f"When presenting baseline vs comparison, use {reference['label']} as reference and report "
                f"{challenger['label']} at {format_delta(metric, delta)} on {metric_display_name(metric)}."
            )
        )

    if per_class_metric_keys:
        conclusions.append(
            "Per-class pages are now included when selected-checkpoint classwise metrics exist, so class imbalance or hard classes can be discussed directly in meetings."
        )

    conclusions.append(
        "The deck is now organized for weekly reporting: setup, overview, key metrics, curves, error analysis, and a conclusion slide instead of raw log-style export."
    )
    return conclusions[:4]


def chunked(items, chunk_size):
    for index in range(0, len(items), chunk_size):
        yield items[index : index + chunk_size]


def estimate_table_font_size(column_count, row_count):
    if column_count <= 5 and row_count <= 6:
        return 14
    if column_count <= 6 and row_count <= 8:
        return 12
    if column_count <= 8 and row_count <= 10:
        return 11
    return 10


def save_summary_outputs(
    report_dir: Path,
    title: str,
    runs,
    metrics,
    selected_metric_keys,
    summary_rows,
    varying_keys,
    headline_insights,
    conclusion_lines,
):
    summary_csv_path = report_dir / "comparison_summary.csv"
    fieldnames = ["run_name", "label", "model_name", "device", "completed_epochs", "selected_epoch"]
    for metric in metrics:
        fieldnames.extend(
            [f"final_{metric}", f"best_{metric}", f"best_epoch_{metric}"]
        )
    for metric in selected_metric_keys:
        fieldnames.append(f"selected_{metric}")
    write_csv(summary_csv_path, fieldnames, summary_rows)

    config_csv_path = report_dir / "config_comparison.csv"
    config_rows = []
    for key in varying_keys:
        row = {"config_key": key}
        for run in runs:
            row[run.label] = stringify_config_value(run.flat_config.get(key))
        config_rows.append(row)
    write_csv(
        config_csv_path,
        ["config_key"] + [run.label for run in runs],
        config_rows,
    )

    overview_path = report_dir / "overview.md"
    overview_lines = [f"# {title}", ""]
    overview_lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    overview_lines.append(f"Reference run: `{runs[0].label}`")
    overview_lines.append("")
    overview_lines.append("## Headline Takeaways")
    overview_lines.append("")
    for line in headline_insights:
        overview_lines.append(f"- {line}")
    overview_lines.append("")
    overview_lines.append("## Runs")
    overview_lines.append("")
    for run in runs:
        overview_lines.append(
            f"- `{run.label}` (`{run.run_name}`), model: `{run.model_name}`, "
            f"epochs completed: `{run.completed_epochs}`, device: `{run.device}`"
        )
    overview_lines.append("")
    overview_lines.append("## Key Metrics")
    overview_lines.append("")
    for row in summary_rows:
        overview_lines.append(f"### {row['label']}")
        for metric in metrics:
            overview_lines.append(
                f"- {metric_display_name(metric)}: final {format_metric_value(metric, row[f'final_{metric}'])}, "
                f"best {format_metric_value(metric, row[f'best_{metric}'])} at epoch {row[f'best_epoch_{metric}']}"
            )
        for metric in selected_metric_keys:
            overview_lines.append(
                f"- Selected {metric_display_name(metric)}: {format_metric_value(metric, row[f'selected_{metric}'])}"
            )
        overview_lines.append("")
    overview_lines.append("## Suggested Meeting Conclusion")
    overview_lines.append("")
    for line in conclusion_lines:
        overview_lines.append(f"- {line}")
    overview_path.write_text("\n".join(overview_lines), encoding="utf-8")

    presentation_summary_path = report_dir / "presentation_summary.json"
    presentation_summary = {
        "title": title,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "reference_run": {"run_name": runs[0].run_name, "label": runs[0].label},
        "headline_insights": headline_insights,
        "conclusion_lines": conclusion_lines,
        "metrics": metrics,
        "selected_checkpoint_metrics": selected_metric_keys,
        "runs": summary_rows,
    }
    presentation_summary_path.write_text(
        json.dumps(presentation_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    manifest_path = report_dir / "report_manifest.json"
    manifest = {
        "report_dir": str(report_dir),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "runs": [{"run_name": run.run_name, "label": run.label} for run in runs],
        "metrics": metrics,
        "selected_checkpoint_metrics": selected_metric_keys,
        "summary_csv": str(summary_csv_path),
        "config_csv": str(config_csv_path),
        "overview_md": str(overview_path),
        "presentation_summary_json": str(presentation_summary_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary_csv_path, config_csv_path, overview_path, presentation_summary_path, manifest_path


def setup_plot_style():
    plt.style.use("seaborn-v0_8-whitegrid")


def plot_metric(figures_dir: Path, runs, metric_name: str):
    setup_plot_style()
    figure, axis = plt.subplots(figsize=(8.2, 4.8))
    colors = ["#2563eb", "#0891b2", "#16a34a", "#dc2626", "#7c3aed", "#ea580c"]
    for index, run in enumerate(runs):
        epochs = [int(row["epoch"]) for row in run.history]
        values = [scale_metric_value(metric_name, row[metric_name]) for row in run.history]
        axis.plot(
            epochs,
            values,
            marker="o",
            linewidth=2.4,
            markersize=3.8,
            label=run.label,
            color=colors[index % len(colors)],
        )

        final_value = values[-1]
        axis.scatter([epochs[-1]], [final_value], s=45, color=colors[index % len(colors)], zorder=3)

    axis.set_xlabel("Epoch")
    axis.set_ylabel("Percentage (%)" if is_percentage_metric(metric_name) else "Value")
    direction = "Higher is better" if not is_lower_better(metric_name) else "Lower is better"
    axis.set_title(f"{metric_display_name(metric_name)} Across Training ({direction})", fontsize=14, pad=12)
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best", frameon=True)
    figure.tight_layout()

    figure_path = figures_dir / f"{metric_name}_comparison.png"
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)
    return figure_path


def load_confusion_matrix_csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        rows = list(reader)
    if not rows or len(rows) < 2:
        raise ValueError(f"Confusion matrix CSV is empty: {path}")

    class_names = rows[0][1:]
    matrix = []
    true_labels = []
    for row in rows[1:]:
        true_labels.append(row[0])
        matrix.append([int(value) for value in row[1:]])
    return true_labels or class_names, matrix


def ensure_confusion_matrix_figure(figures_dir: Path, run: RunArtifacts):
    if run.confusion_matrix_figure and run.confusion_matrix_figure.exists():
        return run.confusion_matrix_figure
    if not run.confusion_matrix_csv or not run.confusion_matrix_csv.exists():
        return None

    class_names, matrix = load_confusion_matrix_csv(run.confusion_matrix_csv)
    data = np.array(matrix, dtype=float)

    setup_plot_style()
    figure, axis = plt.subplots(figsize=(6.2, 5.4))
    heatmap = axis.imshow(data, cmap="Blues")
    axis.set_xticks(range(len(class_names)))
    axis.set_yticks(range(len(class_names)))
    axis.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    axis.set_yticklabels(class_names, fontsize=8)
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_title(f"{run.label} Confusion Matrix", fontsize=13, pad=10)

    for row_index in range(data.shape[0]):
        for column_index in range(data.shape[1]):
            value = int(data[row_index, column_index])
            text_color = "white" if value > data.max() * 0.55 else "#0f172a"
            axis.text(column_index, row_index, str(value), ha="center", va="center", fontsize=7, color=text_color)

    figure.colorbar(heatmap, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure_path = figures_dir / f"{run.run_name}_confusion_matrix.png"
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)
    return figure_path


def determine_available_per_class_metric_keys(runs):
    metric_sets = [set(run.per_class_metrics.keys()) for run in runs if run.per_class_metrics]
    if len(metric_sets) != len(runs):
        return []

    shared = set.intersection(*metric_sets)

    def priority(metric_name):
        if metric_name in PER_CLASS_PRIORITY:
            return PER_CLASS_PRIORITY.index(metric_name)
        return len(PER_CLASS_PRIORITY)

    return sorted(shared, key=lambda metric_name: (priority(metric_name), metric_name))


def infer_class_names(run: RunArtifacts, metric_name: str):
    if run.confusion_matrix_csv and run.confusion_matrix_csv.exists():
        class_names, _ = load_confusion_matrix_csv(run.confusion_matrix_csv)
        return class_names
    value_count = len(run.per_class_metrics.get(metric_name, []))
    return [f"class_{index}" for index in range(value_count)]


def plot_per_class_metric(figures_dir: Path, runs, metric_name: str):
    class_names = infer_class_names(runs[0], metric_name)
    value_count = len(class_names)
    x_positions = np.arange(value_count)
    width = min(0.72 / max(1, len(runs)), 0.22)

    setup_plot_style()
    figure, axis = plt.subplots(figsize=(9.4, 4.8))
    colors = ["#2563eb", "#0891b2", "#16a34a", "#dc2626", "#7c3aed", "#ea580c"]

    for index, run in enumerate(runs):
        values = np.array(run.per_class_metrics[metric_name]) * 100.0
        offset = (index - (len(runs) - 1) / 2) * width
        axis.bar(
            x_positions + offset,
            values,
            width=width,
            label=run.label,
            color=colors[index % len(colors)],
            alpha=0.9,
        )

    axis.set_xticks(x_positions)
    axis.set_xticklabels(class_names, rotation=35, ha="right", fontsize=8)
    axis.set_ylim(0, max(100.0, axis.get_ylim()[1]))
    axis.set_ylabel("Percentage (%)")
    axis.set_title(f"{metric_display_name(metric_name)} by Class", fontsize=14, pad=12)
    axis.legend(loc="upper center", ncol=min(3, len(runs)), frameon=True)
    figure.tight_layout()

    figure_path = figures_dir / f"{metric_name}_per_class.png"
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)
    return figure_path, class_names


def summarize_per_class_metric(runs, metric_name: str, class_names):
    mean_scores = []
    for run in runs:
        values = run.per_class_metrics[metric_name]
        mean_scores.append((sum(values) / len(values), run.label))
    mean_scores.sort(reverse=not is_lower_better(metric_name))

    per_class_spread = []
    for index, class_name in enumerate(class_names):
        class_values = [run.per_class_metrics[metric_name][index] for run in runs]
        spread = max(class_values) - min(class_values)
        per_class_spread.append((spread, class_name))
    per_class_spread.sort(reverse=True)

    lines = [
        f"Highest mean {metric_display_name(metric_name)}: {mean_scores[0][1]} ({mean_scores[0][0] * 100.0:.2f}%).",
        f"Largest cross-run spread appears on class '{per_class_spread[0][1]}' ({per_class_spread[0][0] * 100.0:.2f} pp).",
    ]
    return lines


def top_confusion_pairs(confusion_matrix_csv: Path):
    class_names, matrix = load_confusion_matrix_csv(confusion_matrix_csv)
    pairs = []
    for row_index, true_label in enumerate(class_names):
        for column_index, pred_label in enumerate(class_names):
            if row_index == column_index:
                continue
            count = matrix[row_index][column_index]
            if count > 0:
                pairs.append((count, true_label, pred_label))
    pairs.sort(reverse=True)
    return pairs[:3]


def build_varying_rows(runs, varying_keys, max_rows):
    rows = []
    for key in varying_keys[:max_rows]:
        values = [stringify_config_value(run.flat_config.get(key)) for run in runs]
        rows.append((key, values))
    return rows


def make_default_report_name(run_specs):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_names = "_vs_".join(spec[0] for spec in run_specs[:2])
    short_names = short_names[:48]
    return f"comparison_{short_names}_{timestamp}"


def build_run_profile_lines(run: RunArtifacts, selected_varying_keys):
    lines = [
        f"Run ID: {run.run_name}",
        f"Model / device: {run.model_name} / {run.device}",
        f"Epochs completed: {run.completed_epochs}",
    ]

    batch_size = run.flat_config.get("training.batch_size", run.flat_config.get("batch_size"))
    lr = run.flat_config.get("training.lr", run.flat_config.get("lr"))
    weight_decay = run.flat_config.get("training.weight_decay", run.flat_config.get("weight_decay"))
    if batch_size is not None or lr is not None or weight_decay is not None:
        lines.append(
            "Batch / LR / WD: "
            f"{stringify_config_value(batch_size)} / {stringify_config_value(lr)} / {stringify_config_value(weight_decay)}"
        )

    command = run.config.get("command")
    if command:
        script_name = Path(str(command).split()[0]).name
        lines.append(f"Script: {script_name}")

    early_stopping = run.early_stopping or {}
    if early_stopping.get("enabled"):
        lines.append(
            "Early stopping: "
            f"{early_stopping.get('metric', 'unknown')} (patience={early_stopping.get('patience', 'n/a')}, "
            f"stopped_early={early_stopping.get('stopped_early', False)})"
        )

    def run_info_priority(key):
        if key in RUN_INFO_PRIORITY:
            return RUN_INFO_PRIORITY.index(key)
        return len(RUN_INFO_PRIORITY)

    added = 0
    for key in sorted(selected_varying_keys, key=lambda item: (run_info_priority(item), item)):
        if key in {
            "batch_size",
            "lr",
            "weight_decay",
            "training.batch_size",
            "training.lr",
            "training.weight_decay",
            "command",
        }:
            continue
        value = run.flat_config.get(key)
        if value is None:
            continue
        rendered = f"{key}: {stringify_config_value(value)}"
        lines.append(shorten(rendered, width=72, placeholder="..."))
        added += 1
        if added >= 2:
            break
    return lines[:7]


def build_results_overview_rows(summary_rows, metrics):
    chosen_metrics = metrics[: min(3, len(metrics))]
    headers = ["Run", "Model", "Epochs", "Selected Epoch"]
    headers.extend(metric_display_name(metric) for metric in chosen_metrics)

    rows = []
    for row in summary_rows:
        rows.append(
            [
                row["label"],
                row["model_name"],
                str(row["completed_epochs"]),
                str(row["selected_epoch"]) if row["selected_epoch"] is not None else "n/a",
                *[format_metric_value(metric, row[f"final_{metric}"]) for metric in chosen_metrics],
            ]
        )
    return headers, rows


def build_main_metric_rows(summary_rows, metrics, selected_metric_keys):
    headers = ["Metric", "Direction"] + [row["label"] for row in summary_rows] + ["Best Run"]
    rows = []

    for metric in metrics:
        best_row = min(summary_rows, key=lambda row: row[f"final_{metric}"]) if is_lower_better(metric) else max(summary_rows, key=lambda row: row[f"final_{metric}"])
        rows.append(
            [
                f"Final {metric_display_name(metric)}",
                "Lower" if is_lower_better(metric) else "Higher",
                *[format_metric_value(metric, row[f"final_{metric}"]) for row in summary_rows],
                best_row["label"],
            ]
        )

    for metric in selected_metric_keys:
        best_row = min(summary_rows, key=lambda row: row[f"selected_{metric}"]) if is_lower_better(metric) else max(summary_rows, key=lambda row: row[f"selected_{metric}"])
        rows.append(
            [
                f"Selected {metric_display_name(metric)}",
                "Lower" if is_lower_better(metric) else "Higher",
                *[format_metric_value(metric, row[f"selected_{metric}"]) for row in summary_rows],
                best_row["label"],
            ]
        )

    return headers, rows


def export_ppt(
    report_dir: Path,
    report_name: str,
    title: str,
    runs,
    metrics,
    selected_metric_keys,
    summary_rows,
    varying_rows,
    all_varying_keys,
    metric_figure_paths,
    per_class_figure_payloads,
    headline_insights,
    conclusion_lines,
):
    try:
        from pptx import Presentation
        from pptx.dml.color import RGBColor
        from pptx.enum.shapes import MSO_AUTO_SHAPE_TYPE
        from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
        from pptx.util import Inches, Pt
    except ImportError as exc:
        raise RuntimeError(
            "python-pptx is required for PPT export. Install it or use --skip-ppt."
        ) from exc

    def rgb(color):
        return RGBColor(*color)

    prs = Presentation()
    prs.slide_width = Inches(SLIDE_WIDTH_INCHES)
    prs.slide_height = Inches(SLIDE_HEIGHT_INCHES)

    def apply_slide_background(slide, color):
        fill = slide.background.fill
        fill.solid()
        fill.fore_color.rgb = rgb(color)

    def add_textbox(
        slide,
        left,
        top,
        width,
        height,
        text,
        font_size,
        color=COLOR_TEXT,
        bold=False,
        align=PP_ALIGN.LEFT,
        font_name=FONT_FAMILY,
    ):
        textbox = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
        text_frame = textbox.text_frame
        text_frame.clear()
        paragraph = text_frame.paragraphs[0]
        paragraph.text = text
        paragraph.alignment = align
        paragraph.space_after = 0
        paragraph.space_before = 0
        paragraph.line_spacing = 1.1
        run = paragraph.runs[0]
        run.font.name = font_name
        run.font.size = Pt(font_size)
        run.font.bold = bold
        run.font.color.rgb = rgb(color)
        return textbox

    def add_bullet_box(slide, left, top, width, height, bullets, font_size=16, color=COLOR_TEXT):
        textbox = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
        text_frame = textbox.text_frame
        text_frame.clear()
        text_frame.word_wrap = True
        for index, bullet in enumerate(bullets):
            paragraph = text_frame.paragraphs[0] if index == 0 else text_frame.add_paragraph()
            paragraph.text = bullet
            paragraph.level = 0
            paragraph.space_after = Pt(6)
            paragraph.line_spacing = 1.12
            for run in paragraph.runs:
                run.font.name = FONT_FAMILY
                run.font.size = Pt(font_size)
                run.font.color.rgb = rgb(color)
        return textbox

    def add_card(slide, left, top, width, height, title_text, body_lines, accent=COLOR_BLUE):
        shape = slide.shapes.add_shape(
            MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE,
            Inches(left),
            Inches(top),
            Inches(width),
            Inches(height),
        )
        shape.fill.solid()
        shape.fill.fore_color.rgb = rgb(COLOR_CARD)
        shape.line.color.rgb = rgb(COLOR_BORDER)
        shape.line.width = Pt(1.0)
        accent_bar = slide.shapes.add_shape(
            MSO_AUTO_SHAPE_TYPE.RECTANGLE,
            Inches(left),
            Inches(top),
            Inches(0.15),
            Inches(height),
        )
        accent_bar.fill.solid()
        accent_bar.fill.fore_color.rgb = rgb(accent)
        accent_bar.line.fill.background()
        add_textbox(slide, left + 0.28, top + 0.18, width - 0.42, 0.36, title_text, 18, bold=True)
        add_bullet_box(slide, left + 0.28, top + 0.62, width - 0.42, height - 0.78, body_lines, font_size=11)

    def add_metric_card(slide, left, top, width, height, label_text, value_text, accent=COLOR_BLUE):
        shape = slide.shapes.add_shape(
            MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE,
            Inches(left),
            Inches(top),
            Inches(width),
            Inches(height),
        )
        shape.fill.solid()
        shape.fill.fore_color.rgb = rgb(COLOR_CARD)
        shape.line.color.rgb = rgb(COLOR_BORDER)
        shape.line.width = Pt(1.0)
        add_textbox(slide, left + 0.18, top + 0.18, width - 0.36, 0.28, label_text, 12, color=COLOR_MUTED, bold=True)
        add_textbox(slide, left + 0.18, top + 0.55, width - 0.36, 0.48, value_text, 20, color=accent, bold=True)

    def add_slide_header(slide, title_text, subtitle_text="", section_label="Experiment Comparison"):
        pill = slide.shapes.add_shape(
            MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE,
            Inches(0.55),
            Inches(0.32),
            Inches(1.9),
            Inches(0.34),
        )
        pill.fill.solid()
        pill.fill.fore_color.rgb = rgb(COLOR_HEADER_FILL)
        pill.line.fill.background()
        add_textbox(slide, 0.72, 0.38, 1.6, 0.2, section_label, 10, color=COLOR_BLUE, bold=True)
        add_textbox(slide, 0.55, 0.72, 8.7, 0.45, title_text, 25, bold=True)
        if subtitle_text:
            add_textbox(slide, 0.56, 1.14, 11.9, 0.28, subtitle_text, 11, color=COLOR_MUTED)

        rule = slide.shapes.add_shape(
            MSO_AUTO_SHAPE_TYPE.RECTANGLE,
            Inches(0.55),
            Inches(1.42),
            Inches(12.2),
            Inches(0.02),
        )
        rule.fill.solid()
        rule.fill.fore_color.rgb = rgb(COLOR_BORDER)
        rule.line.fill.background()

    def add_footer(slide, left_text, right_text=""):
        add_textbox(slide, 0.55, 7.05, 6.6, 0.2, left_text, 9, color=COLOR_MUTED)
        if right_text:
            add_textbox(slide, 9.35, 7.05, 3.4, 0.2, right_text, 9, color=COLOR_MUTED, align=PP_ALIGN.RIGHT)

    def style_table(table, font_size):
        for row_index, row in enumerate(table.rows):
            for cell in row.cells:
                cell.vertical_anchor = MSO_ANCHOR.MIDDLE
                cell.margin_left = Inches(0.08)
                cell.margin_right = Inches(0.08)
                cell.margin_top = Inches(0.03)
                cell.margin_bottom = Inches(0.03)
                cell.fill.solid()
                cell.fill.fore_color.rgb = rgb(COLOR_CARD if row_index % 2 == 1 else COLOR_BG)
                if row_index == 0:
                    cell.fill.fore_color.rgb = rgb(COLOR_HEADER_FILL)
                cell.text_frame.word_wrap = True
                for paragraph in cell.text_frame.paragraphs:
                    paragraph.alignment = PP_ALIGN.LEFT
                    paragraph.line_spacing = 1.05
                    for run in paragraph.runs:
                        run.font.name = FONT_FAMILY
                        run.font.size = Pt(font_size)
                        run.font.color.rgb = rgb(COLOR_TEXT)
                        run.font.bold = row_index == 0

        for cell in table.rows[0].cells:
            cell.fill.fore_color.rgb = rgb(COLOR_HEADER_FILL)

    def set_table_column_widths(table, column_count, sticky_columns):
        total_width = 12.2
        sticky_width = 0.0
        if sticky_columns >= 1:
            first_width = min(3.0, max(2.1, total_width * 0.24))
            table.columns[0].width = Inches(first_width)
            sticky_width += first_width

        for column_index in range(1, sticky_columns):
            width = min(1.8, max(1.1, total_width * 0.12))
            table.columns[column_index].width = Inches(width)
            sticky_width += width

        remaining_columns = column_count - sticky_columns
        if remaining_columns <= 0:
            return

        remaining_width = max(2.8, total_width - sticky_width)
        per_width = remaining_width / remaining_columns
        for column_index in range(sticky_columns, column_count):
            table.columns[column_index].width = Inches(per_width)

    def add_title_slide():
        slide = prs.slides.add_slide(prs.slide_layouts[6])
        apply_slide_background(slide, COLOR_NAVY)

        accent = slide.shapes.add_shape(
            MSO_AUTO_SHAPE_TYPE.RECTANGLE,
            Inches(0.0),
            Inches(0.0),
            Inches(13.333),
            Inches(0.24),
        )
        accent.fill.solid()
        accent.fill.fore_color.rgb = rgb(COLOR_CYAN)
        accent.line.fill.background()

        add_textbox(slide, 0.72, 0.9, 11.5, 0.85, title, 28, color=(255, 255, 255), bold=True)
        add_textbox(
            slide,
            0.74,
            1.86,
            10.6,
            0.4,
            "Weekly experiment comparison deck generated from existing run artifacts",
            14,
            color=(226, 232, 240),
        )

        run_chip_top = 2.55
        for index, run in enumerate(runs[:4]):
            chip = slide.shapes.add_shape(
                MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE,
                Inches(0.74 + index * 3.03),
                Inches(run_chip_top),
                Inches(2.74),
                Inches(0.5),
            )
            chip.fill.solid()
            chip.fill.fore_color.rgb = rgb((30, 41, 59))
            chip.line.color.rgb = rgb((71, 85, 105))
            add_textbox(
                slide,
                0.94 + index * 3.03,
                run_chip_top + 0.12,
                2.36,
                0.2,
                run.label,
                12,
                color=(255, 255, 255),
                bold=True,
                align=PP_ALIGN.CENTER,
            )

        meta_box = slide.shapes.add_shape(
            MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE,
            Inches(0.72),
            Inches(4.0),
            Inches(6.1),
            Inches(2.25),
        )
        meta_box.fill.solid()
        meta_box.fill.fore_color.rgb = rgb((30, 41, 59))
        meta_box.line.color.rgb = rgb((71, 85, 105))
        add_textbox(slide, 1.0, 4.3, 5.5, 0.3, "Report scope", 16, color=(255, 255, 255), bold=True)
        add_bullet_box(
            slide,
            1.0,
            4.72,
            5.25,
            1.3,
            [
                f"Reference run: {runs[0].label}",
                f"Compared runs: {len(runs)}",
                f"Shared training metrics: {', '.join(metric_display_name(metric) for metric in metrics[:4])}",
            ],
            font_size=12,
            color=(226, 232, 240),
        )
        add_footer(
            slide,
            f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"Report name: {report_name}",
        )

    def add_run_info_slides():
        run_cards = [(run.label, build_run_profile_lines(run, all_varying_keys)) for run in runs]
        for group_index, run_group in enumerate(chunked(run_cards, 4), start=1):
            slide = prs.slides.add_slide(prs.slide_layouts[6])
            apply_slide_background(slide, COLOR_BG)
            add_slide_header(
                slide,
                "Experiment Setup and Run Context",
                "Keep the first --run as the reference baseline for meeting comparisons.",
                section_label="Run Setup",
            )

            if len(run_group) <= 2:
                positions = [(0.55, 1.72), (6.85, 1.72)]
                card_width = 5.9
                card_height = 3.15
            else:
                positions = [
                    (0.55, 1.7),
                    (6.85, 1.7),
                    (0.55, 4.25),
                    (6.85, 4.25),
                ]
                card_width = 5.9
                card_height = 2.18

            for (card_left, card_top), (card_title, body_lines) in zip(positions, run_group):
                add_card(slide, card_left, card_top, card_width, card_height, card_title, body_lines)

            add_footer(
                slide,
                "Run cards surface the main hyperparameters and monitoring settings without crowding a table.",
                f"Page group {group_index}",
            )

    def add_highlight_slide():
        slide = prs.slides.add_slide(prs.slide_layouts[6])
        apply_slide_background(slide, COLOR_BG)
        add_slide_header(
            slide,
            "Results Overview",
            "Headline findings are phrased for direct use in a weekly update.",
            section_label="Overview",
        )
        add_bullet_box(slide, 0.75, 1.8, 6.35, 3.3, headline_insights, font_size=15)

        primary_metric = metrics[0]
        if is_lower_better(primary_metric):
            best_row = min(summary_rows, key=lambda row: row[f"final_{primary_metric}"])
        else:
            best_row = max(summary_rows, key=lambda row: row[f"final_{primary_metric}"])

        add_metric_card(
            slide,
            7.45,
            1.9,
            2.45,
            1.3,
            "Best primary metric",
            best_row["label"],
            accent=COLOR_GOOD,
        )
        add_metric_card(
            slide,
            10.1,
            1.9,
            2.45,
            1.3,
            metric_display_name(primary_metric),
            format_metric_value(primary_metric, best_row[f"final_{primary_metric}"]),
            accent=COLOR_BLUE,
        )
        add_metric_card(
            slide,
            7.45,
            3.45,
            2.45,
            1.3,
            "Runs compared",
            str(len(runs)),
            accent=COLOR_CYAN,
        )
        add_metric_card(
            slide,
            10.1,
            3.45,
            2.45,
            1.3,
            "Metrics tracked",
            str(len(metrics) + len(selected_metric_keys)),
            accent=COLOR_CYAN,
        )

        add_footer(slide, "This slide replaces generic 'key takeaways' text with meeting-ready comparison statements.")

    def add_table_slides(title_text, subtitle_text, headers, rows, sticky_columns=1, max_total_columns=6, max_body_rows=8, font_size=None, footer_text=""):
        if len(headers) <= max_total_columns:
            column_groups = [list(range(len(headers)))]
        else:
            scrollable_indexes = list(range(sticky_columns, len(headers)))
            scrollable_chunk_size = max(1, max_total_columns - sticky_columns)
            column_groups = []
            for chunk in chunked(scrollable_indexes, scrollable_chunk_size):
                column_groups.append(list(range(sticky_columns)) + chunk)

        row_groups = list(chunked(rows, max_body_rows)) or [[]]
        total_slides = len(column_groups) * len(row_groups)
        slide_number = 0

        for row_group in row_groups:
            for column_group in column_groups:
                slide_number += 1
                slide = prs.slides.add_slide(prs.slide_layouts[6])
                apply_slide_background(slide, COLOR_BG)
                numbered_title = title_text if total_slides == 1 else f"{title_text} ({slide_number}/{total_slides})"
                add_slide_header(slide, numbered_title, subtitle_text, section_label="Comparison Table")

                subset_headers = [headers[index] for index in column_group]
                subset_rows = [
                    [shorten(str(row[index]), width=40, placeholder="...") for index in column_group]
                    for row in row_group
                ]
                local_font_size = font_size or estimate_table_font_size(
                    column_count=len(subset_headers),
                    row_count=len(subset_rows),
                )

                shape = slide.shapes.add_table(
                    len(subset_rows) + 1,
                    len(subset_headers),
                    Inches(0.55),
                    Inches(1.72),
                    Inches(12.2),
                    Inches(4.95),
                )
                table = shape.table

                for column_index, header in enumerate(subset_headers):
                    table.cell(0, column_index).text = header

                for row_index, row_values in enumerate(subset_rows, start=1):
                    for column_index, value in enumerate(row_values):
                        table.cell(row_index, column_index).text = value

                effective_sticky_columns = min(sticky_columns, len(subset_headers))
                set_table_column_widths(table, len(subset_headers), effective_sticky_columns)
                row_height = Inches(4.95 / max(2, len(subset_rows) + 1))
                for row in table.rows:
                    row.height = row_height
                style_table(table, local_font_size)

                add_footer(slide, footer_text)

    def add_curve_slides():
        figure_payloads = []
        for metric in metrics:
            best_row = min(summary_rows, key=lambda row: row[f"final_{metric}"]) if is_lower_better(metric) else max(summary_rows, key=lambda row: row[f"final_{metric}"])
            figure_payloads.append(
                {
                    "title": metric_display_name(metric),
                    "path": metric_figure_paths[metric],
                    "summary": f"Best final value: {best_row['label']} ({format_metric_value(metric, best_row[f'final_{metric}'])})",
                }
            )

        for page_index, group in enumerate(chunked(figure_payloads, 2), start=1):
            slide = prs.slides.add_slide(prs.slide_layouts[6])
            apply_slide_background(slide, COLOR_BG)
            add_slide_header(
                slide,
                f"Training Curves ({page_index}/{len(list(chunked(figure_payloads, 2)))})",
                "Curves use a consistent title pattern and highlight the final operating point.",
                section_label="Curves",
            )

            slot_positions = [(0.55, 1.72), (6.55, 1.72)]
            for (slot_left, slot_top), payload in zip(slot_positions, group):
                slide.shapes.add_picture(str(payload["path"]), Inches(slot_left), Inches(slot_top), width=Inches(5.65))
                add_textbox(slide, slot_left, 5.65, 5.65, 0.22, payload["title"], 13, bold=True)
                add_textbox(slide, slot_left, 5.95, 5.65, 0.28, payload["summary"], 10, color=COLOR_MUTED)

            add_footer(slide, "Use these pages for trend discussion; use tables for exact values.")

    def add_confusion_matrix_slides():
        confusion_runs = [run for run in runs if run.confusion_matrix_csv]
        for run in confusion_runs:
            figure_path = ensure_confusion_matrix_figure(report_dir / "figures", run)
            if not figure_path:
                continue

            slide = prs.slides.add_slide(prs.slide_layouts[6])
            apply_slide_background(slide, COLOR_BG)
            add_slide_header(
                slide,
                f"Confusion Matrix | {run.label}",
                "Selected-checkpoint error analysis pulled from existing evaluation outputs.",
                section_label="Error Analysis",
            )
            slide.shapes.add_picture(str(figure_path), Inches(0.68), Inches(1.8), width=Inches(6.6))

            bullets = []
            if run.selected_epoch is not None:
                bullets.append(f"Selected checkpoint epoch: {run.selected_epoch}")
            if "test_acc" in run.selected_metrics:
                bullets.append(
                    f"Selected test accuracy: {format_metric_value('test_acc', run.selected_metrics['test_acc'])}"
                )
            if "test_macro_f1" in run.selected_metrics:
                bullets.append(
                    f"Selected test macro F1: {format_metric_value('test_macro_f1', run.selected_metrics['test_macro_f1'])}"
                )
            confusion_pairs = top_confusion_pairs(run.confusion_matrix_csv)
            if confusion_pairs:
                bullets.extend(
                    [
                        f"Top confusion: true {true_label} predicted as {pred_label} ({count} samples)."
                        for count, true_label, pred_label in confusion_pairs[:2]
                    ]
                )
            add_card(slide, 7.7, 1.95, 4.55, 3.75, "Reading notes", bullets or ["Confusion matrix figure available, but no summary bullets were derived."])
            add_footer(slide, "This page gives a direct bridge from aggregate metrics to class-level failure modes.")

    def add_per_class_slides():
        for payload in per_class_figure_payloads:
            slide = prs.slides.add_slide(prs.slide_layouts[6])
            apply_slide_background(slide, COLOR_BG)
            add_slide_header(
                slide,
                f"Per-Class Analysis | {metric_display_name(payload['metric'])}",
                "Classwise selected-checkpoint metrics are grouped across runs for easier error analysis.",
                section_label="Per-Class",
            )
            slide.shapes.add_picture(str(payload["path"]), Inches(0.62), Inches(1.85), width=Inches(7.55))
            add_card(slide, 8.45, 1.98, 3.8, 3.45, "Key readout", payload["summary"], accent=COLOR_CYAN)
            add_footer(slide, "Use this page to explain which classes move together and which remain difficult.")

    def add_conclusion_slide():
        slide = prs.slides.add_slide(prs.slide_layouts[6])
        apply_slide_background(slide, COLOR_BG)
        add_slide_header(
            slide,
            "Conclusion",
            "Short, reusable wrap-up slide for weekly meetings.",
            section_label="Close",
        )
        add_bullet_box(slide, 0.8, 1.95, 7.3, 3.2, conclusion_lines, font_size=16)
        add_card(
            slide,
            8.55,
            2.1,
            3.7,
            2.3,
            "Recommended next use",
            [
                "Keep the baseline run first in the command line.",
                "Append new runs with readable labels.",
                "Reuse this deck directly for group meetings.",
            ],
            accent=COLOR_GOOD,
        )
        add_footer(slide, "The goal is to export a presentable deck, not a raw artifact dump.")

    add_title_slide()
    add_run_info_slides()
    add_highlight_slide()

    overview_headers, overview_rows = build_results_overview_rows(summary_rows, metrics)
    add_table_slides(
        "Result Snapshot",
        "Compact overview of the most important final metrics and selected checkpoint epoch.",
        overview_headers,
        overview_rows,
        sticky_columns=2,
        max_total_columns=7,
        max_body_rows=7,
        footer_text="This table is intentionally compact so it remains readable on projected slides.",
    )

    metric_headers, metric_rows = build_main_metric_rows(summary_rows, metrics, selected_metric_keys)
    add_table_slides(
        "Main Metrics Comparison",
        "Each row is a meeting-friendly metric statement rather than a raw logger dump.",
        metric_headers,
        metric_rows,
        sticky_columns=2,
        max_total_columns=6,
        max_body_rows=8,
        footer_text="Direction indicates whether higher or lower values are preferable.",
    )

    if varying_rows:
        config_headers = ["Config"] + [run.label for run in runs]
        config_table_rows = [[key, *values] for key, values in varying_rows]
        add_table_slides(
            "Config Differences",
            f"Showing the top {len(varying_rows)} differing settings for experiment traceability.",
            config_headers,
            config_table_rows,
            sticky_columns=1,
            max_total_columns=5,
            max_body_rows=7,
            font_size=10,
            footer_text="Long config lists are paginated automatically to avoid overflow and tiny fonts.",
        )

    add_curve_slides()
    add_confusion_matrix_slides()
    add_per_class_slides()
    add_conclusion_slide()

    ppt_path = report_dir / f"{report_name}.pptx"
    prs.save(ppt_path)
    return ppt_path


def main():
    args = parse_args()
    project_root = Path.cwd()
    run_specs = [parse_run_spec(spec) for spec in args.runs]
    report_name = args.report_name or make_default_report_name(run_specs)
    title = args.title or "Experiment Comparison Report"

    runs = [
        load_run_artifacts(args.results_dir, project_root, run_name, label)
        for run_name, label in run_specs
    ]
    metrics = determine_metrics(runs, explicit_metrics=args.metrics)
    if not metrics:
        raise ValueError("No shared numeric metrics were found across the selected runs.")

    selected_metric_keys = determine_selected_metric_keys(runs, metrics)
    report_dir, figures_dir = ensure_report_dirs(args.results_dir, report_name)
    summary_rows = build_summary_rows(runs, metrics, selected_metric_keys)
    selected_varying_keys, all_varying_keys = choose_varying_config_keys(
        runs, args.max_config_rows
    )
    varying_rows = build_varying_rows(runs, selected_varying_keys, args.max_config_rows)

    metric_figure_paths = {}
    for metric in metrics:
        metric_figure_paths[metric] = plot_metric(figures_dir, runs, metric)

    per_class_metric_keys = determine_available_per_class_metric_keys(runs)
    per_class_figure_payloads = []
    for metric_name in per_class_metric_keys:
        figure_path, class_names = plot_per_class_metric(figures_dir, runs, metric_name)
        per_class_figure_payloads.append(
            {
                "metric": metric_name,
                "path": figure_path,
                "summary": summarize_per_class_metric(runs, metric_name, class_names),
            }
        )

    headline_insights = build_headline_insights(runs, summary_rows, metrics)
    conclusion_lines = build_conclusion_lines(runs, summary_rows, metrics, per_class_metric_keys)

    summary_csv_path, config_csv_path, overview_path, presentation_summary_path, manifest_path = save_summary_outputs(
        report_dir=report_dir,
        title=title,
        runs=runs,
        metrics=metrics,
        selected_metric_keys=selected_metric_keys,
        summary_rows=summary_rows,
        varying_keys=all_varying_keys,
        headline_insights=headline_insights,
        conclusion_lines=conclusion_lines,
    )

    print(f"Report directory: {report_dir}")
    print(f"Summary CSV: {summary_csv_path}")
    print(f"Config CSV: {config_csv_path}")
    print(f"Overview Markdown: {overview_path}")
    print(f"Presentation Summary JSON: {presentation_summary_path}")
    print(f"Manifest JSON: {manifest_path}")
    for metric, path in metric_figure_paths.items():
        print(f"Figure ({metric}): {path}")
    for payload in per_class_figure_payloads:
        print(f"Per-class Figure ({payload['metric']}): {payload['path']}")

    if not args.skip_ppt:
        ppt_path = export_ppt(
            report_dir=report_dir,
            report_name=report_name,
            title=title,
            runs=runs,
            metrics=metrics,
            selected_metric_keys=selected_metric_keys,
            summary_rows=summary_rows,
            varying_rows=varying_rows,
            all_varying_keys=all_varying_keys,
            metric_figure_paths=metric_figure_paths,
            per_class_figure_payloads=per_class_figure_payloads,
            headline_insights=headline_insights,
            conclusion_lines=conclusion_lines,
        )
        print(f"PPTX: {ppt_path}")


if __name__ == "__main__":
    main()
