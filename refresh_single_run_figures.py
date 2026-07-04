import argparse
import csv
import json
from pathlib import Path

from datasets.cadb_data import CADB_ELEMENT_LABELS, SCENE_CATEGORIES
from experiment_utils import (
    plot_confusion_matrix,
    plot_curves,
    plot_selected_per_class_metrics,
)
from models.registry import EXPERIMENT_REGISTRY, get_dataset_display_name
from result_paths import build_run_artifact_paths, resolve_run_artifact_paths


CIFAR10_LABELS = [
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Refresh per-run figures from saved metrics/config/summary artifacts without retraining."
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--experiment-name", type=str, required=True)
    parser.add_argument(
        "--run-name",
        action="append",
        default=None,
        help="Optional run name to refresh. Repeatable. If omitted, all runs inside the experiment are refreshed.",
    )
    return parser.parse_args()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload: dict):
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


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


def load_confusion_matrix_csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        rows = list(reader)
    class_names = rows[0][1:]
    matrix = []
    for row in rows[1:]:
        matrix.append([int(value) for value in row[1:]])
    return class_names, matrix


def parse_command_model_name(command: str):
    tokens = str(command).split()
    if "--model" in tokens:
        index = tokens.index("--model")
        if index + 1 < len(tokens):
            return tokens[index + 1].strip()
    return None


def discover_run_names(results_dir: Path, experiment_name: str):
    metrics_root = results_dir / experiment_name / "metrics"
    run_names = []
    for summary_path in sorted(metrics_root.glob("*/*_summary.json")):
        run_names.append(summary_path.name.removesuffix("_summary.json"))
    return run_names


def infer_model_name(config: dict, summary: dict, artifact_paths: dict):
    selected_model = summary.get("selected_model", {}) if isinstance(summary.get("selected_model"), dict) else {}
    model_name = selected_model.get("model_name")
    if model_name:
        return str(model_name)
    model_name = parse_command_model_name(str(config.get("command", "")))
    if model_name:
        return model_name
    metrics_dir = artifact_paths.get("metrics_path")
    if metrics_dir is not None:
        return Path(metrics_dir).parent.name
    return "unknown_model"


def infer_title_prefix(dataset_name: str, model_name: str):
    spec = EXPERIMENT_REGISTRY.get(model_name)
    model_title = spec.plot_title_prefix if spec else model_name.replace("_", " ").title()
    return f"{get_dataset_display_name(dataset_name)} {model_title}"


def infer_label_names(dataset_name: str, selected_model: dict, confusion_csv_path: Path | None):
    label_names = selected_model.get("label_names")
    if isinstance(label_names, list) and label_names:
        return [str(item) for item in label_names]
    if confusion_csv_path is not None and confusion_csv_path.exists():
        class_names, _ = load_confusion_matrix_csv(confusion_csv_path)
        return class_names
    if dataset_name == "cifar10":
        return list(CIFAR10_LABELS)
    if dataset_name in {"cadb_orientation", "synthetic_orientation", "synthetic_orientation_clean", "synthetic_orientation_hard"}:
        return ["horizontal", "vertical"]
    if dataset_name == "cadb_scene":
        return list(SCENE_CATEGORIES)
    if dataset_name == "cadb_elements":
        return list(CADB_ELEMENT_LABELS)

    for key in (
        "test_per_class_accuracy",
        "test_per_class_precision",
        "test_per_class_recall",
        "test_per_class_f1",
    ):
        values = selected_model.get(key)
        if isinstance(values, list) and values:
            return [f"class_{index}" for index in range(len(values))]
    return []


def refresh_run(results_dir: Path, experiment_name: str, run_name: str):
    artifact_paths = resolve_run_artifact_paths(results_dir, run_name, experiment_name=experiment_name)
    metrics_path = artifact_paths["metrics_path"]
    config_path = artifact_paths["config_path"]
    summary_path = artifact_paths["summary_path"]
    confusion_csv_path = artifact_paths["confusion_csv_path"]

    if metrics_path is None or config_path is None or summary_path is None:
        raise FileNotFoundError(f"Missing required artifacts for run: {run_name}")

    config = load_json(config_path)
    summary = load_json(summary_path)
    history = load_history(metrics_path)
    selected_model = summary.get("selected_model", {}) if isinstance(summary.get("selected_model"), dict) else {}

    dataset_value = config.get("dataset", {})
    dataset_name = dataset_value.get("name") if isinstance(dataset_value, dict) else str(dataset_value)
    model_name = infer_model_name(config, summary, artifact_paths)
    write_paths = build_run_artifact_paths(
        results_dir=results_dir,
        checkpoint_dir=Path("checkpoints"),
        model_name=model_name,
        run_name=run_name,
        experiment_name=experiment_name,
        dataset_name=dataset_name,
    )
    figure_dir = write_paths["figures_dir"]
    confusion_figure_path = write_paths["confusion_figure_path"]
    title_prefix = infer_title_prefix(dataset_name, model_name)
    selected_epoch = selected_model.get("epoch")
    curve_paths = plot_curves(
        history=history,
        figure_dir=figure_dir,
        run_name=run_name,
        title_prefix=title_prefix,
        selected_epoch=int(selected_epoch) if isinstance(selected_epoch, (int, float)) else None,
    )

    label_names = infer_label_names(dataset_name, selected_model, confusion_csv_path)
    if label_names:
        selected_model["label_names"] = label_names

    if confusion_csv_path is not None and confusion_csv_path.exists() and confusion_figure_path is not None:
        class_names, matrix = load_confusion_matrix_csv(confusion_csv_path)
        plot_confusion_matrix(
            confusion_matrix=matrix,
            class_names=class_names,
            path=confusion_figure_path,
            title=f"{title_prefix} Test Confusion Matrix",
        )
        selected_model["test_confusion_matrix_csv"] = str(confusion_csv_path)
        selected_model["test_confusion_matrix_figure"] = str(confusion_figure_path)

    per_class_figure_paths = plot_selected_per_class_metrics(
        selected_model_metrics=selected_model,
        class_names=label_names,
        figure_dir=figure_dir,
        run_name=run_name,
        title_prefix=title_prefix,
    )
    if per_class_figure_paths:
        selected_model["per_class_figure_paths"] = {
            key: str(path) for key, path in per_class_figure_paths.items()
        }

    selected_model["loss_figure"] = str(curve_paths["loss"])
    selected_model["accuracy_figure"] = str(curve_paths["accuracy"])
    if "macro_f1" in curve_paths:
        selected_model["macro_f1_figure"] = str(curve_paths["macro_f1"])

    summary["selected_model"] = selected_model
    save_json(summary_path, summary)
    print(f"Refreshed: {run_name}")


def main():
    args = parse_args()
    run_names = args.run_name or discover_run_names(args.results_dir, args.experiment_name)
    if not run_names:
        raise FileNotFoundError(f"No runs found under experiment: {args.experiment_name}")

    for run_name in run_names:
        refresh_run(args.results_dir, args.experiment_name, run_name)


if __name__ == "__main__":
    main()
