import copy
import csv
import json
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("results/matplotlib_cache")))

import matplotlib.pyplot as plt
import torch

from paper_plotting import (
    PAPER_BAR_FIGSIZE,
    PAPER_FIGSIZE,
    PAPER_HEATMAP_FIGSIZE,
    SPLIT_STYLES,
    annotate_bars,
    finish_bar_axis,
    finish_epoch_axis,
    mark_every,
    save_figure_pair,
    setup_paper_plot_style,
)

EARLY_STOPPING_METRICS = ("val_acc", "val_loss", "val_macro_f1")
PER_CLASS_COLORS = {
    "test_per_class_accuracy": "#2563eb",
    "test_per_class_precision": "#7c3aed",
    "test_per_class_recall": "#dc2626",
    "test_per_class_f1": "#16a34a",
}
PER_CLASS_TITLES = {
    "test_per_class_accuracy": "Selected Test Per-Class Accuracy",
    "test_per_class_precision": "Selected Test Per-Class Precision",
    "test_per_class_recall": "Selected Test Per-Class Recall",
    "test_per_class_f1": "Selected Test Per-Class F1",
}


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def setup_plot_style():
    setup_paper_plot_style()


def _compute_single_label_accuracy(predictions, targets):
    return (predictions == targets).sum().item() / max(1, targets.numel())


def _compute_multilabel_accuracy(predictions, targets):
    return (predictions == targets).sum().item() / max(1, targets.numel())


def train_one_epoch(model, loader, criterion, optimizer, device, task_type="single_label", threshold=0.5):
    model.train()
    total_loss = 0.0
    total_acc_numerator = 0.0
    total_acc_denominator = 0

    for images, labels in loader:
        images = images.to(device)
        if task_type == "multilabel":
            labels = labels.to(device=device, dtype=torch.float32)
        else:
            labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        if task_type == "multilabel":
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities >= threshold).to(dtype=labels.dtype)
            total_acc_numerator += (predictions == labels).sum().item()
            total_acc_denominator += labels.numel()
        else:
            predictions = logits.argmax(dim=1)
            total_acc_numerator += (predictions == labels).sum().item()
            total_acc_denominator += batch_size

    return {
        "loss": total_loss / max(1, len(loader.dataset)),
        "acc": total_acc_numerator / max(1, total_acc_denominator),
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device, task_type="single_label", threshold=0.5):
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    total_acc_numerator = 0.0
    total_acc_denominator = 0
    num_classes = None

    for images, labels in loader:
        images = images.to(device)
        if task_type == "multilabel":
            labels = labels.to(device=device, dtype=torch.float32)
        else:
            labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        if task_type == "multilabel":
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities >= threshold).to(dtype=labels.dtype)
            total_acc_numerator += (predictions == labels).sum().item()
            total_acc_denominator += labels.numel()
        else:
            if num_classes is None:
                num_classes = logits.shape[1]
            predictions = logits.argmax(dim=1)
            total_acc_numerator += (predictions == labels).sum().item()
            total_acc_denominator += batch_size

        all_predictions.append(predictions.cpu())
        all_targets.append(labels.cpu())

    predictions = torch.cat(all_predictions)
    targets = torch.cat(all_targets)

    if task_type == "multilabel":
        metrics = compute_multilabel_metrics(targets=targets, predictions=predictions)
    else:
        confusion_matrix = compute_confusion_matrix(
            targets=targets.tolist(),
            predictions=predictions.tolist(),
            num_classes=num_classes,
        )
        metrics = compute_classification_metrics(confusion_matrix)

    metrics.update(
        {
            "loss": total_loss / max(1, len(loader.dataset)),
            "acc": total_acc_numerator / max(1, total_acc_denominator),
        }
    )
    return metrics


def compute_confusion_matrix(targets, predictions, num_classes):
    matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for target, prediction in zip(targets, predictions):
        matrix[target, prediction] += 1
    return matrix


def compute_classification_metrics(confusion_matrix):
    matrix = confusion_matrix.to(torch.float32)
    true_positives = matrix.diag()
    predicted_positives = matrix.sum(dim=0)
    actual_positives = matrix.sum(dim=1)

    precision = torch.where(
        predicted_positives > 0,
        true_positives / predicted_positives,
        torch.zeros_like(true_positives),
    )
    recall = torch.where(
        actual_positives > 0,
        true_positives / actual_positives,
        torch.zeros_like(true_positives),
    )
    f1 = torch.where(
        (precision + recall) > 0,
        2 * precision * recall / (precision + recall),
        torch.zeros_like(precision),
    )
    per_class_accuracy = recall

    return {
        "macro_precision": precision.mean().item(),
        "macro_recall": recall.mean().item(),
        "macro_f1": f1.mean().item(),
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
        "per_class_accuracy": per_class_accuracy.tolist(),
    }


def compute_multilabel_metrics(targets, predictions):
    targets = targets.to(torch.float32)
    predictions = predictions.to(torch.float32)

    true_positives = (predictions * targets).sum(dim=0)
    predicted_positives = predictions.sum(dim=0)
    actual_positives = targets.sum(dim=0)

    precision = torch.where(
        predicted_positives > 0,
        true_positives / predicted_positives,
        torch.zeros_like(true_positives),
    )
    recall = torch.where(
        actual_positives > 0,
        true_positives / actual_positives,
        torch.zeros_like(true_positives),
    )
    f1 = torch.where(
        (precision + recall) > 0,
        2 * precision * recall / (precision + recall),
        torch.zeros_like(precision),
    )
    per_class_accuracy = (predictions == targets).to(torch.float32).mean(dim=0)
    subset_accuracy = (predictions == targets).all(dim=1).to(torch.float32).mean()

    return {
        "macro_precision": precision.mean().item(),
        "macro_recall": recall.mean().item(),
        "macro_f1": f1.mean().item(),
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
        "per_class_accuracy": per_class_accuracy.tolist(),
        "subset_accuracy": subset_accuracy.item(),
    }


@torch.no_grad()
def evaluate_with_details(model, loader, criterion, device, num_classes, task_type="single_label", threshold=0.5):
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    total_acc_numerator = 0.0
    total_acc_denominator = 0

    for images, labels in loader:
        images = images.to(device)
        if task_type == "multilabel":
            labels = labels.to(device=device, dtype=torch.float32)
        else:
            labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)
        if task_type == "multilabel":
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities >= threshold).to(dtype=labels.dtype)
            total_acc_numerator += (predictions == labels).sum().item()
            total_acc_denominator += labels.numel()
        else:
            predictions = logits.argmax(dim=1)
            total_acc_numerator += (predictions == labels).sum().item()
            total_acc_denominator += labels.size(0)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size

        all_predictions.append(predictions.cpu())
        all_targets.append(labels.cpu())

    predictions = torch.cat(all_predictions)
    targets = torch.cat(all_targets)
    if task_type == "multilabel":
        metrics = compute_multilabel_metrics(targets=targets, predictions=predictions)
        metrics.update(
            {
                "loss": total_loss / max(1, len(loader.dataset)),
                "acc": total_acc_numerator / max(1, total_acc_denominator),
                "confusion_matrix": None,
            }
        )
    else:
        confusion_matrix = compute_confusion_matrix(
            targets=targets.tolist(),
            predictions=predictions.tolist(),
            num_classes=num_classes,
        )
        metrics = compute_classification_metrics(confusion_matrix)
        metrics.update(
            {
                "loss": total_loss / max(1, len(loader.dataset)),
                "acc": total_acc_numerator / max(1, total_acc_denominator),
                "confusion_matrix": confusion_matrix.tolist(),
            }
        )
    return metrics


def get_metric_mode(metric_name):
    if metric_name.endswith("loss"):
        return "min"
    return "max"


def is_metric_improved(current_value, best_value, metric_name, min_delta):
    mode = get_metric_mode(metric_name)
    if best_value is None:
        return True
    if mode == "min":
        return current_value < (best_value - min_delta)
    return current_value > (best_value + min_delta)


def maybe_update_early_stopping(
    model,
    epoch_metrics,
    monitor_metric,
    min_delta,
    best_metric_value,
    best_epoch,
    best_state_dict,
    patience_counter,
):
    current_value = epoch_metrics[monitor_metric]
    improved = is_metric_improved(current_value, best_metric_value, monitor_metric, min_delta)
    if improved:
        return (
            current_value,
            epoch_metrics["epoch"],
            copy.deepcopy(model.state_dict()),
            0,
            True,
        )
    return best_metric_value, best_epoch, best_state_dict, patience_counter + 1, False


def save_confusion_matrix_csv(confusion_matrix, class_names, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true/pred", *class_names])
        for class_name, row in zip(class_names, confusion_matrix):
            writer.writerow([class_name, *row])


def plot_confusion_matrix(confusion_matrix, class_names, path, title):
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix = torch.tensor(confusion_matrix, dtype=torch.float32)

    setup_plot_style()
    figure, axis = plt.subplots(figsize=PAPER_HEATMAP_FIGSIZE)
    heatmap = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(range(len(class_names)))
    axis.set_yticks(range(len(class_names)))
    axis.set_xticklabels(class_names, rotation=45, ha="right")
    axis.set_yticklabels(class_names)
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_title(title)

    threshold = matrix.max().item() * 0.55 if matrix.numel() > 0 else 0.0
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = int(matrix[row_index, column_index].item())
            text_color = "white" if value > threshold else "#0f172a"
            axis.text(
                column_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                color=text_color,
                fontsize=9,
            )

    figure.colorbar(heatmap, ax=axis, fraction=0.046, pad=0.04)
    save_figure_pair(figure, path)
    plt.close(figure)


def _draw_selected_epoch(axis, selected_epoch):
    if selected_epoch is None:
        return
    axis.axvline(
        x=selected_epoch,
        color="#475569",
        linestyle="--",
        linewidth=1.3,
        alpha=0.75,
        label=f"Selected epoch ({selected_epoch})",
    )


def _plot_history_lines(history, path, title, metric_name, series_specs, selected_epoch=None):
    setup_plot_style()
    epochs = [int(row["epoch"]) for row in history]

    figure, axis = plt.subplots(figsize=PAPER_FIGSIZE)
    marker_interval = mark_every(len(epochs))
    for key, label, split_name, scale in series_specs:
        values = [float(row[key]) * scale for row in history]
        split_style = SPLIT_STYLES[split_name]
        axis.plot(
            epochs,
            values,
            linewidth=2.1,
            linestyle=split_style["linestyle"],
            marker=split_style["marker"],
            markersize=4.0,
            markevery=marker_interval,
            label=label,
            color=split_style["color"],
        )

    _draw_selected_epoch(axis, selected_epoch)
    finish_epoch_axis(axis, metric_name=metric_name, title=title)
    save_figure_pair(figure, path)
    plt.close(figure)


def plot_selected_per_class_metrics(selected_model_metrics, class_names, figure_dir, run_name, title_prefix):
    figure_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = {}

    for metric_name, metric_title in PER_CLASS_TITLES.items():
        values = selected_model_metrics.get(metric_name)
        if not isinstance(values, list) or not values:
            continue

        setup_plot_style()
        figure, axis = plt.subplots(figsize=PAPER_BAR_FIGSIZE)
        x_positions = list(range(len(values)))
        plotted_values = [float(value) * 100.0 for value in values]
        color = PER_CLASS_COLORS.get(metric_name, "#2563eb")
        bars = axis.bar(x_positions, plotted_values, color=color, alpha=0.9)

        axis.set_xticks(x_positions)
        axis.set_xticklabels(class_names, rotation=35, ha="right", fontsize=8)
        y_max = max(100.0, max(plotted_values) * 1.12 if plotted_values else 100.0)
        finish_bar_axis(axis, title=f"{title_prefix} {metric_title}", y_max=y_max)
        annotate_bars(axis, bars, plotted_values)

        figure_path = figure_dir / f"{run_name}_{metric_name}.png"
        save_figure_pair(figure, figure_path)
        plt.close(figure)
        saved_paths[metric_name] = figure_path

    return saved_paths


def save_best_checkpoint(path, model, model_config, device, args, best_epoch, best_metric_value):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "run_name": args.run_name,
            "model_state_dict": model.state_dict(),
            "model_config": model_config,
            "device": str(device),
            "best_epoch": best_epoch,
            "best_metric_name": args.early_stopping_metric,
            "best_metric_value": best_metric_value,
            "args": vars(args),
        },
        path,
    )


def save_metrics_csv(history, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in history:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def save_config_json(args, model_config, train_size, val_size, test_size, device, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset_image_size = (
        args.image_size
        or model_config.get("img_size")
        or model_config.get("image_size")
    )
    config = {
        "command": " ".join(sys.argv),
        "device": str(device),
        "dataset": {
            "name": getattr(args, "dataset", "cifar10"),
            "train_size": train_size,
            "val_size": val_size,
            "test_size": test_size,
            "data_dir": str(args.data_dir),
            "image_size": dataset_image_size,
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "lr_plateau_patience": getattr(args, "lr_plateau_patience", None),
            "lr_plateau_factor": getattr(args, "lr_plateau_factor", None),
            "lr_plateau_min_lr": getattr(args, "lr_plateau_min_lr", None),
            "train_subset": args.train_subset,
            "val_subset": args.val_subset,
            "test_subset": args.test_subset,
            "val_ratio": args.val_ratio,
            "seed": args.seed,
            "num_workers": args.num_workers,
            "early_stopping_patience": args.early_stopping_patience,
            "early_stopping_min_delta": args.early_stopping_min_delta,
            "early_stopping_metric": args.early_stopping_metric,
        },
        "synthetic_dataset": {
            "train_size": getattr(args, "synthetic_train_size", None),
            "val_size": getattr(args, "synthetic_val_size", None),
            "test_size": getattr(args, "synthetic_test_size", None),
            "line_width": getattr(args, "synthetic_line_width", None),
            "noise_std": getattr(args, "synthetic_noise_std", None),
            "max_stripes": getattr(args, "synthetic_max_stripes", None),
        },
        "cadb_dataset": {
            "cadb_root": str(getattr(args, "cadb_root", "") or ""),
            "test_ratio": getattr(args, "cadb_test_ratio", None),
            "label_mode": getattr(args, "cadb_label_mode", None),
            "balance_mode": getattr(args, "cadb_balance_mode", None),
        },
        "model": model_config,
        "outputs": {
            "experiment_name": getattr(args, "experiment_name", None),
            "results_dir": str(args.results_dir),
            "checkpoint_dir": str(args.checkpoint_dir),
            "run_name": args.run_name,
        },
    }

    with path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)


def save_summary_json(args, history, path, early_stopping_info, selected_model_metrics):
    path.parent.mkdir(parents=True, exist_ok=True)
    best_val_key = "val_macro_f1" if "val_macro_f1" in history[0] else "val_acc"
    best_val_epoch = max(history, key=lambda row: row[best_val_key])
    summary = {
        "best_val_epoch": best_val_epoch["epoch"],
        "best_val_acc": best_val_epoch["val_acc"],
        "final_epoch": history[-1],
        "selected_model": selected_model_metrics,
        "early_stopping": early_stopping_info,
        "config": vars(args),
        "test_evaluation_protocol": "selected_checkpoint_only",
    }

    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=str)


def plot_curves(history, figure_dir, run_name, title_prefix, selected_epoch=None):
    figure_dir.mkdir(parents=True, exist_ok=True)
    loss_path = figure_dir / f"{run_name}_loss.png"
    acc_path = figure_dir / f"{run_name}_accuracy.png"
    _plot_history_lines(
        history=history,
        path=loss_path,
        title=f"{title_prefix} Loss",
        metric_name="val_loss",
        series_specs=[
            ("train_loss", "Train", "train", 1.0),
            ("val_loss", "Validation", "val", 1.0),
        ],
        selected_epoch=selected_epoch,
    )
    _plot_history_lines(
        history=history,
        path=acc_path,
        title=f"{title_prefix} Accuracy",
        metric_name="val_acc",
        series_specs=[
            ("train_acc", "Train", "train", 100.0),
            ("val_acc", "Validation", "val", 100.0),
        ],
        selected_epoch=selected_epoch,
    )

    curve_paths = {
        "loss": loss_path,
        "accuracy": acc_path,
    }
    if any("val_macro_f1" in row for row in history):
        macro_f1_path = figure_dir / f"{run_name}_macro_f1.png"
        series_specs = []
        if all("val_macro_f1" in row for row in history):
            series_specs.append(("val_macro_f1", "Validation", "val", 100.0))
        if series_specs:
            _plot_history_lines(
                history=history,
                path=macro_f1_path,
                title=f"{title_prefix} Macro F1",
                metric_name="val_macro_f1",
                series_specs=series_specs,
                selected_epoch=selected_epoch,
            )
            curve_paths["macro_f1"] = macro_f1_path

    return curve_paths
