import argparse
import copy
import csv
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("results/matplotlib_cache")))

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from vit import ViT
from vit_rope import ViTRoPE


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
EARLY_STOPPING_METRICS = ("val_acc", "val_loss")


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


def make_subset(dataset, subset_size, seed):
    if subset_size is None or subset_size >= len(dataset):
        return dataset

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:subset_size].tolist()
    return Subset(dataset, indices)


def split_train_val_indices(dataset_size, val_ratio, seed):
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1.")

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(dataset_size, generator=generator).tolist()
    val_size = max(1, int(dataset_size * val_ratio))
    if val_size >= dataset_size:
        raise ValueError("val_ratio leaves no training samples.")

    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    return train_indices, val_indices


def build_dataloaders(
    data_dir,
    batch_size,
    train_subset,
    val_subset,
    test_subset,
    num_workers,
    seed,
    val_ratio,
):
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )

    full_train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=transform_train,
    )
    full_val_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=False,
        transform=transform_test,
    )
    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=transform_test,
    )

    train_indices, val_indices = split_train_val_indices(
        dataset_size=len(full_train_dataset),
        val_ratio=val_ratio,
        seed=seed,
    )
    train_dataset = Subset(full_train_dataset, train_indices)
    val_dataset = Subset(full_val_dataset, val_indices)

    train_dataset = make_subset(train_dataset, train_subset, seed)
    val_dataset = make_subset(val_dataset, val_subset, seed)
    test_dataset = make_subset(test_dataset, test_subset, seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader, start=1):
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += batch_size

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += batch_size

    return total_loss / total, correct / total


def unwrap_dataset(dataset):
    current = dataset
    while isinstance(current, Subset):
        current = current.dataset
    return current


def get_class_names(dataset):
    base_dataset = unwrap_dataset(dataset)
    if hasattr(base_dataset, "classes"):
        return list(base_dataset.classes)
    return [str(index) for index in range(len(base_dataset))]


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


@torch.no_grad()
def evaluate_with_details(model, loader, criterion, device, num_classes):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)
        predictions = logits.argmax(dim=1)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        correct += (predictions == labels).sum().item()
        total += batch_size

        all_predictions.append(predictions.cpu())
        all_targets.append(labels.cpu())

    predictions = torch.cat(all_predictions)
    targets = torch.cat(all_targets)
    confusion_matrix = compute_confusion_matrix(
        targets=targets.tolist(),
        predictions=predictions.tolist(),
        num_classes=num_classes,
    )
    metrics = compute_classification_metrics(confusion_matrix)
    metrics.update(
        {
            "loss": total_loss / total,
            "acc": correct / total,
            "confusion_matrix": confusion_matrix.tolist(),
        }
    )
    return metrics


def make_run_name():
    return datetime.now().strftime("cifar10_vit_%Y%m%d_%H%M%S")


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

    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, cmap="Blues")
    plt.colorbar()
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    plt.yticks(range(len(class_names)), class_names)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


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
    fieldnames = [
        "epoch",
        "train_loss",
        "train_acc",
        "val_loss",
        "val_acc",
        "test_loss",
        "test_acc",
    ]

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def save_config_json(args, model_config, train_size, val_size, test_size, device, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "command": " ".join(sys.argv),
        "device": str(device),
        "dataset": {
            "name": "CIFAR-10",
            "train_size": train_size,
            "val_size": val_size,
            "test_size": test_size,
            "data_dir": str(args.data_dir),
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
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
        "model": model_config,
        "outputs": {
            "results_dir": str(args.results_dir),
            "checkpoint_dir": str(args.checkpoint_dir),
            "run_name": args.run_name,
        },
    }

    with path.open("w") as f:
        json.dump(config, f, indent=2)


def save_summary_json(args, history, path, early_stopping_info, selected_model_metrics):
    path.parent.mkdir(parents=True, exist_ok=True)
    best_val_epoch = max(history, key=lambda row: row["val_acc"])
    best_test_epoch = max(history, key=lambda row: row["test_acc"])
    summary = {
        "best_val_epoch": best_val_epoch["epoch"],
        "best_val_acc": best_val_epoch["val_acc"],
        "best_test_epoch": best_test_epoch["epoch"],
        "best_test_acc": best_test_epoch["test_acc"],
        "final_epoch": history[-1],
        "selected_model": selected_model_metrics,
        "early_stopping": early_stopping_info,
        "config": vars(args),
    }

    with path.open("w") as f:
        json.dump(summary, f, indent=2, default=str)


def plot_curves(history, figure_dir, run_name, title_prefix="CIFAR-10 ViT"):
    figure_dir.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, [row["train_loss"] for row in history], label="train loss")
    plt.plot(epochs, [row["val_loss"] for row in history], label="val loss")
    plt.plot(epochs, [row["test_loss"] for row in history], label="test loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{title_prefix} Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    loss_path = figure_dir / f"{run_name}_loss.png"
    plt.savefig(loss_path, dpi=150)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, [row["train_acc"] * 100 for row in history], label="train acc")
    plt.plot(epochs, [row["val_acc"] * 100 for row in history], label="val acc")
    plt.plot(epochs, [row["test_acc"] * 100 for row in history], label="test acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(f"{title_prefix} Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    acc_path = figure_dir / f"{run_name}_accuracy.png"
    plt.savefig(acc_path, dpi=150)
    plt.close()

    return loss_path, acc_path


def parse_args():
    parser = argparse.ArgumentParser(description="Train the small ViT on CIFAR-10.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--train-subset", type=int, default=None)
    parser.add_argument("--val-subset", type=int, default=None)
    parser.add_argument("--test-subset", type=int, default=None)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embedding-dropout", type=float, default=0.0)
    parser.add_argument("--attention-dropout", type=float, default=0.0)
    parser.add_argument("--projection-dropout", type=float, default=0.0)
    parser.add_argument("--mlp-dropout", type=float, default=0.0)
    parser.add_argument(
        "--model-variant",
        type=str,
        choices=("baseline", "rope"),
        default="baseline",
    )
    parser.add_argument("--rope-base", type=float, default=10000.0)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument(
        "--early-stopping-metric",
        type=str,
        choices=EARLY_STOPPING_METRICS,
        default="val_acc",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    args.run_name = args.run_name or make_run_name()

    device = get_device()
    print(f"Using device: {device}")
    print(f"Run name: {args.run_name}")
    print(f"Epochs: {args.epochs}")
    if args.early_stopping_patience is not None:
        print(
            "Early stopping: "
            f"metric={args.early_stopping_metric}, "
            f"patience={args.early_stopping_patience}, "
            f"min_delta={args.early_stopping_min_delta}"
        )

    train_loader, val_loader, test_loader = build_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        train_subset=args.train_subset,
        val_subset=args.val_subset,
        test_subset=args.test_subset,
        num_workers=args.num_workers,
        seed=args.seed,
        val_ratio=args.val_ratio,
    )

    model_config = {
        "img_size": 32,
        "patch_size": 4,
        "in_channels": 3,
        "num_classes": 10,
        "embed_dim": 128,
        "num_blocks": 4,
        "num_heads": 4,
        "mlp_hidden_dim": 512,
        "embedding_dropout": args.embedding_dropout,
        "attention_dropout": args.attention_dropout,
        "projection_dropout": args.projection_dropout,
        "mlp_dropout": args.mlp_dropout,
    }
    if args.model_variant == "rope":
        model_config["rope_base"] = args.rope_base
        model = ViTRoPE(**model_config).to(device)
    else:
        model = ViT(**model_config).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    history = []
    best_metric_value = None
    best_epoch = None
    best_state_dict = None
    patience_counter = 0
    stopped_early = False
    class_names = get_class_names(test_loader.dataset)
    num_classes = len(class_names)
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        val_loss, val_acc = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )
        test_loss, test_acc = evaluate(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
        )
        print(
            f"  train loss={train_loss:.4f} acc={train_acc * 100:.2f}% | "
            f"val loss={val_loss:.4f} acc={val_acc * 100:.2f}% | "
            f"test loss={test_loss:.4f} acc={test_acc * 100:.2f}%"
        )
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
        }
        history.append(epoch_metrics)

        best_metric_value, best_epoch, best_state_dict, patience_counter, improved = maybe_update_early_stopping(
            model=model,
            epoch_metrics=epoch_metrics,
            monitor_metric=args.early_stopping_metric,
            min_delta=args.early_stopping_min_delta,
            best_metric_value=best_metric_value,
            best_epoch=best_epoch,
            best_state_dict=best_state_dict,
            patience_counter=patience_counter,
        )
        if args.early_stopping_patience is not None:
            status = "improved" if improved else f"no improvement ({patience_counter}/{args.early_stopping_patience})"
            print(
                f"  early stopping monitor {args.early_stopping_metric}="
                f"{epoch_metrics[args.early_stopping_metric]:.4f} -> {status}"
            )
            if patience_counter >= args.early_stopping_patience:
                stopped_early = True
                print(f"Early stopping triggered at epoch {epoch}.")
                break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    selected_val_metrics = evaluate_with_details(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
        num_classes=num_classes,
    )
    selected_test_metrics = evaluate_with_details(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        num_classes=num_classes,
    )
    print(
        f"Selected checkpoint epoch {best_epoch}: "
        f"val loss={selected_val_metrics['loss']:.4f} acc={selected_val_metrics['acc'] * 100:.2f}% | "
        f"test loss={selected_test_metrics['loss']:.4f} acc={selected_test_metrics['acc'] * 100:.2f}% | "
        f"test macro_f1={selected_test_metrics['macro_f1']:.4f}"
    )

    metrics_dir = args.results_dir / "metrics"
    figure_dir = args.results_dir / "figures"
    checkpoint_dir = args.checkpoint_dir
    metrics_path = metrics_dir / f"{args.run_name}_metrics.csv"
    config_path = metrics_dir / f"{args.run_name}_config.json"
    summary_path = metrics_dir / f"{args.run_name}_summary.json"
    confusion_csv_path = metrics_dir / f"{args.run_name}_test_confusion_matrix.csv"
    confusion_figure_path = figure_dir / f"{args.run_name}_test_confusion_matrix.png"
    checkpoint_path = checkpoint_dir / f"{args.run_name}_best.pt"
    loss_path, acc_path = plot_curves(history, figure_dir, args.run_name)
    save_confusion_matrix_csv(
        confusion_matrix=selected_test_metrics["confusion_matrix"],
        class_names=class_names,
        path=confusion_csv_path,
    )
    plot_confusion_matrix(
        confusion_matrix=selected_test_metrics["confusion_matrix"],
        class_names=class_names,
        path=confusion_figure_path,
        title=f"{args.run_name} Test Confusion Matrix",
    )
    save_best_checkpoint(
        path=checkpoint_path,
        model=model,
        model_config=model_config,
        device=device,
        args=args,
        best_epoch=best_epoch,
        best_metric_value=best_metric_value,
    )
    save_metrics_csv(history, metrics_path)
    save_config_json(
        args=args,
        model_config=model_config,
        train_size=len(train_loader.dataset),
        val_size=len(val_loader.dataset),
        test_size=len(test_loader.dataset),
        device=device,
        path=config_path,
    )
    early_stopping_info = {
        "enabled": args.early_stopping_patience is not None,
        "metric": args.early_stopping_metric,
        "mode": get_metric_mode(args.early_stopping_metric),
        "patience": args.early_stopping_patience,
        "min_delta": args.early_stopping_min_delta,
        "stopped_early": stopped_early,
        "best_epoch": best_epoch,
        "best_metric_value": best_metric_value,
        "epochs_completed": len(history),
    }
    selected_model_metrics = {
        "epoch": best_epoch,
        "val_loss": selected_val_metrics["loss"],
        "val_acc": selected_val_metrics["acc"],
        "val_macro_precision": selected_val_metrics["macro_precision"],
        "val_macro_recall": selected_val_metrics["macro_recall"],
        "val_macro_f1": selected_val_metrics["macro_f1"],
        "test_loss": selected_test_metrics["loss"],
        "test_acc": selected_test_metrics["acc"],
        "test_macro_precision": selected_test_metrics["macro_precision"],
        "test_macro_recall": selected_test_metrics["macro_recall"],
        "test_macro_f1": selected_test_metrics["macro_f1"],
        "test_per_class_accuracy": selected_test_metrics["per_class_accuracy"],
        "test_per_class_precision": selected_test_metrics["per_class_precision"],
        "test_per_class_recall": selected_test_metrics["per_class_recall"],
        "test_per_class_f1": selected_test_metrics["per_class_f1"],
        "test_confusion_matrix_csv": str(confusion_csv_path),
        "test_confusion_matrix_figure": str(confusion_figure_path),
        "checkpoint_path": str(checkpoint_path),
    }
    save_summary_json(
        args,
        history,
        summary_path,
        early_stopping_info,
        selected_model_metrics,
    )

    print(f"Saved metrics: {metrics_path}")
    print(f"Saved config: {config_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved loss plot: {loss_path}")
    print(f"Saved accuracy plot: {acc_path}")
    print(f"Saved confusion matrix CSV: {confusion_csv_path}")
    print(f"Saved confusion matrix figure: {confusion_figure_path}")
    print(f"Saved best checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
