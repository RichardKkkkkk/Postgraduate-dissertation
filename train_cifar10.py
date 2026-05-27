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


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
EARLY_STOPPING_METRICS = ("test_acc", "test_loss")


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


def build_dataloaders(data_dir, batch_size, train_subset, test_subset, num_workers, seed):
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

    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=transform_train,
    )
    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=transform_test,
    )

    train_dataset = make_subset(train_dataset, train_subset, seed)
    test_dataset = make_subset(test_dataset, test_subset, seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
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

    return train_loader, test_loader


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


def save_metrics_csv(history, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["epoch", "train_loss", "train_acc", "test_loss", "test_acc"]

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def save_config_json(args, model_config, train_size, test_size, device, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "command": " ".join(sys.argv),
        "device": str(device),
        "dataset": {
            "name": "CIFAR-10",
            "train_size": train_size,
            "test_size": test_size,
            "data_dir": str(args.data_dir),
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "train_subset": args.train_subset,
            "test_subset": args.test_subset,
            "seed": args.seed,
            "num_workers": args.num_workers,
            "early_stopping_patience": args.early_stopping_patience,
            "early_stopping_min_delta": args.early_stopping_min_delta,
            "early_stopping_metric": args.early_stopping_metric,
        },
        "model": model_config,
        "outputs": {
            "results_dir": str(args.results_dir),
            "run_name": args.run_name,
        },
    }

    with path.open("w") as f:
        json.dump(config, f, indent=2)


def save_summary_json(args, history, path, early_stopping_info):
    path.parent.mkdir(parents=True, exist_ok=True)
    best_epoch = max(history, key=lambda row: row["test_acc"])
    summary = {
        "best_epoch": best_epoch["epoch"],
        "best_test_acc": best_epoch["test_acc"],
        "final_epoch": history[-1],
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
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--train-subset", type=int, default=None)
    parser.add_argument("--test-subset", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embedding-dropout", type=float, default=0.0)
    parser.add_argument("--attention-dropout", type=float, default=0.0)
    parser.add_argument("--projection-dropout", type=float, default=0.0)
    parser.add_argument("--mlp-dropout", type=float, default=0.0)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument(
        "--early-stopping-metric",
        type=str,
        choices=EARLY_STOPPING_METRICS,
        default="test_acc",
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

    train_loader, test_loader = build_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        train_subset=args.train_subset,
        test_subset=args.test_subset,
        num_workers=args.num_workers,
        seed=args.seed,
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
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
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
            f"test loss={test_loss:.4f} acc={test_acc * 100:.2f}%"
        )
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
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

    metrics_dir = args.results_dir / "metrics"
    figure_dir = args.results_dir / "figures"
    metrics_path = metrics_dir / f"{args.run_name}_metrics.csv"
    config_path = metrics_dir / f"{args.run_name}_config.json"
    summary_path = metrics_dir / f"{args.run_name}_summary.json"
    loss_path, acc_path = plot_curves(history, figure_dir, args.run_name)
    save_metrics_csv(history, metrics_path)
    save_config_json(
        args=args,
        model_config=model_config,
        train_size=len(train_loader.dataset),
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
    save_summary_json(args, history, summary_path, early_stopping_info)

    print(f"Saved metrics: {metrics_path}")
    print(f"Saved config: {config_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved loss plot: {loss_path}")
    print(f"Saved accuracy plot: {acc_path}")


if __name__ == "__main__":
    main()
