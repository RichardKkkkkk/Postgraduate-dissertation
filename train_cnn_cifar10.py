import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from train_cifar10 import (
    EARLY_STOPPING_METRICS,
    CIFAR10_MEAN,
    CIFAR10_STD,
    evaluate,
    get_device,
    get_metric_mode,
    make_subset,
    maybe_update_early_stopping,
    plot_curves,
    save_config_json,
    save_metrics_csv,
    save_summary_json,
    set_seed,
    split_train_val_indices,
    train_one_epoch,
)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def make_run_name():
    return datetime.now().strftime("cifar10_resnet18_%Y%m%d_%H%M%S")


def build_dataloaders(
    data_dir,
    batch_size,
    train_subset,
    val_subset,
    test_subset,
    num_workers,
    seed,
    val_ratio,
    image_size,
    use_imagenet_norm,
):
    mean = IMAGENET_MEAN if use_imagenet_norm else CIFAR10_MEAN
    std = IMAGENET_STD if use_imagenet_norm else CIFAR10_STD

    transform_train = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
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
    train_dataset = torch.utils.data.Subset(full_train_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_val_dataset, val_indices)

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


def build_model(weights_name):
    if weights_name == "imagenet":
        weights = models.ResNet18_Weights.DEFAULT
    else:
        weights = None

    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Train a ResNet18 CNN on CIFAR-10.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--weights", choices=["imagenet", "none"], default="imagenet")
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--train-subset", type=int, default=None)
    parser.add_argument("--val-subset", type=int, default=None)
    parser.add_argument("--test-subset", type=int, default=None)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
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
    args.image_size = args.image_size or (224 if args.weights == "imagenet" else 32)

    device = get_device()
    print(f"Using device: {device}")
    print(f"Run name: {args.run_name}")
    print(f"Epochs: {args.epochs}")
    print(f"Weights: {args.weights}")
    print(f"Image size: {args.image_size}")
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
        image_size=args.image_size,
        use_imagenet_norm=args.weights == "imagenet",
    )

    model_config = {
        "architecture": "resnet18",
        "weights": args.weights,
        "image_size": args.image_size,
        "num_classes": 10,
    }
    model = build_model(args.weights).to(device)

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
    selected_val_loss, selected_val_acc = evaluate(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
    )
    selected_test_loss, selected_test_acc = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
    )
    print(
        f"Selected checkpoint epoch {best_epoch}: "
        f"val loss={selected_val_loss:.4f} acc={selected_val_acc * 100:.2f}% | "
        f"test loss={selected_test_loss:.4f} acc={selected_test_acc * 100:.2f}%"
    )

    metrics_dir = args.results_dir / "metrics"
    figure_dir = args.results_dir / "figures"
    metrics_path = metrics_dir / f"{args.run_name}_metrics.csv"
    config_path = metrics_dir / f"{args.run_name}_config.json"
    summary_path = metrics_dir / f"{args.run_name}_summary.json"
    loss_path, acc_path = plot_curves(
        history=history,
        figure_dir=figure_dir,
        run_name=args.run_name,
        title_prefix="CIFAR-10 ResNet18",
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
        "val_loss": selected_val_loss,
        "val_acc": selected_val_acc,
        "test_loss": selected_test_loss,
        "test_acc": selected_test_acc,
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


if __name__ == "__main__":
    main()
