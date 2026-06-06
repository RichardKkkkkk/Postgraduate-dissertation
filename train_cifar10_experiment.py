import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

from cifar10_data import get_class_names
from experiment_utils import (
    EARLY_STOPPING_METRICS,
    evaluate,
    evaluate_with_details,
    get_device,
    get_metric_mode,
    maybe_update_early_stopping,
    plot_confusion_matrix,
    plot_curves,
    save_best_checkpoint,
    save_config_json,
    save_confusion_matrix_csv,
    save_metrics_csv,
    save_summary_json,
    set_seed,
    train_one_epoch,
)
from model_registry import (
    EXPERIMENT_REGISTRY,
    build_selected_experiment,
    resolve_experiment,
)


def make_run_name(model_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{model_name}_{timestamp}"


def parse_args():
    parser = argparse.ArgumentParser(description="Unified CIFAR-10 experiment runner.")
    parser.add_argument("--model", choices=tuple(EXPERIMENT_REGISTRY.keys()), required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--train-subset", type=int, default=None)
    parser.add_argument("--val-subset", type=int, default=None)
    parser.add_argument("--test-subset", type=int, default=None)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--embedding-dropout", type=float, default=0.0)
    parser.add_argument("--attention-dropout", type=float, default=0.0)
    parser.add_argument("--projection-dropout", type=float, default=0.0)
    parser.add_argument("--mlp-dropout", type=float, default=0.0)
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


def print_run_header(args, spec, device):
    print(f"Using device: {device}")
    print(f"Run name: {args.run_name}")
    print(f"Model: {args.model}")
    print(f"Architecture: {spec.architecture}")
    print(f"Variant: {spec.variant}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Weight decay: {args.weight_decay}")
    if args.early_stopping_patience is not None:
        print(
            "Early stopping: "
            f"metric={args.early_stopping_metric}, "
            f"patience={args.early_stopping_patience}, "
            f"min_delta={args.early_stopping_min_delta}"
        )


def train_and_collect_history(args, model, train_loader, val_loader, test_loader, criterion, optimizer, device):
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

    return {
        "history": history,
        "best_metric_value": best_metric_value,
        "best_epoch": best_epoch,
        "best_state_dict": best_state_dict,
        "stopped_early": stopped_early,
    }


def save_run_outputs(
    args,
    spec,
    metadata,
    model,
    model_config,
    history,
    best_epoch,
    best_metric_value,
    stopped_early,
    selected_val_metrics,
    selected_test_metrics,
    class_names,
    train_loader,
    val_loader,
    test_loader,
    device,
):
    metrics_dir = args.results_dir / "metrics"
    figure_dir = args.results_dir / "figures"
    checkpoint_dir = args.checkpoint_dir
    metrics_path = metrics_dir / f"{args.run_name}_metrics.csv"
    config_path = metrics_dir / f"{args.run_name}_config.json"
    summary_path = metrics_dir / f"{args.run_name}_summary.json"
    confusion_csv_path = metrics_dir / f"{args.run_name}_test_confusion_matrix.csv"
    confusion_figure_path = figure_dir / f"{args.run_name}_test_confusion_matrix.png"
    checkpoint_path = checkpoint_dir / f"{args.run_name}_best.pt"

    loss_path, acc_path = plot_curves(
        history=history,
        figure_dir=figure_dir,
        run_name=args.run_name,
        title_prefix=spec.plot_title_prefix,
    )
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
        "model_name": args.model,
        "model_family": metadata["architecture"],
        "model_variant": metadata["variant"],
        "position_encoding": metadata.get("position_encoding"),
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


def main():
    args = parse_args()
    spec = resolve_experiment(args)
    set_seed(args.seed)
    args.run_name = args.run_name or make_run_name(args.model)

    device = get_device()
    print_run_header(args, spec, device)

    model, model_config, train_loader, val_loader, test_loader, metadata = build_selected_experiment(args, spec)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    run_state = train_and_collect_history(
        args=args,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
    )

    if run_state["best_state_dict"] is not None:
        model.load_state_dict(run_state["best_state_dict"])

    class_names = get_class_names(test_loader.dataset)
    num_classes = len(class_names)
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
        f"Selected checkpoint epoch {run_state['best_epoch']}: "
        f"val loss={selected_val_metrics['loss']:.4f} acc={selected_val_metrics['acc'] * 100:.2f}% | "
        f"test loss={selected_test_metrics['loss']:.4f} acc={selected_test_metrics['acc'] * 100:.2f}% | "
        f"test macro_f1={selected_test_metrics['macro_f1']:.4f}"
    )

    save_run_outputs(
        args=args,
        spec=spec,
        metadata=metadata,
        model=model,
        model_config=model_config,
        history=run_state["history"],
        best_epoch=run_state["best_epoch"],
        best_metric_value=run_state["best_metric_value"],
        stopped_early=run_state["stopped_early"],
        selected_val_metrics=selected_val_metrics,
        selected_test_metrics=selected_test_metrics,
        class_names=class_names,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
    )


if __name__ == "__main__":
    main()
