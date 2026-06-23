import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from datasets.cifar10_data import get_class_names
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
from models.registry import (
    EXPERIMENT_REGISTRY,
    SUPPORTED_DATASETS,
    build_selected_experiment,
    get_dataset_display_name,
    resolve_experiment,
)
from result_paths import build_run_artifact_paths


def make_run_name(model_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{model_name}_{timestamp}"


def parse_args():
    parser = argparse.ArgumentParser(description="Unified vision experiment runner.")
    parser.add_argument("--model", choices=tuple(EXPERIMENT_REGISTRY.keys()), required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS, default="cifar10")
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
    parser.add_argument("--cadb-root", type=Path, default=None)
    parser.add_argument("--cadb-test-ratio", type=float, default=0.2)
    parser.add_argument(
        "--cadb-label-mode",
        choices=("exclusive", "inclusive"),
        default="exclusive",
    )
    parser.add_argument(
        "--cadb-balance-mode",
        choices=("none", "train_only", "all_splits"),
        default="none",
    )
    parser.add_argument("--synthetic-train-size", type=int, default=2400)
    parser.add_argument("--synthetic-val-size", type=int, default=600)
    parser.add_argument("--synthetic-test-size", type=int, default=600)
    parser.add_argument("--synthetic-line-width", type=int, default=3)
    parser.add_argument("--synthetic-noise-std", type=float, default=0.08)
    parser.add_argument("--synthetic-max-stripes", type=int, default=4)
    parser.add_argument("--embedding-dropout", type=float, default=0.0)
    parser.add_argument("--attention-dropout", type=float, default=0.0)
    parser.add_argument("--projection-dropout", type=float, default=0.0)
    parser.add_argument("--mlp-dropout", type=float, default=0.0)
    parser.add_argument("--rope-base", type=float, default=10000.0)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--lr-plateau-patience", type=int, default=5)
    parser.add_argument("--lr-plateau-factor", type=float, default=0.5)
    parser.add_argument("--lr-plateau-min-lr", type=float, default=1e-6)
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
    print(f"Dataset: {get_dataset_display_name(args.dataset)} ({args.dataset})")
    print(f"Model: {args.model}")
    print(f"Architecture: {spec.architecture}")
    print(f"Variant: {spec.variant}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Weight decay: {args.weight_decay}")
    if args.dataset == "cadb_orientation":
        print(
            "CADB options: "
            f"label_mode={args.cadb_label_mode}, "
            f"balance_mode={args.cadb_balance_mode}"
        )
    elif args.dataset == "cadb_scene":
        print("CADB options: official scene_categories.json + official split.json")
    elif args.dataset == "cadb_elements":
        print("CADB options: composition_elements multi-label task")
    if args.early_stopping_patience is not None:
        print(
            "Early stopping: "
            f"metric={args.early_stopping_metric}, "
            f"patience={args.early_stopping_patience}, "
            f"min_delta={args.early_stopping_min_delta}"
        )
    print(
        "LR scheduler: "
        f"ReduceLROnPlateau patience={args.lr_plateau_patience}, "
        f"factor={args.lr_plateau_factor}, "
        f"min_lr={args.lr_plateau_min_lr}"
    )


def train_and_collect_history(
    args,
    model,
    train_loader,
    val_loader,
    test_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    task_type,
):
    history = []
    best_metric_value = None
    best_epoch = None
    best_state_dict = None
    patience_counter = 0
    stopped_early = False

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            task_type=task_type,
        )
        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            task_type=task_type,
        )
        test_metrics = evaluate(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            task_type=task_type,
        )
        print(
            f"  train loss={train_metrics['loss']:.4f} acc={train_metrics['acc'] * 100:.2f}% | "
            f"val loss={val_metrics['loss']:.4f} acc={val_metrics['acc'] * 100:.2f}% | "
            f"test loss={test_metrics['loss']:.4f} acc={test_metrics['acc'] * 100:.2f}%"
        )
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "test_loss": test_metrics["loss"],
            "test_acc": test_metrics["acc"],
        }
        if "macro_f1" in val_metrics:
            epoch_metrics["val_macro_f1"] = val_metrics["macro_f1"]
        if "macro_f1" in test_metrics:
            epoch_metrics["test_macro_f1"] = test_metrics["macro_f1"]
        history.append(epoch_metrics)

        previous_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(epoch_metrics[args.early_stopping_metric])
        current_lr = optimizer.param_groups[0]["lr"]
        if current_lr < previous_lr:
            print(f"  lr reduced: {previous_lr:.6g} -> {current_lr:.6g}")

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
    artifact_paths = build_run_artifact_paths(
        results_dir=args.results_dir,
        checkpoint_dir=args.checkpoint_dir,
        model_name=args.model,
        run_name=args.run_name,
    )
    metrics_dir = artifact_paths["metrics_dir"]
    figure_dir = artifact_paths["figures_dir"]
    metrics_path = artifact_paths["metrics_path"]
    config_path = artifact_paths["config_path"]
    summary_path = artifact_paths["summary_path"]
    confusion_csv_path = artifact_paths["confusion_csv_path"]
    confusion_figure_path = artifact_paths["confusion_figure_path"]
    checkpoint_path = artifact_paths["checkpoint_path"]

    loss_path, acc_path = plot_curves(
        history=history,
        figure_dir=figure_dir,
        run_name=args.run_name,
        title_prefix=f"{get_dataset_display_name(args.dataset)} {spec.plot_title_prefix}",
    )
    task_type = metadata.get("task_type", "single_label")
    if task_type == "single_label":
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
        "checkpoint_path": str(checkpoint_path),
        "model_name": args.model,
        "model_family": metadata["architecture"],
        "model_variant": metadata["variant"],
        "position_encoding": metadata.get("position_encoding"),
        "task_type": task_type,
    }
    if "subset_accuracy" in selected_val_metrics:
        selected_model_metrics["val_subset_accuracy"] = selected_val_metrics["subset_accuracy"]
    if "subset_accuracy" in selected_test_metrics:
        selected_model_metrics["test_subset_accuracy"] = selected_test_metrics["subset_accuracy"]
    if task_type == "single_label":
        selected_model_metrics["test_confusion_matrix_csv"] = str(confusion_csv_path)
        selected_model_metrics["test_confusion_matrix_figure"] = str(confusion_figure_path)
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
    if task_type == "single_label":
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
    task_type = metadata.get("task_type", "single_label")

    if task_type == "multilabel":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode=get_metric_mode(args.early_stopping_metric),
        factor=args.lr_plateau_factor,
        patience=args.lr_plateau_patience,
        min_lr=args.lr_plateau_min_lr,
    )

    run_state = train_and_collect_history(
        args=args,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        task_type=task_type,
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
        task_type=task_type,
    )
    selected_test_metrics = evaluate_with_details(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        num_classes=num_classes,
        task_type=task_type,
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
