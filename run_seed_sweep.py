import argparse
import subprocess
import sys
from pathlib import Path

from models.registry import EXPERIMENT_REGISTRY
from result_paths import resolve_run_artifact_paths


DEFAULT_MODELS = ["vit_baseline", "vit_rope", "vit_rope_2d"]
MODEL_LABELS = {
    "vit_baseline": "ViT Baseline",
    "vit_rope": "ViT RoPE",
    "vit_rope_2d": "ViT RoPE 2D",
    "resnet18_scratch": "ResNet18 Scratch",
    "resnet18_imagenet": "ResNet18 ImageNet",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run multiple seeds and generate one comparison report per seed."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(EXPERIMENT_REGISTRY.keys()),
        default=DEFAULT_MODELS,
        help="Models to run for each seed.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        required=True,
        help="Seeds to sweep, for example: --seeds 42 43 44",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--train-subset", type=int, default=None)
    parser.add_argument("--val-subset", type=int, default=None)
    parser.add_argument("--test-subset", type=int, default=None)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--embedding-dropout", type=float, default=0.0)
    parser.add_argument("--attention-dropout", type=float, default=0.0)
    parser.add_argument("--projection-dropout", type=float, default=0.0)
    parser.add_argument("--mlp-dropout", type=float, default=0.0)
    parser.add_argument("--rope-base", type=float, default=10000.0)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.001)
    parser.add_argument("--early-stopping-metric", type=str, default="val_acc")
    parser.add_argument(
        "--run-prefix",
        type=str,
        default=None,
        help="Optional prefix for run names. Example: june08 -> june08_vit_rope_seed42",
    )
    parser.add_argument(
        "--report-prefix",
        type=str,
        default="seed_compare",
        help="Prefix for per-seed report folders.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip training runs whose summary JSON already exists.",
    )
    parser.add_argument(
        "--skip-reports",
        action="store_true",
        help="Run training only and do not generate per-seed comparison reports.",
    )
    parser.add_argument(
        "--with-ppt",
        action="store_true",
        help="Also export PPT files for each seed report. Default is plots/CSV/overview only.",
    )
    return parser.parse_args()


def build_run_name(model_name: str, seed: int, run_prefix: str | None):
    base_name = f"{model_name}_seed{seed}"
    if run_prefix:
        return f"{run_prefix}_{base_name}"
    return base_name


def build_report_name(seed: int, report_prefix: str, run_prefix: str | None):
    if run_prefix:
        return f"{report_prefix}_{run_prefix}_seed{seed}"
    return f"{report_prefix}_seed{seed}"


def summary_exists(results_dir: Path, run_name: str):
    return resolve_run_artifact_paths(results_dir, run_name)["summary_path"] is not None


def append_optional_arg(command: list[str], flag: str, value):
    if value is None:
        return
    command.extend([flag, str(value)])


def run_command(command: list[str]):
    print("")
    print("Running command:")
    print(" ".join(command))
    subprocess.run(command, check=True)


def build_train_command(args, model_name: str, seed: int):
    run_name = build_run_name(model_name, seed, args.run_prefix)
    command = [
        sys.executable,
        "train_cifar10_experiment.py",
        "--model",
        model_name,
        "--seed",
        str(seed),
        "--epochs",
        str(args.epochs),
        "--data-dir",
        str(args.data_dir),
        "--results-dir",
        str(args.results_dir),
        "--checkpoint-dir",
        str(args.checkpoint_dir),
        "--val-ratio",
        str(args.val_ratio),
        "--num-workers",
        str(args.num_workers),
        "--embedding-dropout",
        str(args.embedding_dropout),
        "--attention-dropout",
        str(args.attention_dropout),
        "--projection-dropout",
        str(args.projection_dropout),
        "--mlp-dropout",
        str(args.mlp_dropout),
        "--rope-base",
        str(args.rope_base),
        "--early-stopping-patience",
        str(args.early_stopping_patience),
        "--early-stopping-min-delta",
        str(args.early_stopping_min_delta),
        "--early-stopping-metric",
        args.early_stopping_metric,
        "--run-name",
        run_name,
    ]
    append_optional_arg(command, "--batch-size", args.batch_size)
    append_optional_arg(command, "--lr", args.lr)
    append_optional_arg(command, "--weight-decay", args.weight_decay)
    append_optional_arg(command, "--train-subset", args.train_subset)
    append_optional_arg(command, "--val-subset", args.val_subset)
    append_optional_arg(command, "--test-subset", args.test_subset)
    append_optional_arg(command, "--image-size", args.image_size)
    return command


def build_report_command(args, seed: int, run_specs: list[tuple[str, str]]):
    command = [
        sys.executable,
        "generate_comparison_report.py",
        "--results-dir",
        str(args.results_dir),
        "--report-name",
        build_report_name(seed, args.report_prefix, args.run_prefix),
        "--title",
        f"Seed {seed} Model Comparison",
    ]
    if not args.with_ppt:
        command.append("--skip-ppt")
    for run_name, label in run_specs:
        command.extend(["--run", f"{run_name}={label}"])
    return command


def main():
    args = parse_args()

    for seed in args.seeds:
        print("")
        print(f"===== Seed {seed} =====")
        run_specs = []
        for model_name in args.models:
            run_name = build_run_name(model_name, seed, args.run_prefix)
            run_specs.append((run_name, MODEL_LABELS.get(model_name, model_name)))

            if args.skip_existing and summary_exists(args.results_dir, run_name):
                print(f"Skipping existing run: {run_name}")
                continue

            train_command = build_train_command(args, model_name, seed)
            run_command(train_command)

        if args.skip_reports:
            continue

        report_command = build_report_command(args, seed, run_specs)
        run_command(report_command)


if __name__ == "__main__":
    main()
