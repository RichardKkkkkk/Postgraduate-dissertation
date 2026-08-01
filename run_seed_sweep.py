import argparse
import subprocess
import sys
from pathlib import Path

from models.registry import EXPERIMENT_REGISTRY, SUPPORTED_DATASETS
from paper_plotting import get_model_label
from result_paths import resolve_run_artifact_paths


DEFAULT_MODELS = ["vit_baseline", "vit_learnable_position", "vit_rope"]
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
        "--all-models",
        action="store_true",
        help="Run every registered model. Overrides --models.",
    )
    parser.add_argument(
        "--exclude-models",
        nargs="+",
        choices=tuple(EXPERIMENT_REGISTRY.keys()),
        default=[],
        help="Models to remove from the selected sweep.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        required=True,
        help="Seeds to sweep, for example: --seeds 42 43 44",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS, default="cifar10")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--experiment-name", type=str, default=None)
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
    parser.add_argument("--lr-plateau-patience", type=int, default=5)
    parser.add_argument("--lr-plateau-factor", type=float, default=0.5)
    parser.add_argument("--lr-plateau-min-lr", type=float, default=1e-6)
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


def summary_exists(results_dir: Path, run_name: str, experiment_name: str | None = None):
    return resolve_run_artifact_paths(
        results_dir,
        run_name,
        experiment_name=experiment_name,
    )["summary_path"] is not None


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
        "--dataset",
        args.dataset,
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
        "--lr-plateau-patience",
        str(args.lr_plateau_patience),
        "--lr-plateau-factor",
        str(args.lr_plateau_factor),
        "--lr-plateau-min-lr",
        str(args.lr_plateau_min_lr),
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
    append_optional_arg(command, "--experiment-name", args.experiment_name)
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
    append_optional_arg(command, "--experiment-name", args.experiment_name)
    if not args.with_ppt:
        command.append("--skip-ppt")
    for run_name, label in run_specs:
        command.extend(["--run", f"{run_name}={label}"])
    return command


def main():
    args = parse_args()
    model_names = list(EXPERIMENT_REGISTRY.keys()) if args.all_models else list(args.models)
    excluded_model_names = set(args.exclude_models)
    model_names = [model_name for model_name in model_names if model_name not in excluded_model_names]
    if not model_names:
        raise ValueError("No models selected. Check --models, --all-models, and --exclude-models.")

    for seed in args.seeds:
        print("")
        print(f"===== Seed {seed} =====")
        run_specs = []
        for model_name in model_names:
            run_name = build_run_name(model_name, seed, args.run_prefix)
            run_specs.append((run_name, get_model_label(model_name)))

            if args.skip_existing and summary_exists(args.results_dir, run_name, args.experiment_name):
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
