import argparse
import subprocess
import sys


DEFAULT_MODELS = ("vit_learnable_position", "vit_multiplicative_sinusoidal")
DEFAULT_SEEDS = (42, 43, 44, 45, 46)
DEFAULT_TRAIN_SIZES = (1000, 5000, 10000)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the pre-specified CIFAR-10 low-data thesis experiment."
    )
    parser.add_argument("--train-sizes", nargs="+", type=int, default=DEFAULT_TRAIN_SIZES)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--experiment-name", default="cifar10_low_data_5seeds")
    return parser.parse_args()


def main():
    args = parse_args()
    for train_size in args.train_sizes:
        command = [
            sys.executable,
            "run_seed_sweep.py",
            "--dataset",
            "cifar10",
            "--models",
            *args.models,
            "--seeds",
            *(str(seed) for seed in args.seeds),
            "--split-seed",
            "42",
            "--experiment-name",
            args.experiment_name,
            "--run-prefix",
            f"train{train_size}",
            "--epochs",
            str(args.epochs),
            "--batch-size",
            "128",
            "--lr",
            "0.0003",
            "--weight-decay",
            "0.05",
            "--val-ratio",
            "0.1",
            "--train-subset",
            str(train_size),
            "--num-workers",
            str(args.num_workers),
            "--early-stopping-patience",
            "10",
            "--early-stopping-metric",
            "val_acc",
            "--early-stopping-min-delta",
            "0.001",
            "--lr-plateau-patience",
            "5",
            "--lr-plateau-factor",
            "0.5",
            "--lr-plateau-min-lr",
            "0.000001",
            "--skip-existing",
            "--skip-reports",
        ]
        print("Running:", " ".join(command), flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
