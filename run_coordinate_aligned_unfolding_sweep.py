import argparse
import json
import subprocess
import sys
from pathlib import Path


SEEDS = (42, 43, 44, 45, 46)
PE_SUFFIXES = (
    "row_sinusoidal",
    "col_sinusoidal",
    "additive_sinusoidal",
    "multiplicative_sinusoidal",
    "radial_sinusoidal",
)
TRAINED_UNFOLDINGS = ("normal_col", "proper_row", "proper_col")
TARGET_EXPERIMENT = "cifar10_coordinate_aligned_unfolding_5seeds"
SOURCE_EXPERIMENT = "cifar10_final_vit_models_5seeds"
SOURCE_NORMAL_ROW_MODELS = {
    "row_sinusoidal": "vit_row_sinusoidal",
    "col_sinusoidal": "vit_col_sinusoidal",
    "additive_sinusoidal": "vit_additive_sinusoidal",
    "multiplicative_sinusoidal": "vit_multiplicative_sinusoidal",
    "radial_sinusoidal": "vit_radial_sinusoidal",
}
EXPECTED_CONFIG = {
    "dataset": "cifar10",
    "epochs": 100,
    "batch_size": 128,
    "lr": 3e-4,
    "weight_decay": 0.05,
    "train_subset": None,
    "val_subset": None,
    "test_subset": None,
    "val_ratio": 0.1,
    "split_seed": 42,
    "embedding_dropout": 0.0,
    "attention_dropout": 0.0,
    "projection_dropout": 0.0,
    "mlp_dropout": 0.0,
    "early_stopping_patience": 10,
    "early_stopping_metric": "val_acc",
    "early_stopping_min_delta": 0.001,
    "lr_plateau_patience": 5,
    "lr_plateau_factor": 0.5,
    "lr_plateau_min_lr": 1e-6,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the 75 non-normal-row coordinate-aligned CIFAR-10 runs."
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Validate reuse and write the experiment plan without training.",
    )
    return parser.parse_args()


def find_summary(results_dir, experiment, model, seed):
    paths = sorted((results_dir / experiment / "metrics" / model).glob(f"*_seed{seed}_summary.json"))
    if len(paths) != 1:
        raise ValueError(
            f"Expected one source summary for {model}, seed {seed}; found {len(paths)}"
        )
    return paths[0]


def validate_reused_normal_row(results_dir):
    records = []
    for pe_suffix, model in SOURCE_NORMAL_ROW_MODELS.items():
        for seed in SEEDS:
            path = find_summary(results_dir, SOURCE_EXPERIMENT, model, seed)
            payload = json.loads(path.read_text(encoding="utf-8"))
            config = payload["config"]
            mismatches = {
                field: {"expected": expected, "observed": config.get(field)}
                for field, expected in EXPECTED_CONFIG.items()
                if config.get(field) != expected
            }
            if int(config["seed"]) != seed:
                mismatches["seed"] = {"expected": seed, "observed": config.get("seed")}
            if payload.get("test_evaluation_protocol") != "selected_checkpoint_only":
                mismatches["test_evaluation_protocol"] = {
                    "expected": "selected_checkpoint_only",
                    "observed": payload.get("test_evaluation_protocol"),
                }
            if mismatches:
                raise ValueError(f"Cannot reuse {path}: {mismatches}")
            records.append(
                {
                    "pe_suffix": pe_suffix,
                    "unfolding": "normal_row",
                    "seed": seed,
                    "source_model": model,
                    "source_summary": str(path),
                    "reuse_status": "protocol_match",
                }
            )
    return records


def main():
    args = parse_args()
    reuse_records = validate_reused_normal_row(args.results_dir)
    target_dir = args.results_dir / TARGET_EXPERIMENT
    target_dir.mkdir(parents=True, exist_ok=True)
    model_names = [
        f"vit_ca_{unfolding}_{suffix}"
        for unfolding in TRAINED_UNFOLDINGS
        for suffix in PE_SUFFIXES
    ]
    plan = {
        "target_experiment": TARGET_EXPERIMENT,
        "source_experiment": SOURCE_EXPERIMENT,
        "position_assignment": "coordinate_aligned",
        "seeds": list(SEEDS),
        "pe_suffixes": list(PE_SUFFIXES),
        "all_unfoldings": ["normal_row", *TRAINED_UNFOLDINGS],
        "normal_row_reused_runs": len(reuse_records),
        "new_runs_required": len(model_names) * len(SEEDS),
        "total_matrix_runs": len(PE_SUFFIXES) * 4 * len(SEEDS),
        "expected_config": EXPECTED_CONFIG,
        "normal_row_reuse": reuse_records,
        "new_model_names": model_names,
    }
    plan_path = target_dir / "coordinate_aligned_experiment_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(f"Normal-row reuse gate: PASSED ({len(reuse_records)} runs)", flush=True)
    print(f"New runs required: {plan['new_runs_required']}", flush=True)
    print(f"Plan: {plan_path}", flush=True)
    if args.plan_only:
        return

    command = [
        sys.executable,
        "run_seed_sweep.py",
        "--dataset",
        "cifar10",
        "--models",
        *model_names,
        "--seeds",
        *[str(seed) for seed in SEEDS],
        "--split-seed",
        "42",
        "--experiment-name",
        TARGET_EXPERIMENT,
        "--epochs",
        "100",
        "--batch-size",
        "128",
        "--lr",
        "3e-4",
        "--weight-decay",
        "0.05",
        "--val-ratio",
        "0.1",
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
        "1e-6",
        "--data-dir",
        str(args.data_dir),
        "--results-dir",
        str(args.results_dir),
        "--checkpoint-dir",
        str(args.checkpoint_dir),
        "--skip-reports",
    ]
    if args.skip_existing:
        command.append("--skip-existing")
    subprocess.run(command, check=True)
    subprocess.run(
        [
            sys.executable,
            "analyze_coordinate_aligned_unfolding.py",
            "--results-dir",
            str(args.results_dir),
            "--checkpoint-dir",
            str(args.checkpoint_dir),
            "--target-experiment",
            TARGET_EXPERIMENT,
            "--source-experiment",
            SOURCE_EXPERIMENT,
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
