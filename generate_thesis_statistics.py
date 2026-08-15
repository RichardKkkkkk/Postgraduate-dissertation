import argparse
import csv
import json
import math
import statistics
from argparse import Namespace
from pathlib import Path

import torch

from models.registry import EXPERIMENT_REGISTRY


T_CRITICAL_95_DF4 = 2.7764451051977987
EXPECTED_SEEDS = (42, 43, 44, 45, 46)
CORE_WITH_RADIAL = (
    "vit_baseline",
    "vit_learnable_position",
    "vit_row_sinusoidal",
    "vit_col_sinusoidal",
    "vit_additive_sinusoidal",
    "vit_additive_sinusoidal_shifted",
    "vit_multiplicative_sinusoidal",
    "vit_multiplicative_sinusoidal_shifted",
    "vit_radial_sinusoidal",
)
KEY_CONTRASTS = (
    ("no_pe_to_learnable", "vit_baseline", "vit_learnable_position"),
    (
        "no_pe_to_shifted_multiplicative",
        "vit_baseline",
        "vit_multiplicative_sinusoidal_shifted",
    ),
    (
        "learnable_to_shifted_multiplicative",
        "vit_learnable_position",
        "vit_multiplicative_sinusoidal_shifted",
    ),
    (
        "additive_to_shifted_additive",
        "vit_additive_sinusoidal",
        "vit_additive_sinusoidal_shifted",
    ),
    (
        "multiplicative_to_shifted_multiplicative",
        "vit_multiplicative_sinusoidal",
        "vit_multiplicative_sinusoidal_shifted",
    ),
    (
        "order_matched_learnable_to_hybrid",
        "vit_normal_col_learnable_position",
        "vit_normal_col_learnable_multiplicative_sinusoidal",
    ),
    (
        "learnable_to_best_fusion",
        "vit_learnable_position",
        "vit_row_col_cross_attention_mlp_head_fusion",
    ),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate selected-checkpoint thesis statistics from final summaries."
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=Path("results/cifar10_final_vit_models_5seeds"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/cifar10_final_vit_models_5seeds/reports/thesis_core"
        ),
    )
    return parser.parse_args()


def mean_sd_ci(values):
    n = len(values)
    mean = statistics.mean(values)
    sd = statistics.stdev(values)
    half_width = T_CRITICAL_95_DF4 * sd / math.sqrt(n)
    return mean, sd, half_width, mean - half_width, mean + half_width


def build_model_args():
    return Namespace(
        dataset="cifar10",
        image_size=None,
        embedding_dropout=0.0,
        attention_dropout=0.0,
        projection_dropout=0.0,
        mlp_dropout=0.0,
        rope_base=10000.0,
    )


def count_parameters(model_name):
    model, _ = EXPERIMENT_REGISTRY[model_name].build_model(build_model_args())
    return sum(parameter.numel() for parameter in model.parameters())


def load_records(experiment_dir):
    summary_paths = sorted((experiment_dir / "metrics").rglob("*_summary.json"))
    records = []
    for path in summary_paths:
        summary = json.loads(path.read_text(encoding="utf-8"))
        config = summary["config"]
        selected = summary["selected_model"]
        record = {
            "model": config["model"],
            "seed": int(config["seed"]),
            "split_seed": config.get("split_seed"),
            "selected_epoch": int(selected["epoch"]),
            "test_acc": float(selected["test_acc"]),
            "test_loss": float(selected["test_loss"]),
            "test_macro_f1": float(selected["test_macro_f1"]),
            "protocol": summary.get("test_evaluation_protocol"),
            "summary_path": str(path),
        }
        records.append(record)

    if len(records) != 160:
        raise RuntimeError(f"Expected 160 final summaries; found {len(records)}")
    if {record["protocol"] for record in records} != {"selected_checkpoint_only"}:
        raise RuntimeError("Not every run uses selected_checkpoint_only test evaluation")
    if {record["split_seed"] for record in records} != {42}:
        raise RuntimeError("Final summaries do not share split_seed=42")

    by_model = {}
    for record in records:
        by_model.setdefault(record["model"], []).append(record)
    for model_name, model_records in by_model.items():
        seeds = tuple(sorted(record["seed"] for record in model_records))
        if seeds != EXPECTED_SEEDS:
            raise RuntimeError(f"{model_name} has seeds {seeds}, expected {EXPECTED_SEEDS}")
    return records, by_model


def write_csv(path, rows, fieldnames):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records, by_model = load_records(args.experiment_dir)

    per_seed_rows = []
    for record in sorted(records, key=lambda item: (item["model"], item["seed"])):
        per_seed_rows.append(
            {
                **record,
                "test_acc_pct": record["test_acc"] * 100.0,
                "test_macro_f1_pct": record["test_macro_f1"] * 100.0,
            }
        )
    per_seed_path = args.output_dir / "selected_test_metrics_per_seed.csv"
    write_csv(per_seed_path, per_seed_rows, list(per_seed_rows[0]))

    summary_rows = []
    for model_name, model_records in sorted(by_model.items()):
        model_records = sorted(model_records, key=lambda item: item["seed"])
        acc = [record["test_acc"] * 100.0 for record in model_records]
        loss = [record["test_loss"] for record in model_records]
        f1 = [record["test_macro_f1"] * 100.0 for record in model_records]
        acc_stats = mean_sd_ci(acc)
        loss_stats = mean_sd_ci(loss)
        f1_stats = mean_sd_ci(f1)
        summary_rows.append(
            {
                "model": model_name,
                "n": len(model_records),
                "parameter_count": count_parameters(model_name),
                "mean_test_acc_pct": acc_stats[0],
                "sample_sd_test_acc_pp": acc_stats[1],
                "ci95_half_width_test_acc_pp": acc_stats[2],
                "ci95_lower_test_acc_pct": acc_stats[3],
                "ci95_upper_test_acc_pct": acc_stats[4],
                "mean_test_loss": loss_stats[0],
                "sample_sd_test_loss": loss_stats[1],
                "ci95_half_width_test_loss": loss_stats[2],
                "ci95_lower_test_loss": loss_stats[3],
                "ci95_upper_test_loss": loss_stats[4],
                "mean_test_macro_f1_pct": f1_stats[0],
                "sample_sd_test_macro_f1_pp": f1_stats[1],
                "ci95_half_width_test_macro_f1_pp": f1_stats[2],
                "mean_selected_epoch": statistics.mean(
                    record["selected_epoch"] for record in model_records
                ),
                "core_with_radial": model_name in CORE_WITH_RADIAL,
            }
        )
    summary_rows.sort(key=lambda row: row["mean_test_acc_pct"], reverse=True)
    summary_path = args.output_dir / "selected_test_summary_with_ci.csv"
    write_csv(summary_path, summary_rows, list(summary_rows[0]))

    core_rows = [row for row in summary_rows if row["model"] in CORE_WITH_RADIAL]
    core_rows.sort(key=lambda row: CORE_WITH_RADIAL.index(row["model"]))
    core_path = args.output_dir / "core_pe_with_radial_test_summary.csv"
    write_csv(core_path, core_rows, list(core_rows[0]))

    paired_rows = []
    for contrast, reference_model, comparison_model in KEY_CONTRASTS:
        reference = {
            record["seed"]: record for record in by_model[reference_model]
        }
        comparison = {
            record["seed"]: record for record in by_model[comparison_model]
        }
        acc_deltas = [
            (comparison[seed]["test_acc"] - reference[seed]["test_acc"]) * 100.0
            for seed in EXPECTED_SEEDS
        ]
        loss_deltas = [
            comparison[seed]["test_loss"] - reference[seed]["test_loss"]
            for seed in EXPECTED_SEEDS
        ]
        acc_stats = mean_sd_ci(acc_deltas)
        loss_stats = mean_sd_ci(loss_deltas)
        paired_rows.append(
            {
                "contrast": contrast,
                "reference_model": reference_model,
                "comparison_model": comparison_model,
                "delta_definition": "comparison_minus_reference",
                "n_pairs": len(EXPECTED_SEEDS),
                "mean_test_acc_delta_pp": acc_stats[0],
                "sample_sd_test_acc_delta_pp": acc_stats[1],
                "ci95_half_width_test_acc_delta_pp": acc_stats[2],
                "ci95_lower_test_acc_delta_pp": acc_stats[3],
                "ci95_upper_test_acc_delta_pp": acc_stats[4],
                "mean_test_loss_delta": loss_stats[0],
                "sample_sd_test_loss_delta": loss_stats[1],
                "ci95_half_width_test_loss_delta": loss_stats[2],
                "seed42_acc_delta_pp": acc_deltas[0],
                "seed43_acc_delta_pp": acc_deltas[1],
                "seed44_acc_delta_pp": acc_deltas[2],
                "seed45_acc_delta_pp": acc_deltas[3],
                "seed46_acc_delta_pp": acc_deltas[4],
                "positive_acc_pairs": sum(delta > 0 for delta in acc_deltas),
            }
        )
    paired_path = args.output_dir / "key_paired_test_contrasts.csv"
    write_csv(paired_path, paired_rows, list(paired_rows[0]))

    hybrid_scale_rows = []
    hybrid_model = "vit_normal_col_learnable_multiplicative_sinusoidal"
    for record in sorted(by_model[hybrid_model], key=lambda item: item["seed"]):
        source_summary = json.loads(Path(record["summary_path"]).read_text(encoding="utf-8"))
        checkpoint_path = Path(source_summary["selected_model"]["checkpoint_path"])
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint["model_state_dict"]
        if "fixed_pos_scale" not in state_dict:
            raise RuntimeError(f"fixed_pos_scale missing from {checkpoint_path}")
        hybrid_scale_rows.append(
            {
                "model": hybrid_model,
                "seed": record["seed"],
                "selected_epoch": record["selected_epoch"],
                "fixed_pos_scale": float(state_dict["fixed_pos_scale"].item()),
                "checkpoint_path": str(checkpoint_path),
            }
        )
    hybrid_scale_path = args.output_dir / "hybrid_fixed_pos_scale.csv"
    write_csv(hybrid_scale_path, hybrid_scale_rows, list(hybrid_scale_rows[0]))
    hybrid_scale_values = [row["fixed_pos_scale"] for row in hybrid_scale_rows]
    hybrid_scale_stats = mean_sd_ci(hybrid_scale_values)

    manifest = {
        "source_experiment": str(args.experiment_dir),
        "summary_count": len(records),
        "model_count": len(by_model),
        "seeds": list(EXPECTED_SEEDS),
        "split_seed": 42,
        "test_protocol": "selected_checkpoint_only",
        "confidence_interval": {
            "method": "two-sided Student t interval around the sample mean",
            "confidence_level": 0.95,
            "degrees_of_freedom": 4,
            "t_critical": T_CRITICAL_95_DF4,
        },
        "paired_delta_definition": "comparison minus reference for matching seed",
        "outputs": {
            "per_seed": str(per_seed_path),
            "all_models_summary": str(summary_path),
            "core_with_radial": str(core_path),
            "key_paired_contrasts": str(paired_path),
            "hybrid_fixed_pos_scale": str(hybrid_scale_path),
        },
        "hybrid_fixed_pos_scale_summary": {
            "mean": hybrid_scale_stats[0],
            "sample_sd": hybrid_scale_stats[1],
            "ci95_half_width": hybrid_scale_stats[2],
            "minimum": min(hybrid_scale_values),
            "maximum": max(hybrid_scale_values),
        },
        "statistical_caution": (
            "Five seeds are not treated as five independent datasets. Exact two-sided "
            "Wilcoxon signed-rank tests with n=5 cannot attain p<0.05; paired effects "
            "and uncertainty are reported descriptively."
        ),
    }
    manifest_path = args.output_dir / "thesis_statistics_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(per_seed_path)
    print(summary_path)
    print(core_path)
    print(paired_path)
    print(hybrid_scale_path)
    print(manifest_path)


if __name__ == "__main__":
    main()
