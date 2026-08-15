import argparse
import csv
import json
import math
import statistics
from pathlib import Path


SEEDS = (42, 43, 44, 45, 46)
T95_DF4 = 2.7764451051977987
PE_SUFFIXES = (
    "row_sinusoidal",
    "col_sinusoidal",
    "additive_sinusoidal",
    "multiplicative_sinusoidal",
    "radial_sinusoidal",
)
PE_LABELS = {
    "row_sinusoidal": "Row-wise PE",
    "col_sinusoidal": "Column-wise PE",
    "additive_sinusoidal": "Additive PE",
    "multiplicative_sinusoidal": "Multiplicative PE",
    "radial_sinusoidal": "Radial PE",
}
UNFOLDINGS = ("normal_row", "normal_col", "proper_row", "proper_col")
UNFOLDING_LABELS = {
    "normal_row": "Row-major",
    "normal_col": "Column-major",
    "proper_row": "Serpentine rows",
    "proper_col": "Serpentine columns",
}
TARGET_EXPERIMENT = "cifar10_coordinate_aligned_unfolding_5seeds"
SOURCE_EXPERIMENT = "cifar10_final_vit_models_5seeds"
NORMAL_ROW_MODELS = {
    suffix: f"vit_{suffix}" for suffix in PE_SUFFIXES
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
        description="Validate and summarise coordinate-aligned unfolding results."
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--target-experiment", default=TARGET_EXPERIMENT)
    parser.add_argument("--source-experiment", default=SOURCE_EXPERIMENT)
    return parser.parse_args()


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sample_ci95(values):
    if len(values) != 5:
        raise ValueError(f"Expected five values, found {len(values)}")
    return T95_DF4 * statistics.stdev(values) / math.sqrt(5)


def find_one(directory, pattern):
    paths = sorted(directory.glob(pattern))
    if len(paths) != 1:
        raise ValueError(f"Expected one file matching {directory / pattern}; found {len(paths)}")
    return paths[0]


def expected_new_model(unfolding, suffix):
    return f"vit_ca_{unfolding}_{suffix}"


def legacy_model(unfolding, suffix):
    if unfolding == "normal_row":
        return f"vit_{suffix}"
    if suffix in {"row_sinusoidal", "col_sinusoidal", "multiplicative_sinusoidal"}:
        return f"vit_{unfolding}_{suffix}"
    return None


def artifact_status(results_dir, checkpoint_dir, experiment, model, summary_path, run_name):
    metrics_dir = summary_path.parent
    figures_dir = results_dir / experiment / "figures" / model
    checkpoint_model_dir = checkpoint_dir / experiment / model
    required = {
        "summary": summary_path,
        "config": metrics_dir / f"{run_name}_config.json",
        "metrics": metrics_dir / f"{run_name}_metrics.csv",
        "confusion_csv": metrics_dir / f"{run_name}_test_confusion_matrix.csv",
        "checkpoint": checkpoint_model_dir / f"{run_name}_best.pt",
        "accuracy_png": figures_dir / f"{run_name}_accuracy.png",
        "accuracy_pdf": figures_dir / f"{run_name}_accuracy.pdf",
        "loss_png": figures_dir / f"{run_name}_loss.png",
        "loss_pdf": figures_dir / f"{run_name}_loss.pdf",
        "confusion_png": figures_dir / f"{run_name}_test_confusion_matrix.png",
    }
    missing = [name for name, path in required.items() if not path.exists()]
    return required, missing


def validate_config(payload, seed):
    config = payload["config"]
    mismatches = []
    for field, expected in EXPECTED_CONFIG.items():
        if config.get(field) != expected:
            mismatches.append(f"{field}: expected {expected!r}, observed {config.get(field)!r}")
    if config.get("seed") != seed:
        mismatches.append(f"seed: expected {seed}, observed {config.get('seed')}")
    if payload.get("test_evaluation_protocol") != "selected_checkpoint_only":
        mismatches.append("test_evaluation_protocol is not selected_checkpoint_only")
    return mismatches


def load_coordinate_aligned_matrix(args):
    rows = []
    completeness = []
    for suffix in PE_SUFFIXES:
        for unfolding in UNFOLDINGS:
            for seed in SEEDS:
                if unfolding == "normal_row":
                    experiment = args.source_experiment
                    model = NORMAL_ROW_MODELS[suffix]
                    source_type = "reused_protocol_matched_normal_row"
                else:
                    experiment = args.target_experiment
                    model = expected_new_model(unfolding, suffix)
                    source_type = "new_coordinate_aligned_run"
                metrics_dir = args.results_dir / experiment / "metrics" / model
                summary_path = find_one(metrics_dir, f"*_seed{seed}_summary.json")
                payload = json.loads(summary_path.read_text(encoding="utf-8"))
                config_mismatches = validate_config(payload, seed)
                selected = payload["selected_model"]
                if unfolding != "normal_row":
                    if selected.get("position_assignment") != "coordinate_aligned":
                        config_mismatches.append("selected_model.position_assignment is not coordinate_aligned")
                    if selected.get("unfolding") != unfolding:
                        config_mismatches.append(
                            f"selected_model.unfolding expected {unfolding}, observed {selected.get('unfolding')}"
                        )
                    config_path = summary_path.with_name(
                        summary_path.name.replace("_summary.json", "_config.json")
                    )
                    config_payload = json.loads(config_path.read_text(encoding="utf-8"))
                    model_config = config_payload["model"]
                    if model_config.get("position_assignment") != "coordinate_aligned":
                        config_mismatches.append("model config position_assignment is not coordinate_aligned")
                    if model_config.get("unfolding") != unfolding:
                        config_mismatches.append("model config unfolding mismatch")
                run_name = payload["config"]["run_name"]
                artifacts, missing = artifact_status(
                    args.results_dir,
                    args.checkpoint_dir,
                    experiment,
                    model,
                    summary_path,
                    run_name,
                )
                status = "complete" if not missing and not config_mismatches else "invalid"
                completeness.append(
                    {
                        "position_encoding": suffix,
                        "unfolding": unfolding,
                        "seed": seed,
                        "source_type": source_type,
                        "source_experiment": experiment,
                        "source_model": model,
                        "run_name": run_name,
                        "artifact_status": status,
                        "missing_artifacts": "; ".join(missing),
                        "config_mismatches": "; ".join(config_mismatches),
                        "summary_path": str(summary_path),
                        "checkpoint_path": str(artifacts["checkpoint"]),
                    }
                )
                if status != "complete":
                    continue
                rows.append(
                    {
                        "position_assignment": "coordinate_aligned",
                        "position_encoding": suffix,
                        "position_encoding_label": PE_LABELS[suffix],
                        "unfolding": unfolding,
                        "unfolding_label": UNFOLDING_LABELS[unfolding],
                        "seed": seed,
                        "selected_epoch": int(selected["epoch"]),
                        "test_accuracy_pct": 100.0 * float(selected["test_acc"]),
                        "test_loss": float(selected["test_loss"]),
                        "source_type": source_type,
                        "source_summary": str(summary_path),
                    }
                )
    invalid = [row for row in completeness if row["artifact_status"] != "complete"]
    if invalid:
        examples = "\n".join(str(row) for row in invalid[:5])
        raise ValueError(f"Coordinate-aligned matrix has {len(invalid)} invalid runs:\n{examples}")
    return rows, completeness


def summarise(rows):
    output = []
    for suffix in PE_SUFFIXES:
        for unfolding in UNFOLDINGS:
            group = [
                row
                for row in rows
                if row["position_encoding"] == suffix and row["unfolding"] == unfolding
            ]
            if {row["seed"] for row in group} != set(SEEDS):
                raise ValueError(f"Incomplete group: {suffix}/{unfolding}")
            accuracy = [row["test_accuracy_pct"] for row in group]
            loss = [row["test_loss"] for row in group]
            output.append(
                {
                    "position_assignment": "coordinate_aligned",
                    "position_encoding": suffix,
                    "position_encoding_label": PE_LABELS[suffix],
                    "unfolding": unfolding,
                    "unfolding_label": UNFOLDING_LABELS[unfolding],
                    "num_seeds": 5,
                    "mean_test_accuracy_pct": statistics.mean(accuracy),
                    "sample_sd_test_accuracy_pp": statistics.stdev(accuracy),
                    "ci95_half_width_test_accuracy_pp": sample_ci95(accuracy),
                    "mean_test_loss": statistics.mean(loss),
                    "sample_sd_test_loss": statistics.stdev(loss),
                    "ci95_half_width_test_loss": sample_ci95(loss),
                }
            )
    return output


def load_legacy_rows(args):
    rows = []
    availability = []
    for suffix in PE_SUFFIXES:
        for unfolding in UNFOLDINGS:
            model = legacy_model(unfolding, suffix)
            if model is None:
                availability.append(
                    {
                        "position_encoding": suffix,
                        "unfolding": unfolding,
                        "availability": "not_run_in_legacy_matrix",
                        "source_model": "",
                    }
                )
                continue
            availability.append(
                {
                    "position_encoding": suffix,
                    "unfolding": unfolding,
                    "availability": "available",
                    "source_model": model,
                }
            )
            metrics_dir = args.results_dir / args.source_experiment / "metrics" / model
            for seed in SEEDS:
                summary_path = find_one(metrics_dir, f"*_seed{seed}_summary.json")
                payload = json.loads(summary_path.read_text(encoding="utf-8"))
                mismatches = validate_config(payload, seed)
                if mismatches:
                    raise ValueError(f"Legacy source mismatch in {summary_path}: {mismatches}")
                selected = payload["selected_model"]
                rows.append(
                    {
                        "position_assignment": "sequence_slot",
                        "position_encoding": suffix,
                        "position_encoding_label": PE_LABELS[suffix],
                        "unfolding": unfolding,
                        "unfolding_label": UNFOLDING_LABELS[unfolding],
                        "seed": seed,
                        "selected_epoch": int(selected["epoch"]),
                        "test_accuracy_pct": 100.0 * float(selected["test_acc"]),
                        "test_loss": float(selected["test_loss"]),
                        "source_summary": str(summary_path),
                    }
                )
    return rows, availability


def summarise_available_legacy(rows):
    output = []
    keys = sorted({(row["position_encoding"], row["unfolding"]) for row in rows})
    for suffix, unfolding in keys:
        group = [
            row
            for row in rows
            if row["position_encoding"] == suffix and row["unfolding"] == unfolding
        ]
        accuracy = [row["test_accuracy_pct"] for row in group]
        loss = [row["test_loss"] for row in group]
        output.append(
            {
                "position_assignment": "sequence_slot",
                "position_encoding": suffix,
                "position_encoding_label": PE_LABELS[suffix],
                "unfolding": unfolding,
                "unfolding_label": UNFOLDING_LABELS[unfolding],
                "num_seeds": len(group),
                "mean_test_accuracy_pct": statistics.mean(accuracy),
                "sample_sd_test_accuracy_pp": statistics.stdev(accuracy),
                "ci95_half_width_test_accuracy_pp": sample_ci95(accuracy),
                "mean_test_loss": statistics.mean(loss),
                "sample_sd_test_loss": statistics.stdev(loss),
                "ci95_half_width_test_loss": sample_ci95(loss),
            }
        )
    return output


def write_markdown_table(path, title, summary, note):
    lines = [
        f"# {title}",
        "",
        note,
        "",
        "| Fixed PE | Unfolding | Test accuracy (%) | Test loss |",
        "|---|---|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['position_encoding_label']} | {row['unfolding_label']} | "
            f"{row['mean_test_accuracy_pct']:.3f} ± {row['ci95_half_width_test_accuracy_pp']:.3f} | "
            f"{row['mean_test_loss']:.4f} ± {row['ci95_half_width_test_loss']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    report_dir = args.results_dir / args.target_experiment / "reports" / "coordinate_aligned_review"
    report_dir.mkdir(parents=True, exist_ok=True)

    coordinate_rows, completeness = load_coordinate_aligned_matrix(args)
    coordinate_summary = summarise(coordinate_rows)
    legacy_rows, legacy_availability = load_legacy_rows(args)
    legacy_summary = summarise_available_legacy(legacy_rows)

    write_csv(report_dir / "run_completeness.csv", completeness)
    write_csv(report_dir / "coordinate_aligned_selected_test_per_seed.csv", coordinate_rows)
    write_csv(report_dir / "coordinate_aligned_selected_test_summary.csv", coordinate_summary)
    write_csv(report_dir / "legacy_sequence_slot_availability.csv", legacy_availability)
    write_csv(report_dir / "legacy_sequence_slot_selected_test_per_seed.csv", legacy_rows)
    write_csv(report_dir / "legacy_sequence_slot_selected_test_summary.csv", legacy_summary)
    write_markdown_table(
        report_dir / "coordinate_aligned_comparison_table.md",
        "Coordinate-aligned unfolding comparison",
        coordinate_summary,
        (
            "All values use the validation-selected checkpoint over seeds 42--46. "
            "The ± term is the 95% t confidence-interval half-width (df = 4). "
            "Normal-row cells reuse protocol-matched main-experiment summaries."
        ),
    )
    write_markdown_table(
        report_dir / "legacy_sequence_slot_comparison_table.md",
        "Legacy sequence-slot unfolding comparison",
        legacy_summary,
        (
            "This is a separate summary of the available historical sequence-slot assignment runs. "
            "It is not pooled with the coordinate-aligned results; additive and radial non-normal-row "
            "conditions were not present in the legacy matrix."
        ),
    )
    manifest = {
        "coordinate_aligned_expected_runs": 100,
        "coordinate_aligned_complete_runs": len(coordinate_rows),
        "coordinate_aligned_reused_normal_row_runs": sum(
            row["source_type"] == "reused_protocol_matched_normal_row" for row in coordinate_rows
        ),
        "coordinate_aligned_new_runs": sum(
            row["source_type"] == "new_coordinate_aligned_run" for row in coordinate_rows
        ),
        "legacy_available_runs": len(legacy_rows),
        "legacy_missing_cells": sum(
            row["availability"] != "available" for row in legacy_availability
        ),
        "ci_formula": "mean +/- 2.7764451051977987 * sample_sd / sqrt(5)",
        "test_source": "selected_model from selected_checkpoint_only summaries",
        "mixing_rule": "coordinate-aligned and legacy sequence-slot statistics are written separately",
        "report_dir": str(report_dir),
    }
    manifest_path = report_dir / "coordinate_aligned_results_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Coordinate-aligned completeness: 100/100")
    print(f"New runs: {manifest['coordinate_aligned_new_runs']}; reused normal-row runs: 25")
    print(f"Report: {report_dir}")


if __name__ == "__main__":
    main()
