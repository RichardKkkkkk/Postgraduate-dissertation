from pathlib import Path


def get_model_artifact_dirs(results_dir: Path, checkpoint_dir: Path, model_name: str):
    model_slug = model_name.strip()
    return {
        "metrics_dir": results_dir / "metrics" / model_slug,
        "figures_dir": results_dir / "figures" / model_slug,
        "checkpoint_dir": checkpoint_dir / model_slug,
    }


def build_run_artifact_paths(results_dir: Path, checkpoint_dir: Path, model_name: str, run_name: str):
    artifact_dirs = get_model_artifact_dirs(results_dir, checkpoint_dir, model_name)
    return {
        "metrics_dir": artifact_dirs["metrics_dir"],
        "figures_dir": artifact_dirs["figures_dir"],
        "checkpoint_dir": artifact_dirs["checkpoint_dir"],
        "metrics_path": artifact_dirs["metrics_dir"] / f"{run_name}_metrics.csv",
        "config_path": artifact_dirs["metrics_dir"] / f"{run_name}_config.json",
        "summary_path": artifact_dirs["metrics_dir"] / f"{run_name}_summary.json",
        "confusion_csv_path": artifact_dirs["metrics_dir"] / f"{run_name}_test_confusion_matrix.csv",
        "confusion_figure_path": artifact_dirs["figures_dir"] / f"{run_name}_test_confusion_matrix.png",
        "checkpoint_path": artifact_dirs["checkpoint_dir"] / f"{run_name}_best.pt",
    }


def find_existing_artifact(base_dir: Path, filename: str):
    direct_path = base_dir / filename
    if direct_path.exists():
        return direct_path

    matches = sorted(base_dir.glob(f"**/{filename}"))
    if matches:
        return matches[0]
    return None


def resolve_run_artifact_paths(results_dir: Path, run_name: str):
    metrics_dir = results_dir / "metrics"
    return {
        "metrics_path": find_existing_artifact(metrics_dir, f"{run_name}_metrics.csv"),
        "config_path": find_existing_artifact(metrics_dir, f"{run_name}_config.json"),
        "summary_path": find_existing_artifact(metrics_dir, f"{run_name}_summary.json"),
        "confusion_csv_path": find_existing_artifact(metrics_dir, f"{run_name}_test_confusion_matrix.csv"),
    }
