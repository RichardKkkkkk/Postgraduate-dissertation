from pathlib import Path


def _clean_name(value: str | None):
    if value is None:
        return None
    cleaned = str(value).strip()
    if not cleaned:
        return None
    return cleaned.replace(" ", "_")


def normalize_experiment_name(experiment_name: str | None, dataset_name: str | None = None):
    cleaned = _clean_name(experiment_name)
    if cleaned:
        return cleaned
    dataset_cleaned = _clean_name(dataset_name)
    if dataset_cleaned:
        return dataset_cleaned
    return "default_experiment"


def get_experiment_dirs(
    results_dir: Path,
    checkpoint_dir: Path,
    experiment_name: str | None,
    dataset_name: str | None = None,
):
    experiment_slug = normalize_experiment_name(experiment_name, dataset_name)
    return {
        "experiment_name": experiment_slug,
        "results_experiment_dir": results_dir / experiment_slug,
        "checkpoints_experiment_dir": checkpoint_dir / experiment_slug,
    }


def get_model_artifact_dirs(
    results_dir: Path,
    checkpoint_dir: Path,
    model_name: str,
    experiment_name: str | None = None,
    dataset_name: str | None = None,
):
    model_slug = _clean_name(model_name) or "unknown_model"
    experiment_dirs = get_experiment_dirs(results_dir, checkpoint_dir, experiment_name, dataset_name)
    return {
        "experiment_name": experiment_dirs["experiment_name"],
        "results_experiment_dir": experiment_dirs["results_experiment_dir"],
        "checkpoints_experiment_dir": experiment_dirs["checkpoints_experiment_dir"],
        "metrics_dir": experiment_dirs["results_experiment_dir"] / "metrics" / model_slug,
        "figures_dir": experiment_dirs["results_experiment_dir"] / "figures" / model_slug,
        "checkpoint_dir": experiment_dirs["checkpoints_experiment_dir"] / model_slug,
    }


def build_run_artifact_paths(
    results_dir: Path,
    checkpoint_dir: Path,
    model_name: str,
    run_name: str,
    experiment_name: str | None = None,
    dataset_name: str | None = None,
):
    artifact_dirs = get_model_artifact_dirs(
        results_dir=results_dir,
        checkpoint_dir=checkpoint_dir,
        model_name=model_name,
        experiment_name=experiment_name,
        dataset_name=dataset_name,
    )
    run_slug = _clean_name(run_name) or "unnamed_run"
    return {
        "experiment_name": artifact_dirs["experiment_name"],
        "results_experiment_dir": artifact_dirs["results_experiment_dir"],
        "checkpoints_experiment_dir": artifact_dirs["checkpoints_experiment_dir"],
        "metrics_dir": artifact_dirs["metrics_dir"],
        "figures_dir": artifact_dirs["figures_dir"],
        "checkpoint_dir": artifact_dirs["checkpoint_dir"],
        "metrics_path": artifact_dirs["metrics_dir"] / f"{run_slug}_metrics.csv",
        "config_path": artifact_dirs["metrics_dir"] / f"{run_slug}_config.json",
        "summary_path": artifact_dirs["metrics_dir"] / f"{run_slug}_summary.json",
        "confusion_csv_path": artifact_dirs["metrics_dir"] / f"{run_slug}_test_confusion_matrix.csv",
        "confusion_figure_path": artifact_dirs["figures_dir"] / f"{run_slug}_test_confusion_matrix.png",
        "loss_figure_path": artifact_dirs["figures_dir"] / f"{run_slug}_loss.png",
        "accuracy_figure_path": artifact_dirs["figures_dir"] / f"{run_slug}_accuracy.png",
        "checkpoint_path": artifact_dirs["checkpoint_dir"] / f"{run_slug}_best.pt",
    }


def build_report_artifact_dirs(results_dir: Path, report_name: str, experiment_name: str | None = None):
    report_slug = _clean_name(report_name) or "unnamed_report"
    experiment_slug = _clean_name(experiment_name)
    if experiment_slug:
        report_dir = results_dir / experiment_slug / "reports" / report_slug
    else:
        report_dir = results_dir / "reports" / report_slug
    figures_dir = report_dir / "figures"
    return {
        "report_name": report_slug,
        "experiment_name": experiment_slug,
        "report_dir": report_dir,
        "figures_dir": figures_dir,
    }


def find_existing_artifact(base_dir: Path, filename: str):
    direct_path = base_dir / filename
    if direct_path.exists():
        return direct_path

    matches = sorted(base_dir.glob(f"**/{filename}"))
    if matches:
        return matches[0]
    return None


def _resolve_experiment_folder_paths(results_dir: Path, run_name: str, experiment_name: str | None = None):
    run_slug = _clean_name(run_name)
    if not run_slug:
        return None

    candidate_roots = []
    experiment_slug = _clean_name(experiment_name)
    if experiment_slug:
        candidate_roots.append(results_dir / experiment_slug)
    candidate_roots.append(results_dir)
    if not experiment_slug and results_dir.exists():
        candidate_roots.extend(
            path
            for path in sorted(results_dir.iterdir())
            if path.is_dir() and (path / "metrics").is_dir()
        )

    resolved_candidates = []
    for root in candidate_roots:
        metrics_dir = root / "metrics"
        if not metrics_dir.exists():
            continue
        metrics_path = find_existing_artifact(metrics_dir, f"{run_slug}_metrics.csv")
        if metrics_path is None:
            continue
        figures_dir = root / "figures"
        resolved_candidates.append(
            {
                "experiment_name": root.name if root != results_dir else experiment_slug,
                "results_experiment_dir": root,
                "metrics_dir": metrics_dir,
                "figures_dir": figures_dir,
                "checkpoint_dir": None,
                "metrics_path": metrics_path,
                "config_path": find_existing_artifact(metrics_dir, f"{run_slug}_config.json"),
                "summary_path": find_existing_artifact(metrics_dir, f"{run_slug}_summary.json"),
                "confusion_csv_path": find_existing_artifact(metrics_dir, f"{run_slug}_test_confusion_matrix.csv"),
                "confusion_figure_path": find_existing_artifact(figures_dir, f"{run_slug}_test_confusion_matrix.png"),
                "loss_figure_path": find_existing_artifact(figures_dir, f"{run_slug}_loss.png"),
                "accuracy_figure_path": find_existing_artifact(figures_dir, f"{run_slug}_accuracy.png"),
                "checkpoint_path": None,
            }
        )

    if len(resolved_candidates) > 1:
        matching_paths = ", ".join(str(item["metrics_path"]) for item in resolved_candidates)
        raise ValueError(
            f"Run name '{run_slug}' is ambiguous across experiments: {matching_paths}. "
            "Pass --experiment-name or use unique run names."
        )
    return resolved_candidates[0] if resolved_candidates else None


def _resolve_accidental_run_folder_paths(results_dir: Path, run_name: str, experiment_name: str | None = None):
    run_slug = _clean_name(run_name)
    if not run_slug:
        return None

    candidate_roots = []
    experiment_slug = _clean_name(experiment_name)
    if experiment_slug:
        candidate_roots.append(results_dir / experiment_slug)
    candidate_roots.append(results_dir)

    for root in candidate_roots:
        matches = sorted(root.glob(f"**/{run_slug}/summary.json"))
        for summary_path in matches:
            run_dir = summary_path.parent
            if run_dir.name != run_slug:
                continue
            figures_dir = None
            parts = run_dir.parts
            if "runs" in parts:
                runs_index = parts.index("runs")
                prefix = Path(*parts[:runs_index])
                suffix = parts[runs_index + 1 :]
                if len(suffix) >= 2:
                    figures_dir = prefix / "figures" / Path(*suffix)
            return {
                "experiment_name": run_dir.parents[2].name if len(run_dir.parents) >= 3 else None,
                "results_experiment_dir": run_dir.parents[2] if len(run_dir.parents) >= 3 else None,
                "metrics_dir": run_dir,
                "figures_dir": figures_dir,
                "checkpoint_dir": None,
                "metrics_path": run_dir / "metrics.csv",
                "config_path": run_dir / "config.json",
                "summary_path": summary_path,
                "confusion_csv_path": run_dir / "test_confusion_matrix.csv",
                "confusion_figure_path": figures_dir / "test_confusion_matrix.png" if figures_dir else None,
                "loss_figure_path": figures_dir / "loss.png" if figures_dir else None,
                "accuracy_figure_path": figures_dir / "accuracy.png" if figures_dir else None,
                "checkpoint_path": None,
            }
    return None


def _resolve_legacy_root_paths(results_dir: Path, run_name: str):
    metrics_dir = results_dir / "metrics"
    figures_dir = results_dir / "figures"
    return {
        "experiment_name": None,
        "results_experiment_dir": results_dir,
        "metrics_dir": metrics_dir,
        "figures_dir": figures_dir,
        "checkpoint_dir": None,
        "metrics_path": find_existing_artifact(metrics_dir, f"{run_name}_metrics.csv"),
        "config_path": find_existing_artifact(metrics_dir, f"{run_name}_config.json"),
        "summary_path": find_existing_artifact(metrics_dir, f"{run_name}_summary.json"),
        "confusion_csv_path": find_existing_artifact(metrics_dir, f"{run_name}_test_confusion_matrix.csv"),
        "confusion_figure_path": find_existing_artifact(figures_dir, f"{run_name}_test_confusion_matrix.png"),
        "loss_figure_path": find_existing_artifact(figures_dir, f"{run_name}_loss.png"),
        "accuracy_figure_path": find_existing_artifact(figures_dir, f"{run_name}_accuracy.png"),
        "checkpoint_path": None,
    }


def resolve_run_artifact_paths(results_dir: Path, run_name: str, experiment_name: str | None = None):
    experiment_layout = _resolve_experiment_folder_paths(results_dir, run_name, experiment_name)
    if experiment_layout is not None:
        return experiment_layout

    accidental_run_layout = _resolve_accidental_run_folder_paths(results_dir, run_name, experiment_name)
    if accidental_run_layout is not None:
        return accidental_run_layout

    return _resolve_legacy_root_paths(results_dir, run_name)
