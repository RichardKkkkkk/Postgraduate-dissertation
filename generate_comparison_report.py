import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from textwrap import shorten

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PRIORITY_METRICS = [
    "val_acc",
    "val_loss",
    "test_acc",
    "test_loss",
    "train_acc",
    "train_loss",
]
CONFIG_PRIORITY_PREFIXES = ["model.", "training.", "dataset.", "device", "command"]
SLIDE_WIDTH_INCHES = 13.333
SLIDE_HEIGHT_INCHES = 7.5


@dataclass
class RunArtifacts:
    run_name: str
    label: str
    history: list[dict]
    config: dict
    summary: dict
    available_metrics: list[str]
    flat_config: dict
    model_name: str
    device: str


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a generalized comparison report for experiment runs."
    )
    parser.add_argument(
        "--run",
        dest="runs",
        action="append",
        required=True,
        help="Run spec in the form run_name or run_name=Display Label. Repeatable.",
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--report-name", type=str, default=None)
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        help="Metrics to compare. Defaults to the intersection of numeric metrics across runs.",
    )
    parser.add_argument(
        "--max-config-rows",
        type=int,
        default=12,
        help="Maximum number of varying config rows to show in the PPT summary table.",
    )
    parser.add_argument(
        "--skip-ppt",
        action="store_true",
        help="Skip PowerPoint generation and only write plots and summary files.",
    )
    return parser.parse_args()


def parse_run_spec(spec: str):
    if "=" in spec:
        run_name, label = spec.split("=", 1)
        return run_name.strip(), label.strip()
    return spec.strip(), spec.strip()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_history(path: Path):
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            parsed = {}
            for key, value in row.items():
                if value is None or value == "":
                    parsed[key] = value
                    continue
                try:
                    parsed[key] = float(value)
                except ValueError:
                    parsed[key] = value
            rows.append(parsed)
    return rows


def flatten_dict(data, prefix=""):
    flat = {}
    for key, value in data.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(flatten_dict(value, full_key))
        else:
            flat[full_key] = value
    return flat


def infer_model_name(config: dict):
    model_cfg = config.get("model", {})
    if isinstance(model_cfg, dict) and model_cfg.get("architecture"):
        return str(model_cfg["architecture"])

    command = str(config.get("command", ""))
    if "train_cnn_cifar10.py" in command:
        return "resnet18"
    if "train_cifar10.py" in command:
        return "vit"
    return "unknown"


def load_run_artifacts(results_dir: Path, run_name: str, label: str):
    metrics_dir = results_dir / "metrics"
    history_path = metrics_dir / f"{run_name}_metrics.csv"
    config_path = metrics_dir / f"{run_name}_config.json"
    summary_path = metrics_dir / f"{run_name}_summary.json"

    missing = [path.name for path in [history_path, config_path, summary_path] if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing artifacts for run '{run_name}': {', '.join(missing)}"
        )

    history = load_history(history_path)
    if not history:
        raise ValueError(f"No metric rows found in {history_path}")

    available_metrics = [
        key
        for key, value in history[0].items()
        if key != "epoch" and isinstance(value, (int, float))
    ]
    config = load_json(config_path)
    summary = load_json(summary_path)
    flat_config = flatten_dict(config)

    return RunArtifacts(
        run_name=run_name,
        label=label,
        history=history,
        config=config,
        summary=summary,
        available_metrics=available_metrics,
        flat_config=flat_config,
        model_name=infer_model_name(config),
        device=str(config.get("device", "unknown")),
    )


def order_metrics(metrics):
    seen = set()
    ordered = []
    for metric in PRIORITY_METRICS:
        if metric in metrics and metric not in seen:
            ordered.append(metric)
            seen.add(metric)
    for metric in metrics:
        if metric not in seen:
            ordered.append(metric)
            seen.add(metric)
    return ordered


def determine_metrics(runs, explicit_metrics=None):
    available_sets = [set(run.available_metrics) for run in runs]
    shared_metrics = set.intersection(*available_sets)
    if explicit_metrics:
        missing = [metric for metric in explicit_metrics if metric not in shared_metrics]
        if missing:
            raise ValueError(
                "These metrics are not present in every run: " + ", ".join(missing)
            )
        return order_metrics(explicit_metrics)
    return order_metrics([metric for metric in runs[0].available_metrics if metric in shared_metrics])


def is_percentage_metric(metric_name: str):
    tokens = ["acc", "accuracy", "precision", "recall", "f1"]
    lowered = metric_name.lower()
    return any(token in lowered for token in tokens)


def is_lower_better(metric_name: str):
    lowered = metric_name.lower()
    lower_keywords = ["loss", "error", "wer", "cer", "perplexity"]
    return any(keyword in lowered for keyword in lower_keywords)


def scale_metric_value(metric_name: str, value):
    if is_percentage_metric(metric_name):
        return value * 100.0
    return value


def metric_display_name(metric_name: str):
    tokens = metric_name.replace("_", " ").split()
    return " ".join(token.upper() if token in {"acc", "auc"} else token.capitalize() for token in tokens)


def format_metric_value(metric_name: str, value):
    scaled = scale_metric_value(metric_name, value)
    if is_percentage_metric(metric_name):
        return f"{scaled:.2f}%"
    return f"{scaled:.4f}"


def compute_metric_summary(history, metric_name: str):
    rows = [row for row in history if metric_name in row]
    values = [row[metric_name] for row in rows]
    if is_lower_better(metric_name):
        best_index = min(range(len(values)), key=values.__getitem__)
    else:
        best_index = max(range(len(values)), key=values.__getitem__)

    final_row = rows[-1]
    best_row = rows[best_index]
    return {
        "final_value": final_row[metric_name],
        "final_epoch": int(final_row["epoch"]),
        "best_value": best_row[metric_name],
        "best_epoch": int(best_row["epoch"]),
    }


def build_summary_rows(runs, metrics):
    rows = []
    for run in runs:
        row = {
            "run_name": run.run_name,
            "label": run.label,
            "model_name": run.model_name,
            "device": run.device,
        }
        for metric in metrics:
            metric_summary = compute_metric_summary(run.history, metric)
            row[f"final_{metric}"] = metric_summary["final_value"]
            row[f"best_{metric}"] = metric_summary["best_value"]
            row[f"best_epoch_{metric}"] = metric_summary["best_epoch"]
        rows.append(row)
    return rows


def choose_varying_config_keys(runs, max_rows):
    all_keys = sorted(set().union(*(run.flat_config.keys() for run in runs)))
    varying = []
    for key in all_keys:
        values = [run.flat_config.get(key) for run in runs]
        normalized = [json.dumps(value, sort_keys=True, ensure_ascii=False) for value in values]
        if len(set(normalized)) > 1:
            varying.append(key)

    def priority(key):
        for index, prefix in enumerate(CONFIG_PRIORITY_PREFIXES):
            if key.startswith(prefix):
                return index
        return len(CONFIG_PRIORITY_PREFIXES)

    varying.sort(key=lambda key: (priority(key), key))
    return varying[:max_rows], varying


def stringify_config_value(value):
    if value is None:
        return "None"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value)
    return str(value)


def ensure_report_dirs(results_dir: Path, report_name: str):
    report_dir = results_dir / "reports" / report_name
    figures_dir = report_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    return report_dir, figures_dir


def write_csv(path: Path, fieldnames, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_summary_outputs(report_dir: Path, runs, metrics, summary_rows, varying_keys):
    summary_csv_path = report_dir / "comparison_summary.csv"
    fieldnames = ["run_name", "label", "model_name", "device"]
    for metric in metrics:
        fieldnames.extend(
            [f"final_{metric}", f"best_{metric}", f"best_epoch_{metric}"]
        )
    write_csv(summary_csv_path, fieldnames, summary_rows)

    config_csv_path = report_dir / "config_comparison.csv"
    config_rows = []
    for key in varying_keys:
        row = {"config_key": key}
        for run in runs:
            row[run.label] = stringify_config_value(run.flat_config.get(key))
        config_rows.append(row)
    write_csv(
        config_csv_path,
        ["config_key"] + [run.label for run in runs],
        config_rows,
    )

    overview_path = report_dir / "overview.md"
    overview_lines = ["# Comparison Overview", ""]
    overview_lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    overview_lines.append("")
    overview_lines.append("## Runs")
    overview_lines.append("")
    for run in runs:
        overview_lines.append(f"- `{run.label}` (`{run.run_name}`), model: `{run.model_name}`, device: `{run.device}`")
    overview_lines.append("")
    overview_lines.append("## Metrics")
    overview_lines.append("")
    for row in summary_rows:
        overview_lines.append(f"### {row['label']}")
        for metric in metrics:
            overview_lines.append(
                f"- {metric_display_name(metric)}: final {format_metric_value(metric, row[f'final_{metric}'])}, "
                f"best {format_metric_value(metric, row[f'best_{metric}'])} at epoch {row[f'best_epoch_{metric}']}"
            )
        overview_lines.append("")
    overview_path.write_text("\n".join(overview_lines), encoding="utf-8")

    manifest_path = report_dir / "report_manifest.json"
    manifest = {
        "report_dir": str(report_dir),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "runs": [{"run_name": run.run_name, "label": run.label} for run in runs],
        "metrics": metrics,
        "summary_csv": str(summary_csv_path),
        "config_csv": str(config_csv_path),
        "overview_md": str(overview_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return summary_csv_path, config_csv_path, overview_path, manifest_path


def plot_metric(figures_dir: Path, runs, metric_name: str):
    plt.figure(figsize=(8, 5))
    for run in runs:
        epochs = [int(row["epoch"]) for row in run.history]
        values = [scale_metric_value(metric_name, row[metric_name]) for row in run.history]
        plt.plot(epochs, values, marker="o", linewidth=2, markersize=3, label=run.label)

    plt.xlabel("Epoch")
    plt.ylabel("Percentage (%)" if is_percentage_metric(metric_name) else "Value")
    plt.title(f"{metric_display_name(metric_name)} Comparison")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    figure_path = figures_dir / f"{metric_name}_comparison.png"
    plt.savefig(figure_path, dpi=160)
    plt.close()
    return figure_path


def build_insight_lines(summary_rows, metrics):
    lines = []
    for metric in metrics:
        if is_lower_better(metric):
            best_row = min(summary_rows, key=lambda row: row[f"final_{metric}"])
            comparator = "lowest"
        else:
            best_row = max(summary_rows, key=lambda row: row[f"final_{metric}"])
            comparator = "highest"
        lines.append(
            f"{comparator.title()} final {metric_display_name(metric)}: "
            f"{best_row['label']} ({format_metric_value(metric, best_row[f'final_{metric}'])})"
        )
    return lines


def chunked(items, chunk_size):
    for index in range(0, len(items), chunk_size):
        yield items[index : index + chunk_size]


def estimate_table_font_size(column_count, row_count):
    if column_count <= 4 and row_count <= 8:
        return 14
    if column_count <= 6 and row_count <= 10:
        return 12
    if column_count <= 8 and row_count <= 12:
        return 10
    return 9


def add_title_slide(prs, title, subtitle):
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    slide.shapes.title.text = title
    slide.placeholders[1].text = subtitle


def add_bullets_slide(prs, title, bullets):
    from pptx.util import Pt

    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = title
    text_frame = slide.shapes.placeholders[1].text_frame
    text_frame.clear()
    for index, bullet in enumerate(bullets):
        paragraph = text_frame.paragraphs[0] if index == 0 else text_frame.add_paragraph()
        paragraph.text = bullet
        paragraph.level = 0
        for run in paragraph.runs:
            run.font.size = Pt(20 if index == 0 else 18)


def style_table(table, font_size):
    from pptx.enum.text import MSO_ANCHOR
    from pptx.util import Pt

    for row_index, row in enumerate(table.rows):
        for cell in row.cells:
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            text_frame = cell.text_frame
            text_frame.word_wrap = True
            for paragraph in text_frame.paragraphs:
                for run in paragraph.runs:
                    run.font.size = Pt(font_size)
            if row_index == 0:
                for paragraph in text_frame.paragraphs:
                    for run in paragraph.runs:
                        run.font.bold = True


def set_table_column_widths(table, column_count, sticky_columns):
    from pptx.util import Inches

    total_width = 12.3
    sticky_width = 0.0

    if sticky_columns >= 1:
        first_width = min(3.2, max(2.2, total_width * 0.24))
        table.columns[0].width = Inches(first_width)
        sticky_width += first_width

    for column_index in range(1, sticky_columns):
        width = min(2.0, max(1.2, total_width * 0.13))
        table.columns[column_index].width = Inches(width)
        sticky_width += width

    remaining_columns = column_count - sticky_columns
    if remaining_columns <= 0:
        return

    remaining_width = max(3.2, total_width - sticky_width)
    per_width = remaining_width / remaining_columns
    for column_index in range(sticky_columns, column_count):
        table.columns[column_index].width = Inches(per_width)


def add_table_slides(
    prs,
    title,
    headers,
    rows,
    sticky_columns=1,
    max_total_columns=6,
    max_body_rows=10,
    font_size=None,
):
    from pptx.util import Inches

    if len(headers) <= max_total_columns:
        column_groups = [list(range(len(headers)))]
    else:
        scrollable_indexes = list(range(sticky_columns, len(headers)))
        scrollable_chunk_size = max(1, max_total_columns - sticky_columns)
        column_groups = []
        for chunk in chunked(scrollable_indexes, scrollable_chunk_size):
            column_groups.append(list(range(sticky_columns)) + chunk)

    row_groups = list(chunked(rows, max_body_rows)) or [[]]
    total_slides = len(column_groups) * len(row_groups)
    slide_number = 0

    for row_group in row_groups:
        for column_group in column_groups:
            slide_number += 1
            slide = prs.slides.add_slide(prs.slide_layouts[5])
            numbered_title = title if total_slides == 1 else f"{title} ({slide_number}/{total_slides})"
            slide.shapes.title.text = numbered_title

            subset_headers = [headers[index] for index in column_group]
            subset_rows = [
                [shorten(str(row[index]), width=36, placeholder="...") for index in column_group]
                for row in row_group
            ]
            local_font_size = font_size or estimate_table_font_size(
                column_count=len(subset_headers),
                row_count=len(subset_rows),
            )

            table = slide.shapes.add_table(
                len(subset_rows) + 1,
                len(subset_headers),
                Inches(0.45),
                Inches(1.2),
                Inches(12.3),
                Inches(5.8),
            ).table

            for column_index, header in enumerate(subset_headers):
                table.cell(0, column_index).text = header

            for row_index, row_values in enumerate(subset_rows, start=1):
                for column_index, value in enumerate(row_values):
                    table.cell(row_index, column_index).text = value

            effective_sticky_columns = min(sticky_columns, len(subset_headers))
            set_table_column_widths(table, len(subset_headers), effective_sticky_columns)
            style_table(table, local_font_size)


def add_image_slide(prs, title, image_path: Path, caption_lines):
    from pptx.util import Inches, Pt

    slide = prs.slides.add_slide(prs.slide_layouts[5])
    slide.shapes.title.text = title
    slide.shapes.add_picture(str(image_path), Inches(0.5), Inches(1.1), width=Inches(8.3))

    textbox = slide.shapes.add_textbox(Inches(9.0), Inches(1.4), Inches(3.3), Inches(4.8))
    text_frame = textbox.text_frame
    text_frame.clear()
    for index, line in enumerate(caption_lines):
        paragraph = text_frame.paragraphs[0] if index == 0 else text_frame.add_paragraph()
        paragraph.text = line
        paragraph.level = 0
        for run in paragraph.runs:
            run.font.size = Pt(16 if index == 0 else 12)


def export_ppt(
    report_dir: Path,
    report_name: str,
    title: str,
    runs,
    metrics,
    summary_rows,
    varying_rows,
    metric_figure_paths,
):
    try:
        from pptx import Presentation
    except ImportError as exc:
        raise RuntimeError(
            "python-pptx is required for PPT export. Install it or use --skip-ppt."
        ) from exc

    prs = Presentation()
    subtitle = f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')} | Runs: {', '.join(run.label for run in runs)}"
    add_title_slide(prs, title, subtitle)

    insight_lines = build_insight_lines(summary_rows, metrics)
    add_bullets_slide(prs, "Key Takeaways", insight_lines)

    primary_metrics = metrics[: min(4, len(metrics))]
    summary_headers = ["Run", "Model", "Device"] + [
        f"Final {metric_display_name(metric)}" for metric in primary_metrics
    ]
    summary_table_rows = []
    for row in summary_rows:
        table_row = [row["label"], row["model_name"], row["device"]]
        for metric in primary_metrics:
            table_row.append(format_metric_value(metric, row[f"final_{metric}"]))
        summary_table_rows.append(table_row)
    add_table_slides(
        prs,
        "Result Snapshot",
        summary_headers,
        summary_table_rows,
        sticky_columns=3,
        max_total_columns=6,
        max_body_rows=8,
    )

    if varying_rows:
        config_headers = ["Config"] + [run.label for run in runs]
        config_table_rows = []
        for key, values in varying_rows:
            config_table_rows.append([key, *values])
        add_table_slides(
            prs,
            "Config Differences",
            config_headers,
            config_table_rows,
            sticky_columns=1,
            max_total_columns=5,
            max_body_rows=8,
            font_size=10,
        )

    for metric in metrics:
        caption_lines = [metric_display_name(metric)]
        for row in summary_rows:
            caption_lines.append(
                f"{row['label']}: final {format_metric_value(metric, row[f'final_{metric}'])}, "
                f"best {format_metric_value(metric, row[f'best_{metric}'])} "
                f"(epoch {row[f'best_epoch_{metric}']})"
            )
        add_image_slide(
            prs,
            f"{metric_display_name(metric)} Curve",
            metric_figure_paths[metric],
            caption_lines,
        )

    ppt_path = report_dir / f"{report_name}.pptx"
    prs.save(ppt_path)
    return ppt_path


def build_varying_rows(runs, varying_keys, max_rows):
    rows = []
    for key in varying_keys[:max_rows]:
        values = [stringify_config_value(run.flat_config.get(key)) for run in runs]
        rows.append((key, values))
    return rows


def make_default_report_name(run_specs):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_names = "_vs_".join(spec[0] for spec in run_specs[:2])
    short_names = short_names[:48]
    return f"comparison_{short_names}_{timestamp}"


def main():
    args = parse_args()
    run_specs = [parse_run_spec(spec) for spec in args.runs]
    report_name = args.report_name or make_default_report_name(run_specs)
    title = args.title or "Experiment Comparison Report"

    runs = [
        load_run_artifacts(args.results_dir, run_name, label)
        for run_name, label in run_specs
    ]
    metrics = determine_metrics(runs, explicit_metrics=args.metrics)
    if not metrics:
        raise ValueError("No shared numeric metrics were found across the selected runs.")

    report_dir, figures_dir = ensure_report_dirs(args.results_dir, report_name)
    summary_rows = build_summary_rows(runs, metrics)
    selected_varying_keys, all_varying_keys = choose_varying_config_keys(
        runs, args.max_config_rows
    )
    varying_rows = build_varying_rows(runs, selected_varying_keys, args.max_config_rows)

    metric_figure_paths = {}
    for metric in metrics:
        metric_figure_paths[metric] = plot_metric(figures_dir, runs, metric)

    summary_csv_path, config_csv_path, overview_path, manifest_path = save_summary_outputs(
        report_dir=report_dir,
        runs=runs,
        metrics=metrics,
        summary_rows=summary_rows,
        varying_keys=all_varying_keys,
    )

    print(f"Report directory: {report_dir}")
    print(f"Summary CSV: {summary_csv_path}")
    print(f"Config CSV: {config_csv_path}")
    print(f"Overview Markdown: {overview_path}")
    print(f"Manifest JSON: {manifest_path}")
    for metric, path in metric_figure_paths.items():
        print(f"Figure ({metric}): {path}")

    if not args.skip_ppt:
        ppt_path = export_ppt(
            report_dir=report_dir,
            report_name=report_name,
            title=title,
            runs=runs,
            metrics=metrics,
            summary_rows=summary_rows,
            varying_rows=varying_rows,
            metric_figure_paths=metric_figure_paths,
        )
        print(f"PPTX: {ppt_path}")


if __name__ == "__main__":
    main()
