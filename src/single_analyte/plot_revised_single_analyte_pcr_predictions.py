#!/usr/bin/env python3
"""Plot revised single-analyte PCR actual-vs-predicted training curves."""

from __future__ import annotations

# --- path bootstrap (auto-added during repo reorg): make sibling modules importable ---
import os as _os, sys as _sys
_SRC_DIR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _sub in ("core", "features", "day1_da_only", "single_analyte", "multi_analyte", "models"):
    _p = _os.path.join(_SRC_DIR, _sub)
    if _os.path.isdir(_p) and _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end path bootstrap ---


import argparse
import csv
from pathlib import Path

import numpy as np

from fit_revised_single_analyte_pcr import (
    ANALYTE_COLORS,
    ANALYTE_LABELS,
    ANALYTE_ORDER,
    CURRENT_UNIT,
    DEFAULT_DATA_ROOT,
    DEFAULT_OUTPUT_DIR,
    ROOT,
    SWEEP_LABELS,
    TECHNIQUE_CONFIGS,
    chunks,
    clean_value,
    configure_plot_style,
    current_polarity_label,
    fit_pcr_models,
    load_trace_entries,
    model_lookup,
    params_from_summary,
    plot_condition_fit_pages,
    set_curve_y_limits,
    style_curve_axis,
)

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


DEFAULT_SUMMARY = DEFAULT_OUTPUT_DIR / "selected_setup_summary.csv"


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return [
            {key.strip(): (value.strip() if value is not None else "") for key, value in row.items()}
            for row in csv.DictReader(handle)
            if any(value for value in row.values())
        ]


def selected_setup_rows(summary_path: Path) -> dict[str, dict[str, str]]:
    rows = read_csv_dicts(summary_path)
    selected = {row["analyte"]: row for row in rows}
    missing = [analyte for analyte in ANALYTE_ORDER if analyte not in selected]
    if missing:
        raise ValueError(f"Missing selected setup rows for: {', '.join(missing)}")
    return selected


def plot_intro_page(
    pdf: PdfPages,
    analyte: str,
    selected_row: dict[str, str],
    manifest_rows: list[dict[str, object]],
    model_count: int,
) -> None:
    concentrations = [float(row["concentration_uM"]) for row in manifest_rows]
    lines = [
        f"{ANALYTE_LABELS[analyte]} ({analyte}) PCR Predicted vs Actual",
        "",
        "Purpose:",
        "- Verify the training behavior of the concentration-to-curve PCR pipeline.",
        "- For each held training concentration, the plotted prediction is generated from concentration only.",
        "- Prediction equation: current(V,c)=mean_curve(V)+sum_j(polyval(score_coefficients_j,c)*loading_j(V)).",
        "",
        "Data used:",
        "- CV normal: all current channels E1-E8.",
        "- CV-GC: generator channels only, E1/E3/E5/E7.",
        "- Anodic and cathodic sweeps are fit separately.",
        "- Each file uses the last complete CV cycle only.",
        f"- Current polarity: {current_polarity_label(analyte)}.",
        f"- Conditions: {', '.join(str(int(row['condition'])) for row in manifest_rows)}.",
        f"- Concentrations: {', '.join(clean_value(value) for value in concentrations)} uM.",
        "",
        "Selected PCR setup:",
        f"- setup_id={selected_row['setup_id']}",
        f"- PCs={selected_row['n_components']}, score trend degree={selected_row['score_trend_degree']}",
        f"- smooth window/polyorder={selected_row['smooth_window']}/{selected_row['smooth_polyorder']}",
        f"- grid points={selected_row['grid_points']}",
        f"- fitted groups={model_count}",
        "",
        "Selected setup metrics:",
        f"- avg R2={float(selected_row['avg_r2']):.4f}",
        f"- median R2={float(selected_row['median_r2']):.4f}",
        f"- min R2={float(selected_row['min_r2']):.4f}",
        f"- avg normalized RMSE={float(selected_row['avg_normalized_rmse']):.4g}",
        f"- avg RMSE={float(selected_row['avg_rmse_uA']):.4g} {CURRENT_UNIT}",
    ]
    if analyte == "AA":
        lines.insert(12, "- AA conditions 11 and 12 are excluded because they were flagged as shorted.")

    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.06, 0.94, lines[0], fontsize=17, fontweight="semibold", va="top")
    fig.text(0.06, 0.88, "\n".join(lines[2:]), fontsize=9.2, va="top", family="monospace")
    pdf.savefig(fig)
    plt.close(fig)


def add_prediction_metric_box(ax: plt.Axes, model: dict[str, object]) -> None:
    ax.text(
        0.02,
        0.97,
        (
            f"R2={float(model['r2']):.3f}\n"
            f"RMSE={float(model['rmse_uA']):.3g} {CURRENT_UNIT}\n"
            f"nRMSE={float(model['normalized_rmse']):.3g}\n"
            f"PC trend R2={float(model['avg_pc_trend_r2']):.3f}"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.2,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.76},
    )


def plot_overlay_page(
    pdf: PdfPages,
    analyte: str,
    models: list[dict[str, object]],
    technique: str,
    electrodes: list[str],
    page_label: str,
) -> None:
    lookup = model_lookup(models)
    fig, axes = plt.subplots(len(electrodes), 2, figsize=(12.4, max(7.2, 3.05 * len(electrodes) + 1.25)), squeeze=False)
    fig.suptitle(
        f"{analyte} concentration-to-curve PCR overlay | {TECHNIQUE_CONFIGS[technique]['label']} | {page_label}",
        fontweight="semibold",
    )

    all_concentrations = []
    for model in models:
        if model["technique"] == technique:
            all_concentrations.extend(float(value) for value in model["concentrations"])
    c_min = min(all_concentrations)
    c_max = max(all_concentrations)
    norm = Normalize(vmin=c_min, vmax=c_max)
    cmap = plt.cm.viridis

    for row_index, electrode in enumerate(electrodes):
        for col_index, sweep in enumerate(("anodic", "cathodic")):
            ax = axes[row_index, col_index]
            style_curve_axis(ax, show_xlabel=row_index == len(electrodes) - 1, show_ylabel=col_index == 0)
            ax.set_title(f"{electrode} | {SWEEP_LABELS[sweep]}", fontsize=7.5)
            model = lookup.get((electrode, technique, sweep))
            if model is None:
                ax.text(0.5, 0.5, "No fitted group", transform=ax.transAxes, ha="center", va="center")
                continue
            concentrations = np.asarray(model["concentrations"], dtype=float)
            y_values_for_limits = []
            for index, concentration in enumerate(concentrations):
                color = cmap(norm(float(concentration)))
                actual = model["data_matrix"][:, index]
                predicted = model["predicted_matrix"][:, index]
                y_values_for_limits.extend([actual, predicted])
                ax.plot(model["x_grid"], actual, color=color, lw=0.85, alpha=0.70)
                ax.plot(model["x_grid"], predicted, color=color, lw=1.05, ls="--", alpha=0.98)
            if row_index == 0 and col_index == 0:
                ax.plot([], [], color="#444444", lw=0.95, label="actual")
                ax.plot([], [], color="#444444", lw=1.05, ls="--", label="PCR predicted")
                ax.legend(loc="best", fontsize=6)
            set_curve_y_limits(ax, y_values_for_limits)
            add_prediction_metric_box(ax, model)

    scalar = ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array([])
    fig.subplots_adjust(left=0.07, right=0.86, top=0.90, bottom=0.08, hspace=0.52, wspace=0.24)
    cbar_ax = fig.add_axes([0.895, 0.20, 0.018, 0.58])
    cbar = fig.colorbar(scalar, cax=cbar_ax)
    cbar.set_label(f"{analyte} concentration (uM)")
    pdf.savefig(fig)
    plt.close(fig)


def plot_overlay_pages(pdf: PdfPages, analyte: str, models: list[dict[str, object]]) -> None:
    normal_electrodes = list(TECHNIQUE_CONFIGS["cv_normal"]["electrode_columns"].keys())
    generator_electrodes = list(TECHNIQUE_CONFIGS["cv_gc"]["electrode_columns"].keys())

    for electrode_group in chunks(normal_electrodes, 2):
        plot_overlay_page(pdf, analyte, models, "cv_normal", electrode_group, "-".join(electrode_group))
    for electrode_group in chunks(generator_electrodes, 2):
        plot_overlay_page(pdf, analyte, models, "cv_gc", electrode_group, " / ".join(electrode_group))


def write_prediction_pdf(
    output_path: Path,
    analyte: str,
    selected_row: dict[str, str],
    models: list[dict[str, object]],
    manifest_rows: list[dict[str, object]],
) -> None:
    with PdfPages(output_path) as pdf:
        plot_intro_page(pdf, analyte, selected_row, manifest_rows, len(models))
        plot_overlay_pages(pdf, analyte, models)
        plot_condition_fit_pages(pdf, analyte, models)


def run(args: argparse.Namespace) -> None:
    configure_plot_style()
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    selected = selected_setup_rows(args.summary_csv.resolve())

    outputs = []
    for analyte in ANALYTE_ORDER:
        entries, manifest_rows, _skipped = load_trace_entries(data_root, analyte)
        params = params_from_summary(selected[analyte])
        models = fit_pcr_models(entries, params, scopes={"electrode"})
        output_path = data_root / f"revised_{analyte}_pcr_predicted_vs_actual.pdf"
        write_prediction_pdf(output_path, analyte, selected[analyte], models, manifest_rows)
        outputs.append(output_path)
        print(f"{analyte}: wrote {output_path}")

    print("Predicted-vs-actual PDFs:")
    for output in outputs:
        print(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
