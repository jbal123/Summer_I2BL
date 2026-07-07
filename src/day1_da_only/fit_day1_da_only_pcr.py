#!/usr/bin/env python3
"""Fit PCR models to Day 1 DA-only CV curves.

PCR here treats each full CV curve as one signal vector. For each
technique/sweep/electrode group it:

1. Aligns and optionally smooths the current vectors.
2. Runs PCA on the centered current matrix.
3. Fits each retained PC score as a polynomial function of DA concentration.
4. Reconstructs predicted curves from the interpolated PC scores.
"""

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
import json
import os
import tempfile
from pathlib import Path

import numpy as np

CACHE_ROOT = Path(tempfile.gettempdir()) / "microneedlearrayml_matplotlib"
(CACHE_ROOT / "mplconfig").mkdir(parents=True, exist_ok=True)
(CACHE_ROOT / "xdg_cache").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT / "xdg_cache"))

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.backends.backend_pdf import PdfPages

from day1_da_only_pcr_core import (
    CURRENT_UNIT,
    DEFAULT_GRID_POINTS,
    DEFAULT_MANIFEST,
    DEFAULT_N_COMPONENTS,
    DEFAULT_SCORE_TREND_DEGREE,
    DEFAULT_SMOOTH_POLYORDER,
    DEFAULT_SMOOTH_WINDOW,
    DEFAULT_SUBSET_DIR,
    PANEL_ORDER,
    SWEEPS,
    TECHNIQUES,
    build_prediction_model,
    clean_value,
    fit_pcr_models,
    format_float,
    format_series,
    json_default,
    load_trace_entries,
    predict_curves,
    read_manifest,
    slug_float,
    write_prediction_csv,
)


DEFAULT_OUTPUT_DIR = DEFAULT_SUBSET_DIR / "pcr_analysis"


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "legend.frameon": False,
            "lines.linewidth": 1.3,
        }
    )


def style_curve_axis(ax: plt.Axes) -> None:
    ax.set_xlabel("Potential (V)")
    ax.set_ylabel(f"Current ({CURRENT_UNIT})")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", direction="in")
    ax.tick_params(axis="both", which="minor", direction="in")
    ax.minorticks_on()
    formatter = mticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-3, 3))
    ax.yaxis.set_major_formatter(formatter)


def style_score_axis(ax: plt.Axes) -> None:
    ax.set_xlabel("DA concentration (uM)")
    ax.set_ylabel("PCA score")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", direction="in")
    ax.tick_params(axis="both", which="minor", direction="in")
    ax.minorticks_on()


def style_loading_axis(ax: plt.Axes) -> None:
    ax.set_xlabel("Potential (V)")
    ax.set_ylabel("PCA loading")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", direction="in")
    ax.tick_params(axis="both", which="minor", direction="in")
    ax.minorticks_on()


def model_lookup(models: list[dict[str, object]]) -> dict[tuple[str, str, str], dict[str, object]]:
    return {
        (str(model["electrode"]), str(model["technique"]), str(model["sweep"])): model
        for model in models
    }


def write_json(data: dict[str, object], output_path: Path) -> None:
    with output_path.open("w") as handle:
        json.dump(data, handle, indent=2, default=json_default)
        handle.write("\n")


def write_score_csv(models: list[dict[str, object]], output_path: Path) -> None:
    fieldnames = [
        "scope",
        "technique",
        "technique_label",
        "sweep",
        "sweep_label",
        "electrode",
        "component",
        "n_components",
        "requested_n_components",
        "score_trend_degree",
        "score_trend_degree_effective",
        "explained_variance_ratio",
        "cumulative_explained_variance_ratio",
        "pearson_r",
        "spearman_r",
        "trend_r2",
        "trend_rmse",
        "trend_coefficients_desc",
        "da_values",
        "scores",
        "predicted_scores",
        "n_concentrations",
        "n_voltages",
        "smooth_window",
        "smooth_polyorder",
        "actual_smooth_window",
        "grid_points",
    ]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for model in models:
            for row in model["score_correlations"]:
                writer.writerow(
                    {
                        "scope": model["scope"],
                        "technique": model["technique"],
                        "technique_label": model["technique_label"],
                        "sweep": model["sweep"],
                        "sweep_label": model["sweep_label"],
                        "electrode": model["electrode"],
                        "component": row["component"],
                        "n_components": model["n_components"],
                        "requested_n_components": model["requested_n_components"],
                        "score_trend_degree": model["score_trend_degree"],
                        "score_trend_degree_effective": model["score_trend_degree_effective"],
                        "explained_variance_ratio": format_float(float(row["explained_variance_ratio"])),
                        "cumulative_explained_variance_ratio": format_float(
                            float(model["cumulative_explained_variance_ratio"])
                        ),
                        "pearson_r": format_float(float(row["pearson_r"])),
                        "spearman_r": format_float(float(row["spearman_r"])),
                        "trend_r2": format_float(float(row["trend_r2"])),
                        "trend_rmse": format_float(float(row["trend_rmse"])),
                        "trend_coefficients_desc": format_series(row["trend_coefficients_desc"]),
                        "da_values": format_series(model["concentrations"]),
                        "scores": format_series(row["scores"]),
                        "predicted_scores": format_series(row["predicted_scores"]),
                        "n_concentrations": model["n_concentrations"],
                        "n_voltages": model["n_voltages"],
                        "smooth_window": model["smooth_window"],
                        "smooth_polyorder": model["smooth_polyorder_effective"],
                        "actual_smooth_window": model["actual_smooth_window"],
                        "grid_points": model["grid_points"],
                    }
                )


def write_metrics_csv(models: list[dict[str, object]], output_path: Path) -> None:
    fieldnames = [
        "scope",
        "technique",
        "technique_label",
        "sweep",
        "sweep_label",
        "electrode",
        "method",
        "n_components",
        "requested_n_components",
        "score_trend_degree",
        "score_trend_degree_effective",
        "cumulative_explained_variance_ratio",
        "avg_abs_pc_pearson",
        "avg_abs_pc_spearman",
        "avg_pc_trend_r2",
        "n_concentrations",
        "n_voltages",
        "rmse_uA",
        "mae_uA",
        "max_abs_residual_uA",
        "normalized_rmse",
        "r2",
        "conditions",
        "da_values",
        "per_condition_rmse_uA",
        "per_condition_mae_uA",
        "per_condition_r2",
        "smooth_window",
        "smooth_polyorder",
        "actual_smooth_window",
        "grid_points",
    ]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for model in models:
            writer.writerow(
                {
                    "scope": model["scope"],
                    "technique": model["technique"],
                    "technique_label": model["technique_label"],
                    "sweep": model["sweep"],
                    "sweep_label": model["sweep_label"],
                    "electrode": model["electrode"],
                    "method": "pcr",
                    "n_components": model["n_components"],
                    "requested_n_components": model["requested_n_components"],
                    "score_trend_degree": model["score_trend_degree"],
                    "score_trend_degree_effective": model["score_trend_degree_effective"],
                    "cumulative_explained_variance_ratio": format_float(
                        float(model["cumulative_explained_variance_ratio"])
                    ),
                    "avg_abs_pc_pearson": format_float(float(model["avg_abs_pc_pearson"])),
                    "avg_abs_pc_spearman": format_float(float(model["avg_abs_pc_spearman"])),
                    "avg_pc_trend_r2": format_float(float(model["avg_pc_trend_r2"])),
                    "n_concentrations": model["n_concentrations"],
                    "n_voltages": model["n_voltages"],
                    "rmse_uA": format_float(float(model["rmse_uA"])),
                    "mae_uA": format_float(float(model["mae_uA"])),
                    "max_abs_residual_uA": format_float(float(model["max_abs_residual_uA"])),
                    "normalized_rmse": format_float(float(model["normalized_rmse"])),
                    "r2": format_float(float(model["r2"])),
                    "conditions": format_series(model["conditions"]),
                    "da_values": format_series(model["concentrations"]),
                    "per_condition_rmse_uA": format_series(model["per_condition_rmse_uA"]),
                    "per_condition_mae_uA": format_series(model["per_condition_mae_uA"]),
                    "per_condition_r2": format_series(model["per_condition_r2"]),
                    "smooth_window": model["smooth_window"],
                    "smooth_polyorder": model["smooth_polyorder_effective"],
                    "actual_smooth_window": model["actual_smooth_window"],
                    "grid_points": model["grid_points"],
                }
            )


def plot_score_relationships(
    models: list[dict[str, object]],
    output_path: Path,
    title_prefix: str,
) -> None:
    grouped = {}
    for model in models:
        grouped.setdefault(str(model["electrode"]), []).append(model)

    with PdfPages(output_path) as pdf:
        for electrode, group_models in sorted(grouped.items()):
            lookup = model_lookup(group_models)
            fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.6))
            fig.suptitle(f"{title_prefix} score correlations | {electrode}", fontweight="semibold")
            for ax, (technique, sweep) in zip(axes.ravel(), PANEL_ORDER):
                model = lookup.get((electrode, technique, sweep))
                style_score_axis(ax)
                ax.set_title(f"{TECHNIQUES[technique]['label']} - {SWEEPS[sweep]}")
                if model is None:
                    ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                    continue

                concentrations = np.asarray(model["concentrations"], dtype=float)
                x_fit = np.linspace(float(np.min(concentrations)), float(np.max(concentrations)), 200)
                colors = plt.cm.tab10(np.linspace(0, 1, int(model["n_components"])))
                for score_row, color in zip(model["score_correlations"], colors):
                    component = int(score_row["component"])
                    coeffs = np.asarray(score_row["trend_coefficients_desc"], dtype=float)
                    ax.scatter(
                        concentrations,
                        score_row["scores"],
                        s=22,
                        color=color,
                        label=(
                            f"PC{component} r={float(score_row['pearson_r']):.2f} "
                            f"fitR2={float(score_row['trend_r2']):.2f}"
                        ),
                    )
                    ax.plot(x_fit, np.polyval(coeffs, x_fit), color=color, lw=1.2)
                ax.legend(loc="best", fontsize=6)
                ax.text(
                    0.02,
                    0.97,
                    (
                        f"Curve R2={float(model['r2']):.3f}\n"
                        f"cum var={float(model['cumulative_explained_variance_ratio']):.3f}"
                    ),
                    ha="left",
                    va="top",
                    transform=ax.transAxes,
                    fontsize=8,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75},
                )
            fig.tight_layout(rect=(0, 0.02, 1, 0.94))
            pdf.savefig(fig)
            plt.close(fig)


def plot_loadings(
    models: list[dict[str, object]],
    output_path: Path,
    title_prefix: str,
) -> None:
    grouped = {}
    for model in models:
        grouped.setdefault(str(model["electrode"]), []).append(model)

    with PdfPages(output_path) as pdf:
        for electrode, group_models in sorted(grouped.items()):
            lookup = model_lookup(group_models)
            fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.6))
            fig.suptitle(f"{title_prefix} loadings | {electrode}", fontweight="semibold")
            for ax, (technique, sweep) in zip(axes.ravel(), PANEL_ORDER):
                model = lookup.get((electrode, technique, sweep))
                style_loading_axis(ax)
                ax.set_title(f"{TECHNIQUES[technique]['label']} - {SWEEPS[sweep]}")
                if model is None:
                    ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                    continue
                colors = plt.cm.tab10(np.linspace(0, 1, int(model["n_components"])))
                for index, color in enumerate(colors):
                    explained = float(model["explained_variance_ratio"][index])
                    ax.plot(
                        model["x_grid"],
                        model["pca_loadings"][index],
                        color=color,
                        lw=1.2,
                        label=f"PC{index + 1} ({explained:.1%})",
                    )
                ax.legend(loc="best", fontsize=7)
            fig.tight_layout(rect=(0, 0.02, 1, 0.94))
            pdf.savefig(fig)
            plt.close(fig)


def plot_residual_heatmaps(
    models: list[dict[str, object]],
    output_path: Path,
    title_prefix: str,
) -> None:
    grouped = {}
    for model in models:
        grouped.setdefault(str(model["electrode"]), []).append(model)
    with PdfPages(output_path) as pdf:
        for electrode, group_models in sorted(grouped.items()):
            lookup = model_lookup(group_models)
            fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2))
            fig.suptitle(f"{title_prefix} residual matrix | {electrode}", fontweight="semibold")
            for ax, (technique, sweep) in zip(axes.ravel(), PANEL_ORDER):
                model = lookup.get((electrode, technique, sweep))
                ax.set_title(f"{TECHNIQUES[technique]['label']} - {SWEEPS[sweep]}")
                if model is None:
                    ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                    ax.axis("off")
                    continue
                vmax = float(np.nanmax(np.abs(model["residual_matrix"])))
                if vmax <= 0:
                    vmax = 1.0
                image = ax.imshow(
                    model["residual_matrix"],
                    aspect="auto",
                    origin="lower",
                    cmap="coolwarm",
                    vmin=-vmax,
                    vmax=vmax,
                    extent=[
                        float(np.min(model["concentrations"])),
                        float(np.max(model["concentrations"])),
                        float(np.min(model["x_grid"])),
                        float(np.max(model["x_grid"])),
                    ],
                )
                ax.set_xlabel("DA concentration (uM)")
                ax.set_ylabel("Potential (V)")
                fig.colorbar(image, ax=ax, label=f"Residual ({CURRENT_UNIT})")
            fig.tight_layout(rect=(0, 0.02, 1, 0.94))
            pdf.savefig(fig)
            plt.close(fig)


def plot_mean_training_fits(
    conditions: list[dict[str, str]],
    mean_models: list[dict[str, object]],
    output_path: Path,
) -> None:
    lookup = model_lookup(mean_models)
    with PdfPages(output_path) as pdf:
        for condition in conditions:
            condition_id = int(float(condition["Condition"]))
            da_uM = float(condition["Dopamine (uM)"])
            fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2))
            fig.suptitle(
                f"PCR training fit | Condition {condition_id} | DA {clean_value(da_uM)} uM",
                fontweight="semibold",
            )
            for ax, (technique, sweep) in zip(axes.ravel(), PANEL_ORDER):
                model = lookup.get(("mean", technique, sweep))
                style_curve_axis(ax)
                ax.set_title(f"{TECHNIQUES[technique]['label']} - {SWEEPS[sweep]}")
                if model is None:
                    ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                    continue
                match = np.where(model["conditions"] == condition_id)[0]
                if len(match) == 0:
                    ax.text(0.5, 0.5, "No condition", ha="center", va="center", transform=ax.transAxes)
                    continue
                index = int(match[0])
                ax.plot(model["x_grid"], model["data_matrix"][:, index], color="#555555", lw=1.1, label="aligned signal")
                ax.plot(model["x_grid"], model["predicted_matrix"][:, index], color="#D55E00", lw=1.5, label="PCR predicted")
                ax.plot(model["x_grid"], model["residual_matrix"][:, index], color="#0072B2", lw=0.8, ls="--", label="residual")
                ax.legend(loc="best", fontsize=7)
            fig.tight_layout(rect=(0, 0.02, 1, 0.94))
            pdf.savefig(fig)
            plt.close(fig)


def plot_predicted_curves_pdf(
    predicted_curves: dict[str, dict[str, object]],
    da_uM: float,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(f"PCR predicted mean curves | DA {clean_value(da_uM)} uM", fontweight="semibold")
    for ax, (technique_key, sweep_key) in zip(axes.ravel(), PANEL_ORDER):
        curve_key = f"{technique_key}_{sweep_key}"
        curve = predicted_curves[curve_key]
        style_curve_axis(ax)
        ax.set_title(f"{curve['technique_label']} - {curve['sweep_label']}")
        ax.plot(curve["potential_v"], curve["current_uA"], color="#D55E00", lw=1.6)
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    fig.savefig(output_path)
    plt.close(fig)


def run_analysis(args: argparse.Namespace) -> None:
    configure_plot_style()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    params = {
        "n_components": args.n_components,
        "score_trend_degree": args.score_trend_degree,
        "smooth_window": args.smooth_window,
        "smooth_polyorder": args.smooth_polyorder,
        "grid_points": args.grid_points,
    }

    conditions = read_manifest(args.manifest.resolve())
    entries = load_trace_entries(conditions, args.subset_dir.resolve())
    models = fit_pcr_models(entries, params)
    electrode_models = [model for model in models if model["scope"] == "electrode"]
    mean_models = [model for model in models if model["scope"] == "mean"]

    electrode_scores_csv = output_dir / "pcr_scores_by_electrode.csv"
    mean_scores_csv = output_dir / "pcr_scores_mean_curves.csv"
    electrode_metrics_csv = output_dir / "pcr_metrics_by_electrode.csv"
    mean_metrics_csv = output_dir / "pcr_metrics_mean_curves.csv"
    write_score_csv(electrode_models, electrode_scores_csv)
    write_score_csv(mean_models, mean_scores_csv)
    write_metrics_csv(electrode_models, electrode_metrics_csv)
    write_metrics_csv(mean_models, mean_metrics_csv)

    mean_fit_pdf = output_dir / "mean_curve_pcr_training_fits.pdf"
    mean_scores_pdf = output_dir / "pcr_score_relationships_mean_curves.pdf"
    electrode_scores_pdf = output_dir / "pcr_score_relationships_by_electrode.pdf"
    mean_loadings_pdf = output_dir / "pcr_loadings_mean_curves.pdf"
    electrode_loadings_pdf = output_dir / "pcr_loadings_by_electrode.pdf"
    mean_residual_pdf = output_dir / "pcr_residuals_mean_curves.pdf"
    electrode_residual_pdf = output_dir / "pcr_residuals_by_electrode.pdf"
    model_path = output_dir / "pcr_mean_curve_model.json"

    plot_mean_training_fits(conditions, mean_models, mean_fit_pdf)
    plot_score_relationships(mean_models, mean_scores_pdf, "Mean PCR")
    plot_score_relationships(electrode_models, electrode_scores_pdf, "Electrode PCR")
    plot_loadings(mean_models, mean_loadings_pdf, "Mean PCR")
    plot_loadings(electrode_models, electrode_loadings_pdf, "Electrode PCR")
    plot_residual_heatmaps(mean_models, mean_residual_pdf, "Mean PCR")
    plot_residual_heatmaps(electrode_models, electrode_residual_pdf, "Electrode PCR")

    prediction_model = build_prediction_model(models, params, scope="mean")
    write_json(prediction_model, model_path)

    print(f"Wrote electrode PCR scores: {electrode_scores_csv}")
    print(f"Wrote mean PCR scores: {mean_scores_csv}")
    print(f"Wrote electrode PCR metrics: {electrode_metrics_csv}")
    print(f"Wrote mean PCR metrics: {mean_metrics_csv}")
    print(f"Wrote mean training fit plots: {mean_fit_pdf}")
    print(f"Wrote mean score relationship plots: {mean_scores_pdf}")
    print(f"Wrote electrode score relationship plots: {electrode_scores_pdf}")
    print(f"Wrote mean loading plots: {mean_loadings_pdf}")
    print(f"Wrote electrode loading plots: {electrode_loadings_pdf}")
    print(f"Wrote mean residual plots: {mean_residual_pdf}")
    print(f"Wrote electrode residual plots: {electrode_residual_pdf}")
    print(f"Wrote PCR mean-curve model: {model_path}")

    if args.predict_da is not None:
        predicted_curves = predict_curves(prediction_model, args.predict_da, points=args.prediction_points)
        da_label = slug_float(args.predict_da)
        prediction_csv = output_dir / f"predicted_curves_DA_{da_label}uM.csv"
        prediction_pdf = output_dir / f"predicted_curves_DA_{da_label}uM.pdf"
        write_prediction_csv(predicted_curves, args.predict_da, prediction_csv)
        plot_predicted_curves_pdf(predicted_curves, args.predict_da, prediction_pdf)
        print(f"Wrote predicted curve CSV: {prediction_csv}")
        print(f"Wrote predicted curve PDF: {prediction_pdf}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit PCR models to Day 1 DA-only CV/CV GC data.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--subset-dir", type=Path, default=DEFAULT_SUBSET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--n-components",
        type=int,
        default=DEFAULT_N_COMPONENTS,
        help=f"PCA components retained per curve group. Default: {DEFAULT_N_COMPONENTS}",
    )
    parser.add_argument(
        "--score-trend-degree",
        type=int,
        default=DEFAULT_SCORE_TREND_DEGREE,
        help=(
            "Polynomial degree used to fit each PC score versus DA concentration. "
            f"Default: {DEFAULT_SCORE_TREND_DEGREE}"
        ),
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=DEFAULT_SMOOTH_WINDOW,
        help=f"Odd Savitzky-Golay smoothing window before PCR. Use 0 or 1 to disable. Default: {DEFAULT_SMOOTH_WINDOW}",
    )
    parser.add_argument(
        "--smooth-polyorder",
        type=int,
        default=DEFAULT_SMOOTH_POLYORDER,
        help=f"Savitzky-Golay smoothing polynomial order. Default: {DEFAULT_SMOOTH_POLYORDER}",
    )
    parser.add_argument(
        "--grid-points",
        type=int,
        default=DEFAULT_GRID_POINTS,
        help="Voltage grid points after alignment. Use 0 to keep the native common grid when possible. Default: 0",
    )
    parser.add_argument("--predict-da", type=float, default=None)
    parser.add_argument(
        "--prediction-points",
        type=int,
        default=500,
        help="Voltage points per predicted curve. Use 0 for the aligned PCR grid. Default: 500",
    )
    return parser.parse_args()


def main() -> None:
    run_analysis(parse_args())


if __name__ == "__main__":
    main()
