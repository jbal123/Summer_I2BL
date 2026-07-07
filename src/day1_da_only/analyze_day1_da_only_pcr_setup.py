#!/usr/bin/env python3
"""Analyze and interact with one PCR sweep setup."""

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
import math
import os
import tempfile
from pathlib import Path

import numpy as np

CACHE_ROOT = Path(tempfile.gettempdir()) / "microneedlearrayml_matplotlib"
(CACHE_ROOT / "mplconfig").mkdir(parents=True, exist_ok=True)
(CACHE_ROOT / "xdg_cache").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT / "xdg_cache"))

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.widgets import Button, TextBox

from day1_da_only_pcr_core import (
    CURRENT_UNIT,
    DEFAULT_MANIFEST,
    DEFAULT_N_COMPONENTS,
    DEFAULT_SCORE_TREND_DEGREE,
    DEFAULT_SUBSET_DIR,
    PANEL_ORDER,
    SWEEPS,
    TECHNIQUES,
    build_prediction_model,
    clean_value,
    fit_pcr_models,
    json_default,
    load_trace_entries,
    predict_curves,
    read_csv_dicts,
    read_manifest,
    slug_float,
    write_prediction_csv,
)


DEFAULT_SWEEP_DIR = DEFAULT_SUBSET_DIR / "pcr_sweep"
DEFAULT_OUTPUT_ROOT = DEFAULT_SUBSET_DIR / "selected_pcr_setup_analysis"


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


def safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def selected_scope_values(scope_summary: str) -> set[str]:
    if scope_summary in {"all", "both"}:
        return {"mean", "electrode"}
    return {scope_summary}


def select_setup_row(
    best_setups_path: Path,
    summary_path: Path,
    setup_id: int,
    scope_summary: str | None,
) -> dict[str, str]:
    def matching_rows(path: Path) -> list[dict[str, str]]:
        if not path.exists():
            return []
        rows = [
            row
            for row in read_csv_dicts(path)
            if row.get("setup_id") and int(float(row["setup_id"])) == setup_id
        ]
        if scope_summary is not None:
            rows = [row for row in rows if row["scope_summary"] == scope_summary]
        return rows

    rows = matching_rows(best_setups_path)
    if not rows:
        rows = matching_rows(summary_path)
    if not rows:
        raise ValueError(
            f"No setup_id={setup_id} row found in {best_setups_path} or {summary_path}"
            + (f" with scope_summary={scope_summary}" if scope_summary is not None else "")
        )
    if rows[0].get("method") != "pcr":
        raise ValueError(
            "The selected setup row is not a PCR setup row. Run "
            "sweep_day1_da_only_pcr.py to create Day1_DA_Only_Conditions/pcr_sweep."
        )
    return rows[0]


def setup_params_from_row(row: dict[str, str]) -> dict[str, object]:
    return {
        "setup_id": int(float(row["setup_id"])),
        "n_components": int(float(row.get("requested_n_components") or row.get("n_components") or DEFAULT_N_COMPONENTS)),
        "score_trend_degree": int(float(row.get("score_trend_degree") or DEFAULT_SCORE_TREND_DEGREE)),
        "score_trend_degree_effective": int(float(row.get("score_trend_degree_effective") or row.get("score_trend_degree") or DEFAULT_SCORE_TREND_DEGREE)),
        "smooth_window": int(float(row["smooth_window"])),
        "smooth_polyorder": int(float(row["smooth_polyorder"])),
        "grid_points": int(float(row["grid_points"])),
        "scope_summary": row["scope_summary"],
        "objective_score": safe_float(row.get("objective_score")),
        "avg_r2": safe_float(row.get("avg_r2")),
        "avg_normalized_rmse": safe_float(row.get("avg_normalized_rmse")),
        "avg_abs_pc_pearson": safe_float(row.get("avg_abs_pc_pearson")),
        "avg_pc_trend_r2": safe_float(row.get("avg_pc_trend_r2")),
    }


def setup_output_dir(output_root: Path, params: dict[str, object]) -> Path:
    name = (
        f"setup_{params['setup_id']}"
        f"_pc{params['n_components']}"
        f"_trend{params['score_trend_degree']}"
        f"_sw{params['smooth_window']}"
        f"_sp{params['smooth_polyorder']}"
        f"_grid{params['grid_points']}"
        f"_scope{params['scope_summary']}"
    )
    return output_root / name


def filtered_detail_rows(
    detailed_csv_path: Path,
    params: dict[str, object],
) -> list[dict[str, str]]:
    wanted_scopes = selected_scope_values(str(params["scope_summary"]))
    rows = []
    for row in read_csv_dicts(detailed_csv_path):
        if int(float(row["setup_id"])) != int(params["setup_id"]):
            continue
        if row["scope"] not in wanted_scopes:
            continue
        rows.append(row)
    return rows


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


def plot_setup_metric_charts(
    detailed_rows: list[dict[str, str]],
    params: dict[str, object],
    output_path: Path,
) -> None:
    if not detailed_rows:
        raise ValueError("No PCR detail rows matched the selected setup")

    grouped = {}
    for row in detailed_rows:
        grouped.setdefault(row["scope"], []).append(row)

    with PdfPages(output_path) as pdf:
        for scope, rows in sorted(grouped.items()):
            rows = sorted(rows, key=lambda row: (row["technique"], row["sweep"], row["electrode"]))
            labels = [f"{row['technique']}\n{row['sweep']}\n{row['electrode']}" for row in rows]
            x = np.arange(len(rows))
            r2 = np.array([safe_float(row["r2"]) for row in rows], dtype=float)
            nrmse = np.array([safe_float(row["normalized_rmse"]) for row in rows], dtype=float)
            pearson = np.array([safe_float(row["avg_abs_pc_pearson"]) for row in rows], dtype=float)
            trend_r2 = np.array([safe_float(row["avg_pc_trend_r2"]) for row in rows], dtype=float)

            fig, axes = plt.subplots(4, 1, figsize=(max(11, len(rows) * 0.55), 10.5), sharex=True)
            fig.suptitle(f"PCR setup {params['setup_id']} diagnostics | {scope}", fontweight="semibold")
            axes[0].bar(x, r2, color="#0072B2")
            axes[0].set_ylabel("Curve R2")
            axes[0].axhline(0, color="#777777", lw=0.8)
            axes[1].bar(x, nrmse, color="#D55E00")
            axes[1].set_ylabel("Norm RMSE")
            axes[2].bar(x, pearson, color="#009E73")
            axes[2].set_ylabel("|PC pearson|")
            axes[3].bar(x, trend_r2, color="#CC79A7")
            axes[3].set_ylabel("PC trend R2")
            axes[3].set_xticks(x)
            axes[3].set_xticklabels(labels, rotation=90, fontsize=7)
            for ax in axes:
                ax.tick_params(axis="both", which="major", direction="in")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
            fig.tight_layout(rect=(0, 0.02, 1, 0.94))
            pdf.savefig(fig)
            plt.close(fig)


def plot_score_correlation_charts(
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


def plot_loadings_and_residuals(
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

            fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2))
            fig.suptitle(f"{title_prefix} residual matrices | {electrode}", fontweight="semibold")
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


def plot_predicted_vs_actual_training(
    mean_models: list[dict[str, object]],
    output_path: Path,
) -> None:
    lookup = model_lookup(mean_models)
    conditions = sorted({int(condition) for model in mean_models for condition in model["conditions"]})

    with PdfPages(output_path) as pdf:
        for condition in conditions:
            fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2))
            da_label = ""
            for model in mean_models:
                matches = np.where(model["conditions"] == condition)[0]
                if len(matches):
                    da_label = clean_value(float(model["concentrations"][int(matches[0])]))
                    break
            fig.suptitle(
                f"PCR predicted vs actual training curves | Condition {condition} | DA {da_label} uM",
                fontweight="semibold",
            )
            for ax, (technique, sweep) in zip(axes.ravel(), PANEL_ORDER):
                model = lookup.get(("mean", technique, sweep))
                style_curve_axis(ax)
                ax.set_title(f"{TECHNIQUES[technique]['label']} - {SWEEPS[sweep]}")
                if model is None:
                    ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                    continue
                matches = np.where(model["conditions"] == condition)[0]
                if len(matches) == 0:
                    ax.text(0.5, 0.5, "No condition", ha="center", va="center", transform=ax.transAxes)
                    continue
                index = int(matches[0])
                ax.plot(model["x_grid"], model["data_matrix"][:, index], color="#555555", lw=1.1, label="aligned signal")
                ax.plot(model["x_grid"], model["predicted_matrix"][:, index], color="#D55E00", lw=1.5, label="PCR predicted")
                ax.plot(model["x_grid"], model["residual_matrix"][:, index], color="#0072B2", lw=0.8, ls="--", label="residual")
                ax.legend(loc="best", fontsize=7)
            fig.tight_layout(rect=(0, 0.02, 1, 0.94))
            pdf.savefig(fig)
            plt.close(fig)


def plot_actual_vs_predicted_overlay(
    models: list[dict[str, object]],
    output_path: Path,
    title_prefix: str,
    tight_y: bool = False,
) -> None:
    grouped = {}
    for model in models:
        grouped.setdefault(str(model["electrode"]), []).append(model)

    with PdfPages(output_path) as pdf:
        for electrode, group_models in sorted(grouped.items()):
            lookup = model_lookup(group_models)
            fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.2))
            fig.suptitle(f"{title_prefix} actual vs predicted overlay | {electrode}", fontweight="semibold")
            for ax, (technique, sweep) in zip(axes.ravel(), PANEL_ORDER):
                model = lookup.get((electrode, technique, sweep))
                style_curve_axis(ax)
                ax.set_title(f"{TECHNIQUES[technique]['label']} - {SWEEPS[sweep]}")
                if model is None:
                    ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                    continue

                concentrations = np.asarray(model["concentrations"], dtype=float)
                color_values = plt.cm.viridis(np.linspace(0.08, 0.92, len(concentrations)))
                y_values_for_limits = []
                for index, (da_uM, color) in enumerate(zip(concentrations, color_values)):
                    label_value = clean_value(float(da_uM))
                    actual_y = model["data_matrix"][:, index]
                    predicted_y = model["predicted_matrix"][:, index]
                    y_values_for_limits.extend([actual_y, predicted_y])
                    ax.plot(model["x_grid"], actual_y, color=color, lw=1.2, alpha=0.9, label=f"actual {label_value} uM")
                    ax.plot(model["x_grid"], predicted_y, color=color, lw=1.3, ls="--", alpha=0.95, label=f"pred {label_value} uM")

                if tight_y and y_values_for_limits:
                    combined_y = np.concatenate([np.asarray(values, dtype=float) for values in y_values_for_limits])
                    finite_y = combined_y[np.isfinite(combined_y)]
                    if len(finite_y):
                        y_min = float(np.min(finite_y))
                        y_max = float(np.max(finite_y))
                        y_span = y_max - y_min
                        if y_span <= 0:
                            y_span = max(abs(y_max), 1.0)
                        padding = 0.06 * y_span
                        ax.set_ylim(y_min - padding, y_max + padding)

                ax.text(
                    0.02,
                    0.97,
                    (
                        f"R2={float(model['r2']):.3f}\n"
                        f"RMSE={float(model['rmse_uA']):.3g} {CURRENT_UNIT}\n"
                        f"PC trend={float(model['avg_pc_trend_r2']):.3f}"
                    ),
                    ha="left",
                    va="top",
                    transform=ax.transAxes,
                    fontsize=8,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75},
                )
                ax.legend(loc="best", fontsize=6, ncol=2)
            fig.tight_layout(rect=(0, 0.02, 1, 0.94))
            pdf.savefig(fig)
            plt.close(fig)


def plot_prediction_pdf(
    predicted: dict[str, dict[str, object]],
    concentration: float,
    output_path: Path,
    training_models: list[dict[str, object]] | None = None,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2))
    fig.suptitle(f"Interactive PCR predicted curves | DA {clean_value(concentration)} uM", fontweight="semibold")
    models_by_curve = {}
    if training_models is not None:
        models_by_curve = {(model["technique"], model["sweep"]): model for model in training_models}

    for ax, (technique, sweep) in zip(axes.ravel(), PANEL_ORDER):
        curve_key = f"{technique}_{sweep}"
        curve = predicted[curve_key]
        style_curve_axis(ax)
        ax.set_title(f"{curve['technique_label']} - {curve['sweep_label']}")
        model = models_by_curve.get((technique, sweep))
        if model is not None:
            for index in range(model["data_matrix"].shape[1]):
                ax.plot(model["x_grid"], model["data_matrix"][:, index], color="#CCCCCC", lw=0.5, alpha=0.45)
        ax.plot(curve["potential_v"], curve["current_uA"], color="#D55E00", lw=1.8, label="predicted")
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    fig.savefig(output_path)
    plt.close(fig)


def save_prediction_outputs(
    model: dict[str, object],
    mean_models: list[dict[str, object]],
    concentration: float,
    setup_dir: Path,
    prediction_points: int,
) -> tuple[Path, Path]:
    prediction_dir = setup_dir / "interactive_predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    da_label = slug_float(concentration)
    csv_path = prediction_dir / f"predicted_curves_DA_{da_label}uM.csv"
    pdf_path = prediction_dir / f"predicted_curves_DA_{da_label}uM.pdf"
    predicted = predict_curves(model, concentration, points=prediction_points)
    write_prediction_csv(predicted, concentration, csv_path)
    plot_prediction_pdf(predicted, concentration, pdf_path, mean_models)
    return csv_path, pdf_path


def launch_interactive_viewer(
    model: dict[str, object],
    mean_models: list[dict[str, object]],
    setup_dir: Path,
    initial_concentration: float,
    prediction_points: int,
    no_show: bool,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.7))
    plt.subplots_adjust(left=0.08, right=0.985, top=0.87, bottom=0.18, hspace=0.38, wspace=0.22)
    text_ax = fig.add_axes([0.18, 0.055, 0.22, 0.045])
    save_ax = fig.add_axes([0.43, 0.055, 0.16, 0.045])
    status_ax = fig.add_axes([0.61, 0.04, 0.35, 0.075])
    status_ax.axis("off")

    textbox = TextBox(text_ax, "DA uM", initial=str(clean_value(initial_concentration)))
    save_button = Button(save_ax, "Generate + Save", color="#EEEEEE", hovercolor="#CCDDFF")
    status_text = status_ax.text(0, 0.7, "", fontsize=8, va="center")
    models_by_curve = {(model["technique"], model["sweep"]): model for model in mean_models}

    def draw_prediction(concentration: float, save: bool) -> None:
        predicted = predict_curves(model, concentration, points=prediction_points)
        fig.suptitle(f"Interactive PCR predicted curves | DA {clean_value(concentration)} uM", fontweight="semibold")
        for ax, (technique, sweep) in zip(axes.ravel(), PANEL_ORDER):
            ax.cla()
            curve_key = f"{technique}_{sweep}"
            curve = predicted[curve_key]
            style_curve_axis(ax)
            ax.set_title(f"{curve['technique_label']} - {curve['sweep_label']}")
            training_model = models_by_curve.get((technique, sweep))
            if training_model is not None:
                for index in range(training_model["data_matrix"].shape[1]):
                    ax.plot(training_model["x_grid"], training_model["data_matrix"][:, index], color="#CCCCCC", lw=0.5, alpha=0.45)
            ax.plot(curve["potential_v"], curve["current_uA"], color="#D55E00", lw=1.8, label="predicted")
            ax.legend(loc="best", fontsize=8)

        if save:
            csv_path, pdf_path = save_prediction_outputs(
                model,
                mean_models,
                concentration,
                setup_dir,
                prediction_points,
            )
            status_text.set_text(f"Saved:\n{csv_path.name}\n{pdf_path.name}")
            print(f"Saved prediction CSV: {csv_path}")
            print(f"Saved prediction PDF: {pdf_path}")
        else:
            status_text.set_text("Enter DA concentration, then press Generate + Save.")
        fig.canvas.draw_idle()

    def parse_and_draw(save: bool) -> None:
        try:
            concentration = float(textbox.text)
        except ValueError:
            status_text.set_text(f"Invalid concentration: {textbox.text}")
            fig.canvas.draw_idle()
            return
        draw_prediction(concentration, save=save)

    textbox.on_submit(lambda _text: parse_and_draw(save=True))
    save_button.on_clicked(lambda _event: parse_and_draw(save=True))
    draw_prediction(initial_concentration, save=False)

    if no_show:
        plt.close(fig)
        return
    plt.show()


def write_json(path: Path, data: dict[str, object]) -> None:
    with path.open("w") as handle:
        json.dump(data, handle, indent=2, default=json_default)
        handle.write("\n")


def run(args: argparse.Namespace) -> None:
    configure_plot_style()
    sweep_dir = (args.sweep_dir or DEFAULT_SWEEP_DIR).resolve()
    best_setups_path = args.best_setups.resolve() if args.best_setups else sweep_dir / "best_setups.csv"
    summary_path = args.summary_csv.resolve() if args.summary_csv else sweep_dir / "pcr_summary.csv"
    detailed_csv_path = args.detailed_csv.resolve() if args.detailed_csv else sweep_dir / "pcr_detailed.csv"

    selected_row = select_setup_row(
        best_setups_path,
        summary_path,
        args.setup_id,
        args.scope_summary,
    )
    params = setup_params_from_row(selected_row)
    setup_dir = setup_output_dir((args.output_root or DEFAULT_OUTPUT_ROOT).resolve(), params)
    setup_dir.mkdir(parents=True, exist_ok=True)

    print(f"Selected setup row: {selected_row}")
    print(f"Output directory: {setup_dir}")

    detailed_rows = filtered_detail_rows(detailed_csv_path, params)
    conditions = read_manifest(args.manifest.resolve())
    entries = load_trace_entries(conditions, args.subset_dir.resolve())
    all_models = fit_pcr_models(entries, params)
    mean_models = [model for model in all_models if model["scope"] == "mean"]
    diagnostic_models = [
        model for model in all_models if model["scope"] in selected_scope_values(str(params["scope_summary"]))
    ]
    prediction_model = build_prediction_model(all_models, params, scope="mean")

    write_json(setup_dir / "selected_setup_metadata.json", {"selected_row": selected_row, "params": params})
    write_json(setup_dir / "mean_pcr_model.json", prediction_model)

    metrics_pdf = setup_dir / "pcr_setup_metric_charts.pdf"
    score_pdf = setup_dir / "pcr_score_correlation_charts.pdf"
    diagnostics_pdf = setup_dir / "pcr_loadings_and_residual_diagnostics.pdf"
    training_pdf = setup_dir / "predicted_vs_actual_training_curves.pdf"
    mean_overlay_pdf = setup_dir / "actual_vs_predicted_overlay_mean_curves.pdf"
    selected_scope_overlay_pdf = setup_dir / "actual_vs_predicted_overlay_selected_scope.pdf"

    plot_setup_metric_charts(detailed_rows, params, metrics_pdf)
    plot_score_correlation_charts(diagnostic_models, score_pdf, "PCR selected-scope")
    plot_loadings_and_residuals(diagnostic_models, diagnostics_pdf, "PCR selected-scope")
    plot_predicted_vs_actual_training(mean_models, training_pdf)
    plot_actual_vs_predicted_overlay(mean_models, mean_overlay_pdf, "PCR mean")
    plot_actual_vs_predicted_overlay(diagnostic_models, selected_scope_overlay_pdf, "PCR selected-scope", tight_y=True)

    print(f"Wrote PCR setup metric charts: {metrics_pdf}")
    print(f"Wrote PCR score correlation charts: {score_pdf}")
    print(f"Wrote PCR loading/residual diagnostics: {diagnostics_pdf}")
    print(f"Wrote predicted-vs-actual training PDF: {training_pdf}")
    print(f"Wrote mean actual-vs-predicted overlay PDF: {mean_overlay_pdf}")
    print(f"Wrote selected-scope actual-vs-predicted overlay PDF: {selected_scope_overlay_pdf}")

    if args.concentration is not None:
        csv_path, pdf_path = save_prediction_outputs(
            prediction_model,
            mean_models,
            args.concentration,
            setup_dir,
            args.prediction_points,
        )
        print(f"Wrote initial prediction CSV: {csv_path}")
        print(f"Wrote initial prediction PDF: {pdf_path}")

    initial_concentration = args.concentration
    if initial_concentration is None:
        training_da = sorted({float(value) for model in mean_models for value in model["concentrations"]})
        initial_concentration = float(np.median(training_da))

    launch_interactive_viewer(
        model=prediction_model,
        mean_models=mean_models,
        setup_dir=setup_dir,
        initial_concentration=initial_concentration,
        prediction_points=args.prediction_points,
        no_show=args.no_show,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze one setup_id from PCR best_setups.csv.")
    parser.add_argument("--setup-id", type=int, required=True, help="setup_id from PCR best_setups.csv")
    parser.add_argument(
        "--scope-summary",
        choices=["mean", "electrode", "all"],
        default=None,
        help="Optional disambiguator when setup_id appears multiple times",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--subset-dir", type=Path, default=DEFAULT_SUBSET_DIR)
    parser.add_argument("--sweep-dir", type=Path, default=None)
    parser.add_argument("--best-setups", type=Path, default=None)
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--detailed-csv", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--concentration", type=float, default=None)
    parser.add_argument(
        "--prediction-points",
        type=int,
        default=500,
        help="Voltage points per predicted curve. Use 0 for the aligned PCR grid. Default: 500",
    )
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
