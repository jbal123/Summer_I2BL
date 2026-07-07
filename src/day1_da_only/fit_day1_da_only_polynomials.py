#!/usr/bin/env python3
"""Fit Classical Least Squares (CLS) models to Day 1 DA-only CV curves.

The measured CV current vector is treated as the signal matrix D and dopamine
concentration as the single-component concentration matrix C. For each
technique/sweep/electrode group, CLS solves one voltage-dependent sensitivity
vector S:

    D = S.T @ C
    S = (C @ C.T)^-1 @ C @ D.T

Predicted curves are then generated directly as d_pred = S.T * c_new.
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

from day1_da_only_cls_core import (
    CURRENT_UNIT,
    DEFAULT_GRID_POINTS,
    DEFAULT_MANIFEST,
    DEFAULT_MODEL_ORDER,
    DEFAULT_SMOOTH_POLYORDER,
    DEFAULT_SMOOTH_WINDOW,
    DEFAULT_SUBSET_DIR,
    PANEL_ORDER,
    SENSITIVITY_UNIT,
    SWEEPS,
    TECHNIQUES,
    build_prediction_model,
    clean_value,
    fit_cls_models,
    format_float,
    format_series,
    json_default,
    load_trace_entries,
    normalize_model_order,
    predict_curves,
    read_manifest,
    slug_float,
    write_prediction_csv,
)


DEFAULT_OUTPUT_DIR = DEFAULT_SUBSET_DIR / "cls_analysis"
QUADRATIC_OUTPUT_DIR = DEFAULT_SUBSET_DIR / "cls_quadratic_analysis"
CUBIC_OUTPUT_DIR = DEFAULT_SUBSET_DIR / "cls_cubic_analysis"


def default_output_dir(model_order: str) -> Path:
    order = normalize_model_order(model_order)
    if order == "cubic":
        return CUBIC_OUTPUT_DIR
    if order == "quadratic":
        return QUADRATIC_OUTPUT_DIR
    return DEFAULT_OUTPUT_DIR


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


def style_sensitivity_axis(ax: plt.Axes) -> None:
    ax.set_xlabel("Potential (V)")
    ax.set_ylabel("CLS calibration vector")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", direction="in")
    ax.tick_params(axis="both", which="minor", direction="in")
    ax.minorticks_on()
    formatter = mticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-3, 3))
    ax.yaxis.set_major_formatter(formatter)


def model_lookup(models: list[dict[str, object]]) -> dict[tuple[str, str, str], dict[str, object]]:
    return {
        (str(model["electrode"]), str(model["technique"]), str(model["sweep"])): model
        for model in models
    }


def write_json(data: dict[str, object], output_path: Path) -> None:
    with output_path.open("w") as handle:
        json.dump(data, handle, indent=2, default=json_default)
        handle.write("\n")


def write_sensitivity_csv(models: list[dict[str, object]], output_path: Path) -> None:
    fieldnames = [
        "scope",
        "technique",
        "technique_label",
        "sweep",
        "sweep_label",
        "electrode",
        "model_order",
        "method",
        "potential_v",
        "sensitivity_uA_per_uM",
        "quadratic_sensitivity_uA_per_uM2",
        "cubic_sensitivity_uA_per_uM3",
        "per_voltage_rmse_uA",
        "per_voltage_mae_uA",
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
            for index, potential_v in enumerate(model["x_grid"]):
                writer.writerow(
                    {
                        "scope": model["scope"],
                        "technique": model["technique"],
                        "technique_label": model["technique_label"],
                        "sweep": model["sweep"],
                        "sweep_label": model["sweep_label"],
                        "electrode": model["electrode"],
                        "model_order": model["model_order"],
                        "method": "cls",
                        "potential_v": format_float(float(potential_v)),
                        "sensitivity_uA_per_uM": format_float(float(model["sensitivity_uA_per_uM"][index])),
                        "quadratic_sensitivity_uA_per_uM2": (
                            format_float(float(model["quadratic_sensitivity_uA_per_uM2"][index]))
                            if model["quadratic_sensitivity_uA_per_uM2"] is not None
                            else ""
                        ),
                        "cubic_sensitivity_uA_per_uM3": (
                            format_float(float(model["cubic_sensitivity_uA_per_uM3"][index]))
                            if model["cubic_sensitivity_uA_per_uM3"] is not None
                            else ""
                        ),
                        "per_voltage_rmse_uA": format_float(float(model["per_voltage_rmse_uA"][index])),
                        "per_voltage_mae_uA": format_float(float(model["per_voltage_mae_uA"][index])),
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
        "model_order",
        "method",
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
                    "model_order": model["model_order"],
                    "method": "cls",
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
                f"{model_order_title(mean_models)} CLS training fit | Condition {condition_id} | DA {clean_value(da_uM)} uM",
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
                ax.plot(model["x_grid"], model["predicted_matrix"][:, index], color="#D55E00", lw=1.5, label="CLS predicted")
                ax.plot(model["x_grid"], model["residual_matrix"][:, index], color="#0072B2", lw=0.8, ls="--", label="residual")
                ax.legend(loc="best", fontsize=7)
            fig.tight_layout(rect=(0, 0.02, 1, 0.94))
            pdf.savefig(fig)
            plt.close(fig)


def plot_sensitivity_vectors(
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
            fig.suptitle(f"{title_prefix} | {electrode}", fontweight="semibold")
            for ax, (technique, sweep) in zip(axes.ravel(), PANEL_ORDER):
                model = lookup.get((electrode, technique, sweep))
                style_sensitivity_axis(ax)
                ax.set_title(f"{TECHNIQUES[technique]['label']} - {SWEEPS[sweep]}")
                if model is None:
                    ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                    continue
                term_styles = [
                    ("sensitivity_uA_per_uM", "#D55E00", "-", f"linear term ({SENSITIVITY_UNIT})"),
                    ("quadratic_sensitivity_uA_per_uM2", "#0072B2", "--", "quadratic term (uA/uM^2)"),
                    ("cubic_sensitivity_uA_per_uM3", "#009E73", ":", "cubic term (uA/uM^3)"),
                ]
                for key, color, linestyle, label in term_styles:
                    if model.get(key) is None:
                        continue
                    ax.plot(
                        model["x_grid"],
                        model[key],
                        color=color,
                        lw=1.6 if key == "sensitivity_uA_per_uM" else 1.2,
                        ls=linestyle,
                        label=label,
                    )
                ax.text(
                    0.02,
                    0.96,
                    f"R2={float(model['r2']):.3f}\nRMSE={float(model['rmse_uA']):.3g} {CURRENT_UNIT}",
                    ha="left",
                    va="top",
                    transform=ax.transAxes,
                    fontsize=8,
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


def plot_predicted_curves_pdf(
    predicted_curves: dict[str, dict[str, object]],
    da_uM: float,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    order = next(iter(predicted_curves.values())).get("model_order", "linear")
    fig.suptitle(f"{str(order).title()} CLS predicted mean curves | DA {clean_value(da_uM)} uM", fontweight="semibold")
    for ax, (technique_key, sweep_key) in zip(axes.ravel(), PANEL_ORDER):
        curve_key = f"{technique_key}_{sweep_key}"
        curve = predicted_curves[curve_key]
        style_curve_axis(ax)
        ax.set_title(f"{curve['technique_label']} - {curve['sweep_label']}")
        ax.plot(curve["potential_v"], curve["current_uA"], color="#D55E00", lw=1.6)
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    fig.savefig(output_path)
    plt.close(fig)


def model_order_title(models: list[dict[str, object]]) -> str:
    if not models:
        return "Linear"
    return str(models[0].get("model_order", "linear")).title()


def run_analysis(args: argparse.Namespace) -> None:
    configure_plot_style()
    model_order = normalize_model_order(args.model_order)
    output_dir = (args.output_dir or default_output_dir(model_order)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    params = {
        "model_order": model_order,
        "smooth_window": args.smooth_window,
        "smooth_polyorder": args.smooth_polyorder,
        "grid_points": args.grid_points,
    }

    conditions = read_manifest(args.manifest.resolve())
    entries = load_trace_entries(conditions, args.subset_dir.resolve())
    models = fit_cls_models(entries, params)
    electrode_models = [model for model in models if model["scope"] == "electrode"]
    mean_models = [model for model in models if model["scope"] == "mean"]

    electrode_sensitivity_csv = output_dir / "cls_sensitivity_by_electrode.csv"
    mean_sensitivity_csv = output_dir / "cls_sensitivity_mean_curves.csv"
    electrode_metrics_csv = output_dir / "cls_metrics_by_electrode.csv"
    mean_metrics_csv = output_dir / "cls_metrics_mean_curves.csv"
    write_sensitivity_csv(electrode_models, electrode_sensitivity_csv)
    write_sensitivity_csv(mean_models, mean_sensitivity_csv)
    write_metrics_csv(electrode_models, electrode_metrics_csv)
    write_metrics_csv(mean_models, mean_metrics_csv)

    mean_fit_pdf = output_dir / "mean_curve_cls_training_fits.pdf"
    mean_sensitivity_pdf = output_dir / "sensitivity_vectors_mean_curves.pdf"
    electrode_sensitivity_pdf = output_dir / "sensitivity_vectors_by_electrode.pdf"
    mean_residual_pdf = output_dir / "cls_residuals_mean_curves.pdf"
    electrode_residual_pdf = output_dir / "cls_residuals_by_electrode.pdf"
    model_path = output_dir / "cls_mean_curve_model.json"

    plot_mean_training_fits(conditions, mean_models, mean_fit_pdf)
    title_order = model_order.title()
    plot_sensitivity_vectors(mean_models, mean_sensitivity_pdf, f"Mean {title_order} CLS sensitivity vectors")
    plot_sensitivity_vectors(electrode_models, electrode_sensitivity_pdf, f"Electrode {title_order} CLS sensitivity vectors")
    plot_residual_heatmaps(mean_models, mean_residual_pdf, f"Mean {title_order} CLS")
    plot_residual_heatmaps(electrode_models, electrode_residual_pdf, f"Electrode {title_order} CLS")

    prediction_model = build_prediction_model(models, params, scope="mean")
    write_json(prediction_model, model_path)

    print(f"Wrote electrode sensitivities: {electrode_sensitivity_csv}")
    print(f"Wrote mean sensitivities: {mean_sensitivity_csv}")
    print(f"Wrote electrode metrics: {electrode_metrics_csv}")
    print(f"Wrote mean metrics: {mean_metrics_csv}")
    print(f"Wrote mean training fit plots: {mean_fit_pdf}")
    print(f"Wrote mean sensitivity plots: {mean_sensitivity_pdf}")
    print(f"Wrote electrode sensitivity plots: {electrode_sensitivity_pdf}")
    print(f"Wrote mean residual plots: {mean_residual_pdf}")
    print(f"Wrote electrode residual plots: {electrode_residual_pdf}")
    print(f"Wrote CLS mean-curve model: {model_path}")

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
    parser = argparse.ArgumentParser(description="Fit CLS models to Day 1 DA-only CV/CV GC data.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--subset-dir", type=Path, default=DEFAULT_SUBSET_DIR)
    parser.add_argument(
        "--model-order",
        choices=["linear", "quadratic", "cubic"],
        default=DEFAULT_MODEL_ORDER,
        help="CLS calibration order. Linear uses [C], quadratic [C, C^2], cubic [C, C^2, C^3]. Default: linear.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            f"Output directory. Defaults to {DEFAULT_OUTPUT_DIR} for linear and "
            f"{QUADRATIC_OUTPUT_DIR} for quadratic and {CUBIC_OUTPUT_DIR} for cubic."
        ),
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=DEFAULT_SMOOTH_WINDOW,
        help=f"Odd Savitzky-Golay smoothing window before CLS. Use 0 or 1 to disable. Default: {DEFAULT_SMOOTH_WINDOW}",
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
        help="Voltage points per predicted curve. Use 0 for the aligned CLS grid. Default: 500",
    )
    return parser.parse_args()


def main() -> None:
    run_analysis(parse_args())


if __name__ == "__main__":
    main()
