#!/usr/bin/env python3
"""Test linear superposition of isolated-analyte PCR curves on mixtures.

For every multi-analyte condition in Days 1-4, this script predicts isolated
DA, AA, and UA final single-analyte CV/CV-GC responses from the all-analyte PCR
predictor, sums those predicted curves, and compares the summed prediction with
the actual mixture CV/CV-GC mean curves.
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

try:
    from scipy.optimize import curve_fit
except Exception:  # pragma: no cover
    curve_fit = None

from all_analyte_isolated_pcr_predictor import (
    ANALYTE_ORDER,
    ANALYTES,
    DAY_CONFIGS,
    DEFAULT_OUTPUT_ROOT as DEFAULT_COMPONENT_ROOT,
    build_prediction_model,
    clean_value,
    condition_code,
    configure_plot_style,
    load_trace_entries_for_series,
    params_from_setup_row,
    predict_curves,
    read_design_table,
    scan_isolated_conditions,
    select_best_setup,
)
from day1_da_only_cls_core import (
    CURRENT_UNIT,
    PANEL_ORDER,
    RESULTS_ROOT,
    ROOT,
    SWEEPS,
    TECHNIQUES,
    extract_electrode_traces,
    find_technique_file,
    load_numeric_rows,
    mean_trace_from_electrodes,
    safe_r2,
    split_cv_sweeps,
)
from day1_da_only_pcr_core import fit_pcr_models, json_default


DEFAULT_OUTPUT_ROOT = RESULTS_ROOT / "All_Analyte_Superposition_Analysis"

METRIC_FIELDNAMES = [
    "day",
    "condition",
    "condition_code",
    "dopamine_uM",
    "ascorbic_acid_uM",
    "uric_acid_uM",
    "technique",
    "sweep",
    "n_points",
    "rmse_uA",
    "mae_uA",
    "max_abs_residual_uA",
    "normalized_rmse",
    "r2",
    "superposition_mode",
    "baseline_source",
    "linear_correction",
    "gaussian_correction",
    "linear_edge_fraction",
    "linear_left_range_v",
    "linear_right_range_v",
    "gaussian_fit_start_v",
    "gaussian_center_v",
    "gaussian_sigma_v",
    "gaussian_fit_mode",
]


def default_output_root(args: argparse.Namespace) -> Path:
    if args.output_root is not None:
        return args.output_root
    suffixes = []
    if args.linear_correction:
        suffixes.append("linear")
    if args.gaussian_correction:
        suffixes.append("gaussian")
    if not suffixes:
        return DEFAULT_OUTPUT_ROOT
    return RESULTS_ROOT / f"All_Analyte_Superposition_Analysis_{'_'.join(suffixes)}"


def correction_config(args: argparse.Namespace) -> dict[str, object]:
    linear_left_range = parse_voltage_range(args.linear_left_range)
    linear_right_range = parse_voltage_range(args.linear_right_range)
    return {
        "linear_correction": bool(args.linear_correction),
        "gaussian_correction": bool(args.gaussian_correction),
        "linear_edge_fraction": float(args.linear_edge_fraction),
        "linear_left_range": linear_left_range,
        "linear_right_range": linear_right_range,
        "linear_left_range_text": args.linear_left_range,
        "linear_right_range_text": args.linear_right_range,
        "gaussian_fit_start_v": float(args.gaussian_fit_start),
        "gaussian_center_v": float(args.gaussian_center),
        "gaussian_sigma_v": float(args.gaussian_sigma),
        "gaussian_fit_mode": args.gaussian_fit_mode,
        "allow_negative_gaussian": bool(args.allow_negative_gaussian),
    }


def correction_label(config: dict[str, object]) -> str:
    labels = []
    if config["linear_correction"]:
        labels.append("linear drift corrected")
    if config["gaussian_correction"]:
        labels.append("Gaussian tail corrected")
    if not labels:
        return "no correction"
    return " + ".join(labels)


def parse_voltage_range(value: str | None) -> tuple[float, float] | None:
    if value is None:
        return None
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if len(parts) != 2:
        raise ValueError(f"Voltage range must be 'min,max', got: {value!r}")
    lower = float(parts[0])
    upper = float(parts[1])
    if upper <= lower:
        raise ValueError(f"Voltage range upper bound must be greater than lower bound: {value!r}")
    return lower, upper


def edge_mask(
    x_values: np.ndarray,
    fraction: float,
    voltage_range: tuple[float, float] | None,
    side: str,
) -> np.ndarray:
    if voltage_range is not None:
        lower, upper = voltage_range
        mask = (x_values >= lower) & (x_values <= upper)
        if np.count_nonzero(mask) >= 2:
            return mask
    count = max(2, int(round(len(x_values) * max(0.0, min(float(fraction), 0.45)))))
    if side == "left":
        mask = np.zeros(len(x_values), dtype=bool)
        mask[:count] = True
    else:
        mask = np.zeros(len(x_values), dtype=bool)
        mask[-count:] = True
    return mask


def apply_linear_correction(
    x_values: np.ndarray,
    y_values: np.ndarray,
    config: dict[str, object],
) -> np.ndarray:
    left_range = config.get("linear_left_range")
    right_range = config.get("linear_right_range")
    left_mask = edge_mask(x_values, float(config["linear_edge_fraction"]), left_range, "left")
    right_mask = edge_mask(x_values, float(config["linear_edge_fraction"]), right_range, "right")
    x_left = float(np.mean(x_values[left_mask]))
    y_left = float(np.mean(y_values[left_mask]))
    x_right = float(np.mean(x_values[right_mask]))
    y_right = float(np.mean(y_values[right_mask]))
    if abs(x_right - x_left) < 1e-12:
        return y_values.copy()
    slope = (y_right - y_left) / (x_right - x_left)
    baseline = y_left + slope * (x_values - x_left)
    return y_values - baseline


def apply_gaussian_tail_correction(
    x_values: np.ndarray,
    y_values: np.ndarray,
    config: dict[str, object],
) -> np.ndarray:
    sigma = float(config["gaussian_sigma_v"])
    if sigma <= 0:
        raise ValueError("--gaussian-sigma must be > 0")
    center = float(config["gaussian_center_v"])
    fit_start = float(config["gaussian_fit_start_v"])
    fit_mask = x_values >= fit_start
    if np.count_nonzero(fit_mask) < 3:
        fit_mask = np.ones(len(x_values), dtype=bool)

    if str(config.get("gaussian_fit_mode", "optimize")) == "optimize":
        amplitude, center, sigma = fit_gaussian_tail(
            x_values[fit_mask],
            y_values[fit_mask],
            center,
            sigma,
            fit_start,
            bool(config["allow_negative_gaussian"]),
        )
    else:
        basis = np.exp(-0.5 * ((x_values - center) / sigma) ** 2)
        denominator = float(np.dot(basis[fit_mask], basis[fit_mask]))
        if denominator <= 0:
            return y_values.copy()
        amplitude = float(np.dot(basis[fit_mask], y_values[fit_mask]) / denominator)
        if not bool(config["allow_negative_gaussian"]):
            amplitude = max(0.0, amplitude)

    basis = np.exp(-0.5 * ((x_values - center) / sigma) ** 2)
    return y_values - amplitude * basis


def gaussian_function(x_values: np.ndarray, amplitude: float, center: float, sigma: float) -> np.ndarray:
    return amplitude * np.exp(-0.5 * ((x_values - center) / sigma) ** 2)


def fit_gaussian_tail(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    initial_center: float,
    initial_sigma: float,
    fit_start: float,
    allow_negative: bool,
) -> tuple[float, float, float]:
    x_fit = np.asarray(x_fit, dtype=float)
    y_target = np.asarray(y_fit, dtype=float)
    if not allow_negative:
        y_target = np.maximum(y_target, 0.0)
    if len(x_fit) < 3:
        return 0.0, initial_center, initial_sigma

    amplitude0 = float(np.max(np.abs(y_target)))
    if not allow_negative:
        amplitude0 = float(np.max(y_target))
    if amplitude0 <= 0:
        return 0.0, initial_center, initial_sigma

    x_max = float(np.max(x_fit))
    center_lower = min(fit_start, x_max)
    center_upper = x_max + 0.30
    sigma_lower = 0.015
    sigma_upper = 0.40

    if curve_fit is not None:
        amp_lower = -np.inf if allow_negative else 0.0
        try:
            params, _cov = curve_fit(
                gaussian_function,
                x_fit,
                y_target,
                p0=[amplitude0, initial_center, initial_sigma],
                bounds=(
                    [amp_lower, center_lower, sigma_lower],
                    [np.inf, center_upper, sigma_upper],
                ),
                maxfev=10000,
            )
            amplitude, center, sigma = [float(value) for value in params]
            if not allow_negative:
                amplitude = max(0.0, amplitude)
            return amplitude, center, max(sigma, sigma_lower)
        except Exception:
            pass

    best = (math.inf, 0.0, initial_center, initial_sigma)
    centers = np.linspace(center_lower, center_upper, 80)
    sigmas = np.linspace(sigma_lower, sigma_upper, 80)
    for center in centers:
        for sigma in sigmas:
            basis = np.exp(-0.5 * ((x_fit - center) / sigma) ** 2)
            denominator = float(np.dot(basis, basis))
            if denominator <= 0:
                continue
            amplitude = float(np.dot(basis, y_target) / denominator)
            if not allow_negative:
                amplitude = max(0.0, amplitude)
            residual = y_target - amplitude * basis
            error = float(np.mean(residual ** 2))
            if error < best[0]:
                best = (error, amplitude, float(center), float(sigma))
    _error, amplitude, center, sigma = best
    return amplitude, center, sigma


def apply_curve_corrections(
    x_values: np.ndarray,
    y_values: np.ndarray,
    config: dict[str, object],
) -> np.ndarray:
    corrected = np.asarray(y_values, dtype=float).copy()
    x_array = np.asarray(x_values, dtype=float)
    if config["linear_correction"]:
        corrected = apply_linear_correction(x_array, corrected, config)
    if config["gaussian_correction"]:
        corrected = apply_gaussian_tail_correction(x_array, corrected, config)
    return corrected


def apply_corrections_to_entries(
    entries: list[dict[str, object]],
    config: dict[str, object],
) -> list[dict[str, object]]:
    if not config["linear_correction"] and not config["gaussian_correction"]:
        return entries
    corrected_entries = []
    for entry in entries:
        corrected = dict(entry)
        corrected["y"] = apply_curve_corrections(
            np.asarray(entry["x"], dtype=float),
            np.asarray(entry["y"], dtype=float),
            config,
        )
        corrected["correction_applied"] = correction_label(config)
        corrected_entries.append(corrected)
    return corrected_entries


def write_json(path: Path, data: dict[str, object]) -> None:
    with path.open("w") as handle:
        json.dump(data, handle, indent=2, default=json_default)
        handle.write("\n")


def design_value(row: dict[str, str], analyte: str) -> float:
    return float(row[ANALYTES[analyte]["column"]])


def scan_multi_analyte_conditions() -> list[dict[str, object]]:
    conditions = []
    for day, config in DAY_CONFIGS.items():
        for row in read_design_table(config["design_csv"]):
            values = {analyte: design_value(row, analyte) for analyte in ANALYTE_ORDER}
            nonzero_count = sum(1 for value in values.values() if abs(value) > 1e-12)
            if nonzero_count < 2:
                continue
            condition = int(float(row["Condition"]))
            conditions.append(
                {
                    "day": day,
                    "condition": condition,
                    "condition_code": condition_code(day, condition),
                    "da_uM": values["DA"],
                    "aa_uM": values["AA"],
                    "ua_uM": values["UA"],
                    "condition_folder": config["output_dir"] / f"Condition_{condition}",
                }
            )
    return conditions


def write_multi_condition_manifest(rows: list[dict[str, object]], output_path: Path) -> None:
    fieldnames = [
        "day",
        "condition",
        "condition_code",
        "dopamine_uM",
        "ascorbic_acid_uM",
        "uric_acid_uM",
        "condition_folder",
    ]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "day": row["day"],
                    "condition": row["condition"],
                    "condition_code": row["condition_code"],
                    "dopamine_uM": clean_value(row["da_uM"]),
                    "ascorbic_acid_uM": clean_value(row["aa_uM"]),
                    "uric_acid_uM": clean_value(row["ua_uM"]),
                    "condition_folder": str(Path(row["condition_folder"]).relative_to(ROOT)),
                }
            )


def load_actual_mean_curves(
    condition_row: dict[str, object],
    config: dict[str, object],
) -> dict[str, dict[str, object]]:
    condition_folder = Path(condition_row["condition_folder"])
    curves = {}
    if not condition_folder.is_dir():
        print(f"Missing condition folder: {condition_folder}")
        return curves

    for technique_key, technique_config in TECHNIQUES.items():
        source_file = find_technique_file(condition_folder, str(technique_config["filename_pattern"]))
        if source_file is None:
            print(
                f"Missing {technique_config['label']} file for "
                f"D{condition_row['day']} Condition {condition_row['condition']}"
            )
            continue
        rows = load_numeric_rows(source_file)
        sweeps = split_cv_sweeps(rows)
        for sweep_key, sweep_rows in sweeps.items():
            traces = extract_electrode_traces(sweep_rows, technique_config["electrode_columns"])
            mean_trace = mean_trace_from_electrodes(traces)
            if mean_trace is None:
                continue
            x_values, y_values = mean_trace
            y_corrected = apply_curve_corrections(x_values, y_values, config)
            curve_key = f"{technique_key}_{sweep_key}"
            curves[curve_key] = {
                "technique": technique_key,
                "technique_label": TECHNIQUES[technique_key]["label"],
                "sweep": sweep_key,
                "sweep_label": SWEEPS[sweep_key],
                "x": np.asarray(x_values, dtype=float),
                "y": np.asarray(y_corrected, dtype=float),
                "raw_y": np.asarray(y_values, dtype=float),
                "correction_applied": correction_label(config),
                "source_file": str(source_file.relative_to(ROOT)),
            }
    return curves


def build_component_models(
    component_root: Path,
    config: dict[str, object],
) -> tuple[dict[str, dict[str, object]], dict[str, dict[str, object]]]:
    isolated_series = scan_isolated_conditions()
    prediction_models = {}
    setup_metadata = {}
    for analyte in ANALYTE_ORDER:
        selected_row = select_best_setup(component_root, analyte)
        params = params_from_setup_row(selected_row)
        entries = load_trace_entries_for_series(isolated_series[analyte])
        entries = apply_corrections_to_entries(entries, config)
        models = fit_pcr_models(entries, params)
        prediction_models[analyte] = build_prediction_model(models, params, scope="mean")
        setup_metadata[analyte] = {
            "selected_row": selected_row,
            "params": params,
            "n_trace_entries": len(entries),
            "correction": config,
        }
    return prediction_models, setup_metadata


def interpolate_curve(curve: dict[str, object], x_target: np.ndarray) -> np.ndarray:
    return np.interp(
        x_target,
        np.asarray(curve["potential_v"], dtype=float),
        np.asarray(curve["current_uA"], dtype=float),
    )


def component_prediction_for_condition(
    prediction_models: dict[str, dict[str, object]],
    concentrations: dict[str, float],
    x_target: np.ndarray,
    curve_key: str,
    prediction_points: int,
    baseline_source: str,
    superposition_mode: str,
) -> dict[str, object]:
    full_curves = {
        analyte: predict_curves(
            prediction_models[analyte],
            concentrations[analyte],
            points=prediction_points,
        )[curve_key]
        for analyte in ANALYTE_ORDER
    }
    zero_curves = {
        analyte: predict_curves(
            prediction_models[analyte],
            0.0,
            points=prediction_points,
        )[curve_key]
        for analyte in ANALYTE_ORDER
    }

    full_y = {
        analyte: interpolate_curve(full_curves[analyte], x_target)
        for analyte in ANALYTE_ORDER
    }
    zero_y = {
        analyte: interpolate_curve(zero_curves[analyte], x_target)
        for analyte in ANALYTE_ORDER
    }

    if superposition_mode in {"full_curve_sum", "raw_sum"}:
        predicted = sum(full_y.values())
        baseline = np.zeros_like(x_target)
    else:
        deltas = {
            analyte: full_y[analyte] - zero_y[analyte]
            for analyte in ANALYTE_ORDER
        }
        if baseline_source == "mean":
            baseline = np.mean(np.vstack([zero_y[analyte] for analyte in ANALYTE_ORDER]), axis=0)
        else:
            baseline = zero_y[baseline_source]
        predicted = baseline + sum(deltas.values())

    return {
        "single_analyte_curves": full_y,
        "deltas": {
            analyte: full_y[analyte] - zero_y[analyte]
            for analyte in ANALYTE_ORDER
        },
        "full_curves": full_y,
        "zero_curves": zero_y,
        "baseline": baseline,
        "predicted": predicted,
    }


def residual_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    residual = actual - predicted
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    mae = float(np.mean(np.abs(residual)))
    max_abs = float(np.max(np.abs(residual)))
    span = float(np.max(actual) - np.min(actual))
    normalized_rmse = rmse / span if span > 0 else math.nan
    return {
        "rmse_uA": rmse,
        "mae_uA": mae,
        "max_abs_residual_uA": max_abs,
        "normalized_rmse": normalized_rmse,
        "r2": safe_r2(actual, predicted),
    }


def style_axis(ax: plt.Axes) -> None:
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


def set_shared_row_ylim(axes: list[plt.Axes], y_arrays: list[np.ndarray]) -> None:
    finite = np.concatenate([array[np.isfinite(array)] for array in y_arrays if np.any(np.isfinite(array))])
    if len(finite) == 0:
        return
    y_min = float(np.min(finite))
    y_max = float(np.max(finite))
    span = y_max - y_min
    if span <= 0:
        span = max(abs(y_max), 1.0)
    pad = 0.08 * span
    for ax in axes:
        ax.set_ylim(y_min - pad, y_max + pad)


def plot_condition_page(
    pdf: PdfPages,
    condition_row: dict[str, object],
    actual_curves: dict[str, dict[str, object]],
    predictions_by_curve: dict[str, dict[str, object]],
    metrics_by_curve: dict[str, dict[str, float]],
    superposition_mode: str,
    config: dict[str, object],
) -> None:
    fig, axes = plt.subplots(4, 4, figsize=(17, 13.2))
    fig.suptitle(
        (
            f"Linear superposition test | Day {condition_row['day']} Condition {condition_row['condition']} | "
            f"DA {clean_value(condition_row['da_uM'])} uM, "
            f"AA {clean_value(condition_row['aa_uM'])} uM, "
            f"UA {clean_value(condition_row['ua_uM'])} uM | "
            f"{correction_label(config)}"
        ),
        fontweight="semibold",
    )
    column_titles = [
        "DA single-analyte PCR curve",
        "AA single-analyte PCR curve",
        "UA single-analyte PCR curve",
        "Actual mixture vs summed PCR curves",
    ]
    concentrations = {
        "DA": float(condition_row["da_uM"]),
        "AA": float(condition_row["aa_uM"]),
        "UA": float(condition_row["ua_uM"]),
    }

    for row_index, (technique, sweep) in enumerate(PANEL_ORDER):
        curve_key = f"{technique}_{sweep}"
        actual = actual_curves.get(curve_key)
        prediction = predictions_by_curve.get(curve_key)
        if actual is None or prediction is None:
            for col_index in range(4):
                axes[row_index, col_index].text(0.5, 0.5, "No data", ha="center", va="center", transform=axes[row_index, col_index].transAxes)
            continue

        x_values = actual["x"]
        row_y_arrays = []
        for col_index, analyte in enumerate(ANALYTE_ORDER):
            ax = axes[row_index, col_index]
            style_axis(ax)
            if row_index == 0:
                ax.set_title(column_titles[col_index])
            if col_index == 0:
                ax.text(
                    -0.24,
                    0.5,
                    f"{TECHNIQUES[technique]['label']}\n{SWEEPS[sweep]}",
                    rotation=90,
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=8,
                )
            y_values = prediction["single_analyte_curves"][analyte]
            row_y_arrays.append(y_values)
            ax.plot(
                x_values,
                y_values,
                color=ANALYTES[analyte]["color"],
                lw=1.4,
                label=f"isolated {analyte} prediction\n{clean_value(concentrations[analyte])} uM",
            )
            ax.legend(loc="best", fontsize=6)

        ax = axes[row_index, 3]
        style_axis(ax)
        if row_index == 0:
            ax.set_title(column_titles[3])
        ax.plot(x_values, actual["y"], color="#333333", lw=1.1, label="actual mixture")
        ax.plot(x_values, prediction["predicted"], color="#D55E00", lw=1.5, ls="--", label="summed PCR curves")
        if superposition_mode == "baseline_delta":
            ax.plot(x_values, prediction["baseline"], color="#999999", lw=0.8, ls=":", label="baseline")
        metrics = metrics_by_curve[curve_key]
        ax.text(
            0.02,
            0.97,
            f"R2={metrics['r2']:.3f}\nRMSE={metrics['rmse_uA']:.3g} {CURRENT_UNIT}",
            ha="left",
            va="top",
            transform=ax.transAxes,
            fontsize=7,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75},
        )
        ax.legend(loc="best", fontsize=6)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    pdf.savefig(fig)
    plt.close(fig)


def write_metrics_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    fieldname: format_metric_value(row.get(fieldname, ""))
                    for fieldname in METRIC_FIELDNAMES
                }
            )


def format_metric_value(value: object) -> object:
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return ""
        return f"{float(value):.12g}"
    return value


def run(args: argparse.Namespace) -> None:
    configure_plot_style()
    config = correction_config(args)
    output_root = default_output_root(args).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    component_root = args.component_root.resolve()

    prediction_models, setup_metadata = build_component_models(component_root, config)
    multi_conditions = scan_multi_analyte_conditions()
    if args.limit_conditions is not None:
        multi_conditions = multi_conditions[: args.limit_conditions]
    write_multi_condition_manifest(multi_conditions, output_root / "multi_analyte_conditions.csv")
    write_json(output_root / "component_setup_metadata.json", setup_metadata)
    write_json(output_root / "correction_config.json", config)

    pdf_path = output_root / "multi_analyte_linear_superposition.pdf"
    metrics_path = output_root / "multi_analyte_superposition_metrics.csv"
    metric_rows = []

    with PdfPages(pdf_path) as pdf:
        for index, condition_row in enumerate(multi_conditions, 1):
            actual_curves = load_actual_mean_curves(condition_row, config)
            concentrations = {
                "DA": float(condition_row["da_uM"]),
                "AA": float(condition_row["aa_uM"]),
                "UA": float(condition_row["ua_uM"]),
            }
            predictions_by_curve = {}
            metrics_by_curve = {}
            for curve_key, actual in actual_curves.items():
                prediction = component_prediction_for_condition(
                    prediction_models=prediction_models,
                    concentrations=concentrations,
                    x_target=actual["x"],
                    curve_key=curve_key,
                    prediction_points=args.prediction_points,
                    baseline_source=args.baseline_source,
                    superposition_mode=args.superposition_mode,
                )
                metrics = residual_metrics(actual["y"], prediction["predicted"])
                predictions_by_curve[curve_key] = prediction
                metrics_by_curve[curve_key] = metrics
                metric_rows.append(
                    {
                        "day": condition_row["day"],
                        "condition": condition_row["condition"],
                        "condition_code": condition_row["condition_code"],
                        "dopamine_uM": condition_row["da_uM"],
                        "ascorbic_acid_uM": condition_row["aa_uM"],
                        "uric_acid_uM": condition_row["ua_uM"],
                        "technique": actual["technique"],
                        "sweep": actual["sweep"],
                        "n_points": len(actual["x"]),
                        "superposition_mode": args.superposition_mode,
                        "baseline_source": args.baseline_source,
                        "linear_correction": config["linear_correction"],
                        "gaussian_correction": config["gaussian_correction"],
                        "linear_edge_fraction": config["linear_edge_fraction"],
                        "linear_left_range_v": config["linear_left_range_text"],
                        "linear_right_range_v": config["linear_right_range_text"],
                        "gaussian_fit_start_v": config["gaussian_fit_start_v"],
                        "gaussian_center_v": config["gaussian_center_v"],
                        "gaussian_sigma_v": config["gaussian_sigma_v"],
                        "gaussian_fit_mode": config["gaussian_fit_mode"],
                        **metrics,
                    }
                )

            plot_condition_page(
                pdf=pdf,
                condition_row=condition_row,
                actual_curves=actual_curves,
                predictions_by_curve=predictions_by_curve,
                metrics_by_curve=metrics_by_curve,
                superposition_mode=args.superposition_mode,
                config=config,
            )
            if index % args.progress_every == 0 or index == len(multi_conditions):
                print(f"Processed {index}/{len(multi_conditions)} multi-analyte conditions")

    write_metrics_csv(metric_rows, metrics_path)
    print(f"Wrote superposition PDF: {pdf_path}")
    print(f"Wrote superposition metrics CSV: {metrics_path}")
    print(f"Wrote multi-analyte manifest: {output_root / 'multi_analyte_conditions.csv'}")
    print(f"Correction mode: {correction_label(config)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare actual multi-analyte CV curves with linearly superposed isolated-analyte PCR predictions."
    )
    parser.add_argument("--component-root", type=Path, default=DEFAULT_COMPONENT_ROOT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Output directory. Defaults to All_Analyte_Superposition_Analysis, "
            "or a correction-specific suffix when correction flags are used."
        ),
    )
    parser.add_argument(
        "--superposition-mode",
        choices=["full_curve_sum", "baseline_delta", "raw_sum"],
        default="full_curve_sum",
        help=(
            "full_curve_sum directly sums final isolated PCR-predicted CV curves. "
            "baseline_delta uses baseline + sum(PCR(c)-PCR(0)). "
            "raw_sum is kept as an alias for full_curve_sum. Default: full_curve_sum."
        ),
    )
    parser.add_argument(
        "--baseline-source",
        choices=["mean", "DA", "AA", "UA"],
        default="mean",
        help="Baseline used for baseline_delta mode. Default: mean of DA/AA/UA zero predictions.",
    )
    parser.add_argument("--prediction-points", type=int, default=500)
    parser.add_argument("--limit-conditions", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument(
        "--linear-correction",
        action="store_true",
        help="Subtract a straight-line baseline estimated from the low/high voltage edges before PCR and comparison.",
    )
    parser.add_argument(
        "--linear-edge-fraction",
        type=float,
        default=0.08,
        help=(
            "Fallback fraction of points used for baseline anchors if explicit ranges are unavailable. "
            "Default: 0.08."
        ),
    )
    parser.add_argument(
        "--linear-left-range",
        default="0.10,0.15",
        help="Low-potential linear baseline anchor range as 'min,max' volts. Default: 0.10,0.15.",
    )
    parser.add_argument(
        "--linear-right-range",
        default="0.60,0.68",
        help=(
            "Pre-water-oxidation linear baseline anchor range as 'min,max' volts. "
            "Default: 0.60,0.68."
        ),
    )
    parser.add_argument(
        "--gaussian-correction",
        action="store_true",
        help="Subtract a fitted broad Gaussian tail from the high-potential region before PCR and comparison.",
    )
    parser.add_argument(
        "--gaussian-fit-start",
        type=float,
        default=0.70,
        help="Voltage where Gaussian tail fitting starts. Default: 0.70 V.",
    )
    parser.add_argument(
        "--gaussian-center",
        type=float,
        default=0.90,
        help="Gaussian tail center voltage. Default: 0.90 V.",
    )
    parser.add_argument(
        "--gaussian-sigma",
        type=float,
        default=0.08,
        help="Gaussian tail width in volts. Default: 0.08 V.",
    )
    parser.add_argument(
        "--gaussian-fit-mode",
        choices=["optimize", "fixed"],
        default="optimize",
        help=(
            "Gaussian tail fitting mode. optimize fits amplitude/center/sigma in the tail region; "
            "fixed only fits amplitude using --gaussian-center and --gaussian-sigma. Default: optimize."
        ),
    )
    parser.add_argument(
        "--allow-negative-gaussian",
        action="store_true",
        help="Allow the Gaussian tail amplitude to be negative. By default only positive tails are removed.",
    )
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
