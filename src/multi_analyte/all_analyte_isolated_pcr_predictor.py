#!/usr/bin/env python3
"""PCR sweeps and interactive predictor for isolated DA/AA/UA calibration sets.

This is intentionally separate from the Day 1 DA-only workflow. It scans the
Day 1-4 experiment design tables, finds conditions where only one analyte is
present, fits one PCR model per analyte, and opens a single interactive viewer
for generating predicted CV/CV-GC curves for DA, AA, and UA independently.
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
import time
from itertools import product
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

from day1_da_only_cls_core import (
    CURRENT_UNIT,
    DEFAULT_GRID_POINTS,
    DEFAULT_SMOOTH_POLYORDER,
    DEFAULT_SMOOTH_WINDOW,
    PANEL_ORDER,
    RESULTS_ROOT,
    ROOT,
    SWEEPS,
    TECHNIQUES,
    extract_electrode_traces,
    find_technique_file,
    load_numeric_rows,
    mean_trace_from_electrodes,
    split_cv_sweeps,
)
from day1_da_only_pcr_core import (
    DEFAULT_N_COMPONENTS,
    DEFAULT_SCORE_TREND_DEGREE,
    build_prediction_model,
    clean_value,
    fit_pcr_models,
    format_float,
    format_series,
    json_default,
    parse_int_list,
    predict_curves,
    slug_float,
)


EXPERIMENT_ROOT = ROOT / "Day 1-4 Outputs w. TXT"
DEFAULT_OUTPUT_ROOT = RESULTS_ROOT / "All_Analyte_Isolated_PCR"

DAY_CONFIGS = {
    1: {
        "design_csv": EXPERIMENT_ROOT / "Conditions for Experiments - Day 1 (UA-0).csv",
        "output_dir": EXPERIMENT_ROOT / "Outputs_Day_1",
    },
    2: {
        "design_csv": EXPERIMENT_ROOT / "Conditions for Experiments - Day 2 (UA-100).csv",
        "output_dir": EXPERIMENT_ROOT / "Outputs_Day_2_with_txt",
    },
    3: {
        "design_csv": EXPERIMENT_ROOT / "Conditions for Experiments - Day 3 (UA-200).csv",
        "output_dir": EXPERIMENT_ROOT / "Outputs_Day_3_with_txt",
    },
    4: {
        "design_csv": EXPERIMENT_ROOT / "Conditions for Experiments - Day 4 (UA-400).csv",
        "output_dir": EXPERIMENT_ROOT / "Outputs_Day_4_with_txt",
    },
}

ANALYTES = {
    "DA": {"label": "Dopamine", "column": "Dopamine (uM)", "color": "#D55E00"},
    "AA": {"label": "Ascorbic Acid", "column": "Ascorbic Acid (uM)", "color": "#0072B2"},
    "UA": {"label": "Uric Acid", "column": "Uric Acid (uM)", "color": "#009E73"},
}
ANALYTE_ORDER = ["DA", "AA", "UA"]

DETAIL_FIELDNAMES = [
    "analyte",
    "setup_id",
    "method",
    "n_components",
    "requested_n_components",
    "score_trend_degree",
    "score_trend_degree_effective",
    "smooth_window",
    "smooth_polyorder",
    "grid_points",
    "scope",
    "technique",
    "sweep",
    "electrode",
    "n_concentrations",
    "n_voltages",
    "cumulative_explained_variance_ratio",
    "avg_abs_pc_pearson",
    "avg_abs_pc_spearman",
    "avg_pc_trend_r2",
    "pc_pearson_r",
    "pc_spearman_r",
    "pc_trend_r2",
    "pc_explained_variance_ratio",
    "rmse_uA",
    "mae_uA",
    "max_abs_residual_uA",
    "normalized_rmse",
    "r2",
    "condition_codes",
    "concentration_values_uM",
    "per_condition_rmse_uA",
    "per_condition_mae_uA",
    "per_condition_r2",
]

SUMMARY_FIELDNAMES = [
    "analyte",
    "setup_id",
    "method",
    "n_components",
    "requested_n_components",
    "score_trend_degree",
    "score_trend_degree_effective",
    "smooth_window",
    "smooth_polyorder",
    "grid_points",
    "scope_summary",
    "n_detail_rows",
    "avg_cumulative_explained_variance_ratio",
    "avg_abs_pc_pearson",
    "avg_abs_pc_spearman",
    "avg_pc_trend_r2",
    "avg_rmse_uA",
    "median_rmse_uA",
    "max_rmse_uA",
    "avg_mae_uA",
    "avg_max_abs_residual_uA",
    "avg_normalized_rmse",
    "median_normalized_rmse",
    "avg_r2",
    "median_r2",
    "min_r2",
    "objective_score",
]


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "legend.frameon": False,
            "lines.linewidth": 1.2,
        }
    )


def read_design_table(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = []
        for row in csv.DictReader(handle):
            cleaned = {
                key.strip(): (value.strip() if value is not None else "")
                for key, value in row.items()
            }
            if any(cleaned.values()):
                rows.append(cleaned)
    return rows


def value_from_row(row: dict[str, str], analyte: str) -> float:
    return float(row[ANALYTES[analyte]["column"]])


def condition_code(day: int, condition: int) -> int:
    return day * 100 + condition


def scan_isolated_conditions() -> dict[str, list[dict[str, object]]]:
    series = {analyte: [] for analyte in ANALYTE_ORDER}
    for day, config in DAY_CONFIGS.items():
        rows = read_design_table(config["design_csv"])
        for row in rows:
            condition = int(float(row["Condition"]))
            values = {analyte: value_from_row(row, analyte) for analyte in ANALYTE_ORDER}
            nonzero = [analyte for analyte, value in values.items() if abs(value) > 1e-12]
            if not nonzero:
                target_analytes = ANALYTE_ORDER
            elif len(nonzero) == 1:
                target_analytes = nonzero
            else:
                continue

            for analyte in target_analytes:
                concentration = values[analyte]
                condition_folder = config["output_dir"] / f"Condition_{condition}"
                series[analyte].append(
                    {
                        "analyte": analyte,
                        "analyte_label": ANALYTES[analyte]["label"],
                        "day": day,
                        "condition": condition,
                        "condition_code": condition_code(day, condition),
                        "concentration_uM": concentration,
                        "da_uM": values["DA"],
                        "aa_uM": values["AA"],
                        "ua_uM": values["UA"],
                        "condition_folder": condition_folder,
                    }
                )

    for analyte in ANALYTE_ORDER:
        seen = set()
        unique_rows = []
        for row in sorted(series[analyte], key=lambda item: (float(item["concentration_uM"]), int(item["day"]), int(item["condition"]))):
            key = (float(row["concentration_uM"]), int(row["day"]), int(row["condition"]))
            if key in seen:
                continue
            seen.add(key)
            unique_rows.append(row)
        series[analyte] = unique_rows
    return series


def write_isolated_manifest(series: dict[str, list[dict[str, object]]], output_path: Path) -> None:
    fieldnames = [
        "analyte",
        "analyte_label",
        "calibration_concentration_uM",
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
        for analyte in ANALYTE_ORDER:
            for row in series[analyte]:
                writer.writerow(
                    {
                        "analyte": analyte,
                        "analyte_label": row["analyte_label"],
                        "calibration_concentration_uM": clean_value(row["concentration_uM"]),
                        "day": row["day"],
                        "condition": row["condition"],
                        "condition_code": row["condition_code"],
                        "dopamine_uM": clean_value(row["da_uM"]),
                        "ascorbic_acid_uM": clean_value(row["aa_uM"]),
                        "uric_acid_uM": clean_value(row["ua_uM"]),
                        "condition_folder": str(Path(row["condition_folder"]).relative_to(ROOT)),
                    }
                )


def load_trace_entries_for_series(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    entries = []
    for calibration_row in rows:
        condition_folder = Path(calibration_row["condition_folder"])
        if not condition_folder.is_dir():
            print(f"Missing condition folder: {condition_folder}")
            continue

        for technique_key, technique_config in TECHNIQUES.items():
            source_file = find_technique_file(condition_folder, str(technique_config["filename_pattern"]))
            if source_file is None:
                print(
                    f"Missing {technique_config['label']} file for "
                    f"D{calibration_row['day']} Condition {calibration_row['condition']}"
                )
                continue
            rows_numeric = load_numeric_rows(source_file)
            sweeps = split_cv_sweeps(rows_numeric)
            for sweep_key, sweep_rows in sweeps.items():
                traces = extract_electrode_traces(sweep_rows, technique_config["electrode_columns"])
                for electrode, (x_values, y_values) in traces.items():
                    entries.append(
                        {
                            "scope": "electrode",
                            "condition": int(calibration_row["condition_code"]),
                            "condition_label": f"D{calibration_row['day']}C{calibration_row['condition']}",
                            "day": int(calibration_row["day"]),
                            "original_condition": int(calibration_row["condition"]),
                            "da_uM": float(calibration_row["concentration_uM"]),
                            "analyte": calibration_row["analyte"],
                            "technique": technique_key,
                            "sweep": sweep_key,
                            "electrode": electrode,
                            "x": x_values,
                            "y": y_values,
                            "source_file": str(source_file.relative_to(ROOT)),
                        }
                    )
                mean_trace = mean_trace_from_electrodes(traces)
                if mean_trace is not None:
                    mean_x, mean_y = mean_trace
                    entries.append(
                        {
                            "scope": "mean",
                            "condition": int(calibration_row["condition_code"]),
                            "condition_label": f"D{calibration_row['day']}C{calibration_row['condition']}",
                            "day": int(calibration_row["day"]),
                            "original_condition": int(calibration_row["condition"]),
                            "da_uM": float(calibration_row["concentration_uM"]),
                            "analyte": calibration_row["analyte"],
                            "technique": technique_key,
                            "sweep": sweep_key,
                            "electrode": "mean",
                            "x": mean_x,
                            "y": mean_y,
                            "source_file": str(source_file.relative_to(ROOT)),
                        }
                    )
    return entries


def selected_scopes(scope: str) -> set[str]:
    if scope == "both":
        return {"mean", "electrode"}
    return {scope}


def nanmean(values: list[float]) -> float:
    finite = [value for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else math.nan


def nanmedian(values: list[float]) -> float:
    finite = [value for value in values if np.isfinite(value)]
    return float(np.median(finite)) if finite else math.nan


def nanmin(values: list[float]) -> float:
    finite = [value for value in values if np.isfinite(value)]
    return float(np.min(finite)) if finite else math.nan


def nanmax(values: list[float]) -> float:
    finite = [value for value in values if np.isfinite(value)]
    return float(np.max(finite)) if finite else math.nan


def build_detail_rows(
    analyte: str,
    setup_id: int,
    params: dict[str, object],
    models: list[dict[str, object]],
) -> list[dict[str, object]]:
    detail_rows = []
    for model in models:
        detail_rows.append(
            {
                "analyte": analyte,
                "setup_id": setup_id,
                "method": "pcr",
                "n_components": model["n_components"],
                "requested_n_components": model["requested_n_components"],
                "score_trend_degree": model["score_trend_degree"],
                "score_trend_degree_effective": model["score_trend_degree_effective"],
                "smooth_window": params["smooth_window"],
                "smooth_polyorder": model["smooth_polyorder_effective"],
                "grid_points": params["grid_points"],
                "scope": model["scope"],
                "technique": model["technique"],
                "sweep": model["sweep"],
                "electrode": model["electrode"],
                "n_concentrations": model["n_concentrations"],
                "n_voltages": model["n_voltages"],
                "cumulative_explained_variance_ratio": model["cumulative_explained_variance_ratio"],
                "avg_abs_pc_pearson": model["avg_abs_pc_pearson"],
                "avg_abs_pc_spearman": model["avg_abs_pc_spearman"],
                "avg_pc_trend_r2": model["avg_pc_trend_r2"],
                "pc_pearson_r": format_series([row["pearson_r"] for row in model["score_correlations"]]),
                "pc_spearman_r": format_series([row["spearman_r"] for row in model["score_correlations"]]),
                "pc_trend_r2": format_series([row["trend_r2"] for row in model["score_correlations"]]),
                "pc_explained_variance_ratio": format_series(model["explained_variance_ratio"]),
                "rmse_uA": model["rmse_uA"],
                "mae_uA": model["mae_uA"],
                "max_abs_residual_uA": model["max_abs_residual_uA"],
                "normalized_rmse": model["normalized_rmse"],
                "r2": model["r2"],
                "condition_codes": format_series(model["conditions"]),
                "concentration_values_uM": format_series(model["concentrations"]),
                "per_condition_rmse_uA": format_series(model["per_condition_rmse_uA"]),
                "per_condition_mae_uA": format_series(model["per_condition_mae_uA"]),
                "per_condition_r2": format_series(model["per_condition_r2"]),
            }
        )
    return detail_rows


def build_summary_row(detail_rows: list[dict[str, object]], scope_summary: str) -> dict[str, object]:
    if not detail_rows:
        raise ValueError("Cannot summarize empty PCR detail rows")
    first = detail_rows[0]
    rows = detail_rows if scope_summary == "all" else [
        row for row in detail_rows if row["scope"] == scope_summary
    ]
    if not rows:
        return {}

    rmse = [float(row["rmse_uA"]) for row in rows]
    mae = [float(row["mae_uA"]) for row in rows]
    max_abs = [float(row["max_abs_residual_uA"]) for row in rows]
    nrmse = [float(row["normalized_rmse"]) for row in rows]
    r2 = [float(row["r2"]) for row in rows]
    cum_var = [float(row["cumulative_explained_variance_ratio"]) for row in rows]
    abs_pearson = [float(row["avg_abs_pc_pearson"]) for row in rows]
    abs_spearman = [float(row["avg_abs_pc_spearman"]) for row in rows]
    trend_r2 = [float(row["avg_pc_trend_r2"]) for row in rows]

    avg_r2 = nanmean(r2)
    avg_nrmse = nanmean(nrmse)
    avg_trend_r2 = nanmean(trend_r2)
    avg_abs_pearson = nanmean(abs_pearson)
    objective_score = math.nan
    if np.isfinite(avg_r2) and np.isfinite(avg_nrmse):
        objective_score = avg_r2 - avg_nrmse
        if np.isfinite(avg_trend_r2):
            objective_score += 0.05 * avg_trend_r2
        if np.isfinite(avg_abs_pearson):
            objective_score += 0.05 * avg_abs_pearson

    return {
        "analyte": first["analyte"],
        "setup_id": first["setup_id"],
        "method": "pcr",
        "n_components": first["n_components"],
        "requested_n_components": first["requested_n_components"],
        "score_trend_degree": first["score_trend_degree"],
        "score_trend_degree_effective": first["score_trend_degree_effective"],
        "smooth_window": first["smooth_window"],
        "smooth_polyorder": first["smooth_polyorder"],
        "grid_points": first["grid_points"],
        "scope_summary": scope_summary,
        "n_detail_rows": len(rows),
        "avg_cumulative_explained_variance_ratio": nanmean(cum_var),
        "avg_abs_pc_pearson": avg_abs_pearson,
        "avg_abs_pc_spearman": nanmean(abs_spearman),
        "avg_pc_trend_r2": avg_trend_r2,
        "avg_rmse_uA": nanmean(rmse),
        "median_rmse_uA": nanmedian(rmse),
        "max_rmse_uA": nanmax(rmse),
        "avg_mae_uA": nanmean(mae),
        "avg_max_abs_residual_uA": nanmean(max_abs),
        "avg_normalized_rmse": avg_nrmse,
        "median_normalized_rmse": nanmedian(nrmse),
        "avg_r2": avg_r2,
        "median_r2": nanmedian(r2),
        "min_r2": nanmin(r2),
        "objective_score": objective_score,
    }


def csv_safe_row(row: dict[str, object], fieldnames: list[str]) -> dict[str, object]:
    safe = {}
    for fieldname in fieldnames:
        value = row.get(fieldname, "")
        if isinstance(value, (float, np.floating)):
            safe[fieldname] = format_float(float(value))
        else:
            safe[fieldname] = value
    return safe


def finite_sort_value(row: dict[str, object], fieldname: str) -> float:
    value = float(row.get(fieldname, math.nan))
    return value if np.isfinite(value) else -math.inf


def analyte_output_dir(output_root: Path, analyte: str) -> Path:
    return output_root / analyte.lower()


def run_analyte_sweep(
    analyte: str,
    entries: list[dict[str, object]],
    concentration_count: int,
    args: argparse.Namespace,
    output_root: Path,
) -> list[dict[str, object]]:
    output_dir = analyte_output_dir(output_root, analyte)
    output_dir.mkdir(parents=True, exist_ok=True)

    smooth_windows = parse_int_list(args.smooth_windows)
    smooth_polyorders = parse_int_list(args.smooth_polyorders)
    grid_points_values = parse_int_list(args.grid_points)
    n_component_values = [
        value
        for value in parse_int_list(args.n_components)
        if value <= max(1, concentration_count - 1)
    ]
    trend_degree_values = [
        value
        for value in parse_int_list(args.score_trend_degrees)
        if value <= max(1, concentration_count - 1)
    ]
    scopes = selected_scopes(args.scope)

    setup_parameters = []
    seen = set()
    for smooth_window, smooth_polyorder, grid_points, n_components, trend_degree in product(
        smooth_windows,
        smooth_polyorders,
        grid_points_values,
        n_component_values,
        trend_degree_values,
    ):
        effective_polyorder = 0 if smooth_window <= 1 else smooth_polyorder
        key = (smooth_window, effective_polyorder, grid_points, n_components, trend_degree)
        if key in seen:
            continue
        seen.add(key)
        setup_parameters.append(key)

    if args.limit_setups is not None:
        setup_parameters = setup_parameters[: args.limit_setups]
    if not setup_parameters:
        raise ValueError(f"No valid PCR setup parameters for {analyte}")

    detailed_path = output_dir / "pcr_detailed.csv"
    summary_path = output_dir / "pcr_summary.csv"
    best_path = output_dir / "best_setups.csv"
    summary_rows = []
    total_detail_rows = 0
    start_time = time.time()

    with detailed_path.open("w", newline="") as detailed_handle:
        detailed_writer = csv.DictWriter(detailed_handle, fieldnames=DETAIL_FIELDNAMES)
        detailed_writer.writeheader()
        for setup_id, (smooth_window, smooth_polyorder, grid_points, n_components, trend_degree) in enumerate(setup_parameters, 1):
            params = {
                "n_components": n_components,
                "score_trend_degree": trend_degree,
                "smooth_window": smooth_window,
                "smooth_polyorder": smooth_polyorder,
                "grid_points": grid_points,
            }
            models = fit_pcr_models(entries, params, scopes=scopes)
            detail_rows = build_detail_rows(analyte, setup_id, params, models)
            total_detail_rows += len(detail_rows)
            for row in detail_rows:
                detailed_writer.writerow(csv_safe_row(row, DETAIL_FIELDNAMES))

            scope_summaries = sorted(scopes)
            if len(scopes) > 1:
                scope_summaries.append("all")
            for scope_summary in scope_summaries:
                summary_row = build_summary_row(detail_rows, scope_summary)
                if summary_row:
                    summary_rows.append(summary_row)

            if setup_id % args.progress_every == 0 or setup_id == len(setup_parameters):
                elapsed = time.time() - start_time
                print(f"{analyte}: completed {setup_id}/{len(setup_parameters)} PCR setups in {elapsed:.1f}s")

    summary_rows = sorted(
        summary_rows,
        key=lambda row: (
            finite_sort_value(row, "objective_score"),
            finite_sort_value(row, "avg_r2"),
            finite_sort_value(row, "avg_pc_trend_r2"),
            finite_sort_value(row, "avg_abs_pc_pearson"),
            -float(row["avg_normalized_rmse"]) if np.isfinite(float(row["avg_normalized_rmse"])) else -math.inf,
        ),
        reverse=True,
    )

    with summary_path.open("w", newline="") as summary_handle:
        writer = csv.DictWriter(summary_handle, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(csv_safe_row(row, SUMMARY_FIELDNAMES))

    with best_path.open("w", newline="") as best_handle:
        writer = csv.DictWriter(best_handle, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        for row in summary_rows[: args.top_n]:
            writer.writerow(csv_safe_row(row, SUMMARY_FIELDNAMES))

    print(f"{analyte}: wrote detail rows: {detailed_path}")
    print(f"{analyte}: wrote ranked summaries: {summary_path}")
    print(f"{analyte}: wrote top {args.top_n} setups: {best_path}")
    print(f"{analyte}: detailed rows: {total_detail_rows}")
    return summary_rows


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = []
        for row in csv.DictReader(handle):
            cleaned = {
                key.strip(): (value.strip() if value is not None else "")
                for key, value in row.items()
            }
            if any(cleaned.values()):
                rows.append(cleaned)
        return rows


def select_best_setup(output_root: Path, analyte: str) -> dict[str, str]:
    best_path = analyte_output_dir(output_root, analyte) / "best_setups.csv"
    rows = read_csv_dicts(best_path)
    if not rows:
        raise ValueError(f"No best setup rows found for {analyte}: {best_path}")
    return rows[0]


def params_from_setup_row(row: dict[str, str]) -> dict[str, object]:
    return {
        "setup_id": int(float(row["setup_id"])),
        "n_components": int(float(row.get("requested_n_components") or row.get("n_components"))),
        "score_trend_degree": int(float(row["score_trend_degree"])),
        "smooth_window": int(float(row["smooth_window"])),
        "smooth_polyorder": int(float(row["smooth_polyorder"])),
        "grid_points": int(float(row["grid_points"])),
        "scope_summary": row["scope_summary"],
        "objective_score": float(row["objective_score"]),
        "avg_r2": float(row["avg_r2"]),
        "avg_normalized_rmse": float(row["avg_normalized_rmse"]),
    }


def write_json(path: Path, data: dict[str, object]) -> None:
    with path.open("w") as handle:
        json.dump(data, handle, indent=2, default=json_default)
        handle.write("\n")


def model_lookup(models: list[dict[str, object]]) -> dict[tuple[str, str, str], dict[str, object]]:
    return {
        (str(model["electrode"]), str(model["technique"]), str(model["sweep"])): model
        for model in models
    }


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


def plot_training_fit_pdf(
    analyte: str,
    mean_models: list[dict[str, object]],
    output_path: Path,
) -> None:
    lookup = model_lookup(mean_models)
    conditions = sorted({int(condition) for model in mean_models for condition in model["conditions"]})
    with PdfPages(output_path) as pdf:
        for condition in conditions:
            concentration_label = ""
            for model in mean_models:
                matches = np.where(model["conditions"] == condition)[0]
                if len(matches):
                    concentration_label = clean_value(float(model["concentrations"][int(matches[0])]))
                    break
            fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2))
            fig.suptitle(
                f"{analyte} PCR training fit | condition code {condition} | {concentration_label} uM",
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
                ax.plot(model["x_grid"], model["data_matrix"][:, index], color="#555555", lw=1.0, label="actual")
                ax.plot(model["x_grid"], model["predicted_matrix"][:, index], color=ANALYTES[analyte]["color"], lw=1.4, label="PCR predicted")
                ax.plot(model["x_grid"], model["residual_matrix"][:, index], color="#0072B2", lw=0.8, ls="--", label="residual")
                ax.legend(loc="best", fontsize=7)
            fig.tight_layout(rect=(0, 0.02, 1, 0.94))
            pdf.savefig(fig)
            plt.close(fig)


def write_equation_summary(
    prediction_models: dict[str, dict[str, object]],
    output_path: Path,
) -> None:
    fieldnames = [
        "analyte",
        "curve",
        "technique",
        "sweep",
        "n_components",
        "score_trend_degree_effective",
        "component",
        "explained_variance_ratio",
        "score_trend_coefficients_desc",
        "equation",
    ]
    equation = "current(V,c)=mean_curve(V)+sum_j(polyval(score_coefficients_j,c)*loading_j(V))"
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for analyte in ANALYTE_ORDER:
            model = prediction_models[analyte]
            for curve_key, curve in model["curves"].items():
                coefficients = np.asarray(curve["score_trend_coefficients"], dtype=float)
                explained = np.asarray(curve["explained_variance_ratio"], dtype=float)
                for index, component_coefficients in enumerate(coefficients):
                    writer.writerow(
                        {
                            "analyte": analyte,
                            "curve": curve_key,
                            "technique": curve["technique"],
                            "sweep": curve["sweep"],
                            "n_components": curve["n_components"],
                            "score_trend_degree_effective": curve["score_trend_degree_effective"],
                            "component": index + 1,
                            "explained_variance_ratio": format_float(float(explained[index])),
                            "score_trend_coefficients_desc": format_series(component_coefficients),
                            "equation": equation,
                        }
                    )


def build_best_models(
    series: dict[str, list[dict[str, object]]],
    entries_by_analyte: dict[str, list[dict[str, object]]],
    output_root: Path,
) -> tuple[dict[str, dict[str, object]], dict[str, list[dict[str, object]]], dict[str, dict[str, object]]]:
    prediction_models = {}
    mean_models_by_analyte = {}
    setup_params_by_analyte = {}
    for analyte in ANALYTE_ORDER:
        selected_row = select_best_setup(output_root, analyte)
        params = params_from_setup_row(selected_row)
        setup_params_by_analyte[analyte] = params
        all_models = fit_pcr_models(entries_by_analyte[analyte], params)
        mean_models = [model for model in all_models if model["scope"] == "mean"]
        mean_models_by_analyte[analyte] = mean_models
        prediction_models[analyte] = build_prediction_model(all_models, params, scope="mean")

        output_dir = analyte_output_dir(output_root, analyte)
        write_json(output_dir / "selected_setup_metadata.json", {"selected_row": selected_row, "params": params})
        write_json(output_dir / "mean_pcr_model.json", prediction_models[analyte])
        plot_training_fit_pdf(analyte, mean_models, output_dir / "mean_curve_pcr_training_fits.pdf")
        print(
            f"{analyte}: selected setup {params['setup_id']} | "
            f"PCs={params['n_components']} trend={params['score_trend_degree']} "
            f"smooth={params['smooth_window']}/{params['smooth_polyorder']} "
            f"grid={params['grid_points']} scope={params['scope_summary']}"
        )

    write_json(output_root / "all_analyte_pcr_models.json", prediction_models)
    write_json(output_root / "selected_setup_metadata.json", setup_params_by_analyte)
    write_equation_summary(prediction_models, output_root / "predictive_equation_summary.csv")
    return prediction_models, mean_models_by_analyte, setup_params_by_analyte


def concentration_ranges(series: dict[str, list[dict[str, object]]]) -> dict[str, tuple[float, float]]:
    ranges = {}
    for analyte, rows in series.items():
        values = [float(row["concentration_uM"]) for row in rows]
        ranges[analyte] = (float(min(values)), float(max(values)))
    return ranges


def default_initial_concentrations(series: dict[str, list[dict[str, object]]]) -> dict[str, float]:
    values = {}
    for analyte, rows in series.items():
        concentrations = sorted(float(row["concentration_uM"]) for row in rows)
        values[analyte] = float(np.median(concentrations))
    return values


def write_combined_prediction_csv(
    predicted_by_analyte: dict[str, dict[str, dict[str, object]]],
    concentrations: dict[str, float],
    output_path: Path,
) -> None:
    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "analyte",
            "requested_concentration_uM",
            "curve",
            "technique",
            "sweep",
            "potential_v",
            "current_uA",
        ])
        for analyte in ANALYTE_ORDER:
            for curve_key, curve in predicted_by_analyte[analyte].items():
                for potential_v, current_uA in zip(curve["potential_v"], curve["current_uA"]):
                    writer.writerow(
                        [
                            analyte,
                            concentrations[analyte],
                            curve_key,
                            curve["technique"],
                            curve["sweep"],
                            float(potential_v),
                            float(current_uA),
                        ]
                    )


def plot_combined_prediction(
    predicted_by_analyte: dict[str, dict[str, dict[str, object]]],
    concentrations: dict[str, float],
    mean_models_by_analyte: dict[str, list[dict[str, object]]],
    output_path: Path | None = None,
    figure: plt.Figure | None = None,
    axes: np.ndarray | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    if figure is None or axes is None:
        figure, axes = plt.subplots(3, 4, figsize=(17.0, 10.8))
    figure.suptitle("Isolated-analyte PCR predicted CV/CV-GC curves", fontweight="semibold")

    for row_index, analyte in enumerate(ANALYTE_ORDER):
        models_by_curve = {
            (model["technique"], model["sweep"]): model
            for model in mean_models_by_analyte[analyte]
        }
        for col_index, (technique, sweep) in enumerate(PANEL_ORDER):
            ax = axes[row_index, col_index]
            ax.cla()
            style_curve_axis(ax)
            if row_index == 0:
                ax.set_title(f"{TECHNIQUES[technique]['label']}\n{SWEEPS[sweep]}")
            else:
                ax.set_title(SWEEPS[sweep])
            if col_index == 0:
                ax.set_ylabel(f"{analyte}\nCurrent ({CURRENT_UNIT})")

            training_model = models_by_curve.get((technique, sweep))
            if training_model is not None:
                for index in range(training_model["data_matrix"].shape[1]):
                    label = None
                    if index == 0:
                        label = "training curves"
                    ax.plot(
                        training_model["x_grid"],
                        training_model["data_matrix"][:, index],
                        color="#C8C8C8",
                        lw=0.5,
                        alpha=0.55,
                        label=label,
                    )

            curve_key = f"{technique}_{sweep}"
            curve = predicted_by_analyte[analyte][curve_key]
            ax.plot(
                curve["potential_v"],
                curve["current_uA"],
                color=ANALYTES[analyte]["color"],
                lw=1.7,
                label=f"{analyte} {clean_value(concentrations[analyte])} uM",
            )
            ax.legend(loc="best", fontsize=6)

    if output_path is not None:
        figure.tight_layout(rect=(0, 0.02, 1, 0.95))
    else:
        figure.subplots_adjust(
            left=0.055,
            right=0.99,
            top=0.90,
            bottom=0.15,
            hspace=0.45,
            wspace=0.25,
        )
    if output_path is not None:
        figure.savefig(output_path)
    return figure, axes


def save_prediction_outputs(
    prediction_models: dict[str, dict[str, object]],
    mean_models_by_analyte: dict[str, list[dict[str, object]]],
    concentrations: dict[str, float],
    output_root: Path,
    prediction_points: int,
) -> tuple[Path, Path]:
    prediction_dir = output_root / "interactive_predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    label = "_".join(f"{analyte}{slug_float(concentrations[analyte])}uM" for analyte in ANALYTE_ORDER)
    csv_path = prediction_dir / f"predicted_curves_{label}.csv"
    pdf_path = prediction_dir / f"predicted_curves_{label}.pdf"
    predicted = {
        analyte: predict_curves(prediction_models[analyte], concentrations[analyte], points=prediction_points)
        for analyte in ANALYTE_ORDER
    }
    write_combined_prediction_csv(predicted, concentrations, csv_path)
    fig, _axes = plot_combined_prediction(predicted, concentrations, mean_models_by_analyte, pdf_path)
    plt.close(fig)
    return csv_path, pdf_path


def parse_concentrations_from_textboxes(
    textboxes: dict[str, TextBox],
    ranges: dict[str, tuple[float, float]],
) -> tuple[dict[str, float] | None, str]:
    concentrations = {}
    for analyte in ANALYTE_ORDER:
        try:
            value = float(textboxes[analyte].text)
        except ValueError:
            return None, f"Invalid {analyte} concentration: {textboxes[analyte].text}"
        lower, upper = ranges[analyte]
        if value < lower or value > upper:
            return None, f"{analyte} must be within {clean_value(lower)}-{clean_value(upper)} uM"
        concentrations[analyte] = value
    return concentrations, ""


def launch_interactive_viewer(
    prediction_models: dict[str, dict[str, object]],
    mean_models_by_analyte: dict[str, list[dict[str, object]]],
    ranges: dict[str, tuple[float, float]],
    initial_concentrations: dict[str, float],
    output_root: Path,
    prediction_points: int,
    no_show: bool,
) -> None:
    fig, axes = plt.subplots(3, 4, figsize=(17.0, 11.2))
    plt.subplots_adjust(left=0.055, right=0.99, top=0.90, bottom=0.15, hspace=0.45, wspace=0.25)

    textboxes = {}
    left_positions = [0.12, 0.33, 0.54]
    for analyte, left in zip(ANALYTE_ORDER, left_positions):
        lower, upper = ranges[analyte]
        box_ax = fig.add_axes([left, 0.055, 0.12, 0.04])
        label = f"{analyte} uM ({clean_value(lower)}-{clean_value(upper)})"
        textboxes[analyte] = TextBox(box_ax, label, initial=str(clean_value(initial_concentrations[analyte])))

    save_ax = fig.add_axes([0.72, 0.055, 0.14, 0.04])
    status_ax = fig.add_axes([0.865, 0.035, 0.13, 0.08])
    status_ax.axis("off")
    save_button = Button(save_ax, "Generate + Save", color="#EEEEEE", hovercolor="#CCDDFF")
    status_text = status_ax.text(0, 0.7, "", fontsize=7, va="center")

    def draw(concentrations: dict[str, float], save: bool) -> None:
        predicted = {
            analyte: predict_curves(prediction_models[analyte], concentrations[analyte], points=prediction_points)
            for analyte in ANALYTE_ORDER
        }
        plot_combined_prediction(
            predicted,
            concentrations,
            mean_models_by_analyte,
            figure=fig,
            axes=axes,
        )
        if save:
            csv_path, pdf_path = save_prediction_outputs(
                prediction_models,
                mean_models_by_analyte,
                concentrations,
                output_root,
                prediction_points,
            )
            status_text.set_text(f"Saved:\n{csv_path.name}\n{pdf_path.name}")
            print(f"Saved prediction CSV: {csv_path}")
            print(f"Saved prediction PDF: {pdf_path}")
        else:
            status_text.set_text("Enter values, then Generate + Save.")
        fig.canvas.draw_idle()

    def parse_and_draw(save: bool) -> None:
        concentrations, error = parse_concentrations_from_textboxes(textboxes, ranges)
        if concentrations is None:
            status_text.set_text(error)
            fig.canvas.draw_idle()
            return
        draw(concentrations, save=save)

    for textbox in textboxes.values():
        textbox.on_submit(lambda _text: parse_and_draw(save=True))
    save_button.on_clicked(lambda _event: parse_and_draw(save=True))
    draw(initial_concentrations, save=False)

    if no_show:
        plt.close(fig)
        return
    plt.show()


def initial_concentrations_from_args(
    args: argparse.Namespace,
    series: dict[str, list[dict[str, object]]],
) -> dict[str, float]:
    concentrations = default_initial_concentrations(series)
    if args.da is not None:
        concentrations["DA"] = args.da
    if args.aa is not None:
        concentrations["AA"] = args.aa
    if args.ua is not None:
        concentrations["UA"] = args.ua
    return concentrations


def run(args: argparse.Namespace) -> None:
    configure_plot_style()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    series = scan_isolated_conditions()
    write_isolated_manifest(series, output_root / "isolated_analyte_conditions.csv")
    print("Isolated calibration sets:")
    for analyte in ANALYTE_ORDER:
        concentrations = [clean_value(row["concentration_uM"]) for row in series[analyte]]
        print(f"  {analyte}: {', '.join(concentrations)} uM")

    entries_by_analyte = {}
    for analyte in ANALYTE_ORDER:
        entries = load_trace_entries_for_series(series[analyte])
        if not entries:
            raise RuntimeError(f"No trace entries loaded for {analyte}")
        entries_by_analyte[analyte] = entries
        print(f"{analyte}: loaded {len(entries)} trace entries")

    for analyte in ANALYTE_ORDER:
        best_path = analyte_output_dir(output_root, analyte) / "best_setups.csv"
        should_sweep = args.force_sweep or (not args.skip_sweep and not best_path.exists())
        if should_sweep:
            run_analyte_sweep(
                analyte,
                entries_by_analyte[analyte],
                len(series[analyte]),
                args,
                output_root,
            )
        else:
            print(f"{analyte}: using existing sweep results at {best_path}")

    prediction_models, mean_models_by_analyte, _setup_params_by_analyte = build_best_models(
        series,
        entries_by_analyte,
        output_root,
    )

    ranges = concentration_ranges(series)
    initial_concentrations = initial_concentrations_from_args(args, series)
    csv_path, pdf_path = save_prediction_outputs(
        prediction_models,
        mean_models_by_analyte,
        initial_concentrations,
        output_root,
        args.prediction_points,
    )
    print(f"Wrote initial combined prediction CSV: {csv_path}")
    print(f"Wrote initial combined prediction PDF: {pdf_path}")

    launch_interactive_viewer(
        prediction_models,
        mean_models_by_analyte,
        ranges,
        initial_concentrations,
        output_root,
        args.prediction_points,
        args.no_show,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run isolated-analyte PCR sweeps for DA/AA/UA and launch a combined predictor GUI."
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--force-sweep", action="store_true", help="Re-run all DA/AA/UA PCR sweeps.")
    parser.add_argument("--skip-sweep", action="store_true", help="Use existing best_setups.csv files; fail if missing.")
    parser.add_argument(
        "--smooth-windows",
        default="0,31,51,101,151",
        help='Smoothing windows to sweep. Use 0 to disable. Example: "0,51,101".',
    )
    parser.add_argument(
        "--smooth-polyorders",
        default=f"2,{DEFAULT_SMOOTH_POLYORDER}",
        help='Savitzky-Golay smoothing polyorders to sweep. Example: "2,3".',
    )
    parser.add_argument(
        "--grid-points",
        default=f"{DEFAULT_GRID_POINTS},250,500",
        help='Aligned voltage grid point counts. Use 0 for native common grid. Example: "0,250,500".',
    )
    parser.add_argument(
        "--n-components",
        default=f"1-{DEFAULT_N_COMPONENTS + 2}",
        help='PCA component counts to sweep. Values above the analyte series limit are skipped. Example: "1-5".',
    )
    parser.add_argument(
        "--score-trend-degrees",
        default=f"1,{DEFAULT_SCORE_TREND_DEGREE},3",
        help='Polynomial degrees for PC-score interpolation. Example: "1,2,3".',
    )
    parser.add_argument(
        "--scope",
        choices=["mean", "electrode", "both"],
        default="both",
        help="Evaluate mean curves, electrode curves, or both. Default: both.",
    )
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--limit-setups", type=int, default=None)
    parser.add_argument("--prediction-points", type=int, default=500)
    parser.add_argument("--da", type=float, default=None, help="Initial DA concentration for the GUI/prediction.")
    parser.add_argument("--aa", type=float, default=None, help="Initial AA concentration for the GUI/prediction.")
    parser.add_argument("--ua", type=float, default=None, help="Initial UA concentration for the GUI/prediction.")
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
