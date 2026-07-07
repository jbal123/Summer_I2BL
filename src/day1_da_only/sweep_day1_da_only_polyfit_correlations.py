#!/usr/bin/env python3
"""Sweep CLS preprocessing settings and save calibration diagnostics.

CLS has no curve degree or coefficient interpolation stage. The sweep now
varies preprocessing choices that affect the signal matrix D before the closed
form solve:

    D = S.T @ C
    S = (C @ C.T)^-1 @ C @ D.T

The ranked setup CSV is still produced so downstream setup analysis can use a
stable setup_id even when preprocessing choices change.
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
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT / "xdg_cache"))

from day1_da_only_cls_core import (
    DEFAULT_GRID_POINTS,
    DEFAULT_MANIFEST,
    DEFAULT_MODEL_ORDER,
    DEFAULT_SMOOTH_POLYORDER,
    DEFAULT_SMOOTH_WINDOW,
    DEFAULT_SUBSET_DIR,
    fit_cls_models,
    format_float,
    format_series,
    load_trace_entries,
    normalize_model_order,
    parse_int_list,
    read_manifest,
)


DEFAULT_OUTPUT_DIR = DEFAULT_SUBSET_DIR / "cls_sweep"
QUADRATIC_OUTPUT_DIR = DEFAULT_SUBSET_DIR / "cls_quadratic_sweep"
CUBIC_OUTPUT_DIR = DEFAULT_SUBSET_DIR / "cls_cubic_sweep"


def default_output_dir(model_order: str) -> Path:
    order = normalize_model_order(model_order)
    if order == "cubic":
        return CUBIC_OUTPUT_DIR
    if order == "quadratic":
        return QUADRATIC_OUTPUT_DIR
    return DEFAULT_OUTPUT_DIR

DETAILED_FIELDNAMES = [
    "setup_id",
    "method",
    "model_order",
    "smooth_window",
    "smooth_polyorder",
    "grid_points",
    "scope",
    "technique",
    "sweep",
    "electrode",
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
]

SUMMARY_FIELDNAMES = [
    "setup_id",
    "method",
    "model_order",
    "smooth_window",
    "smooth_polyorder",
    "grid_points",
    "scope_summary",
    "n_detail_rows",
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
    setup_id: int,
    params: dict[str, object],
    models: list[dict[str, object]],
) -> list[dict[str, object]]:
    rows = []
    for model in models:
        rows.append(
            {
                "setup_id": setup_id,
                "method": "cls",
                "model_order": model["model_order"],
                "smooth_window": params["smooth_window"],
                "smooth_polyorder": model["smooth_polyorder_effective"],
                "grid_points": params["grid_points"],
                "scope": model["scope"],
                "technique": model["technique"],
                "sweep": model["sweep"],
                "electrode": model["electrode"],
                "n_concentrations": model["n_concentrations"],
                "n_voltages": model["n_voltages"],
                "rmse_uA": model["rmse_uA"],
                "mae_uA": model["mae_uA"],
                "max_abs_residual_uA": model["max_abs_residual_uA"],
                "normalized_rmse": model["normalized_rmse"],
                "r2": model["r2"],
                "conditions": format_series(model["conditions"]),
                "da_values": format_series(model["concentrations"]),
                "per_condition_rmse_uA": format_series(model["per_condition_rmse_uA"]),
                "per_condition_mae_uA": format_series(model["per_condition_mae_uA"]),
                "per_condition_r2": format_series(model["per_condition_r2"]),
            }
        )
    return rows


def build_summary_row(detail_rows: list[dict[str, object]], scope_summary: str) -> dict[str, object]:
    if not detail_rows:
        raise ValueError("Cannot summarize empty CLS detail rows")
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
    avg_r2 = nanmean(r2)
    avg_nrmse = nanmean(nrmse)

    # High R2 and low normalized residual are both valuable. This keeps the
    # ranking scale intuitive while penalizing structured linearity failures.
    objective_score = avg_r2 - avg_nrmse if np.isfinite(avg_r2) and np.isfinite(avg_nrmse) else math.nan

    return {
        "setup_id": first["setup_id"],
        "method": "cls",
        "model_order": first["model_order"],
        "smooth_window": first["smooth_window"],
        "smooth_polyorder": first["smooth_polyorder"],
        "grid_points": first["grid_points"],
        "scope_summary": scope_summary,
        "n_detail_rows": len(rows),
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


def run_sweep(args: argparse.Namespace) -> None:
    start_time = time.time()
    model_order = normalize_model_order(args.model_order)
    output_dir = (args.output_dir or default_output_dir(model_order)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    smooth_windows = parse_int_list(args.smooth_windows)
    smooth_polyorders = parse_int_list(args.smooth_polyorders)
    grid_points_values = parse_int_list(args.grid_points)
    scopes = selected_scopes(args.scope)

    conditions = read_manifest(args.manifest.resolve())
    entries = load_trace_entries(conditions, args.subset_dir.resolve())
    if not entries:
        raise RuntimeError("No trace entries loaded")

    setup_parameters = []
    seen = set()
    for smooth_window, smooth_polyorder, grid_points in product(
        smooth_windows,
        smooth_polyorders,
        grid_points_values,
    ):
        effective_polyorder = 0 if smooth_window <= 1 else smooth_polyorder
        key = (smooth_window, effective_polyorder, grid_points)
        if key in seen:
            continue
        seen.add(key)
        setup_parameters.append(key)

    if args.limit_setups is not None:
        setup_parameters = setup_parameters[: args.limit_setups]
    if not setup_parameters:
        raise ValueError("No valid CLS setup parameters to run")

    detailed_path = output_dir / "cls_detailed.csv"
    summary_path = output_dir / "cls_summary.csv"
    best_path = output_dir / "best_setups.csv"

    summary_rows = []
    total_detail_rows = 0
    setup_id = 0

    with detailed_path.open("w", newline="") as detailed_handle:
        detailed_writer = csv.DictWriter(detailed_handle, fieldnames=DETAILED_FIELDNAMES)
        detailed_writer.writeheader()

        for smooth_window, smooth_polyorder, grid_points in setup_parameters:
            setup_id += 1
            params = {
                "model_order": model_order,
                "smooth_window": smooth_window,
                "smooth_polyorder": smooth_polyorder,
                "grid_points": grid_points,
            }
            models = fit_cls_models(entries, params, scopes=scopes)
            detail_rows = build_detail_rows(setup_id, params, models)
            total_detail_rows += len(detail_rows)

            for row in detail_rows:
                detailed_writer.writerow(csv_safe_row(row, DETAILED_FIELDNAMES))

            scope_summaries = sorted(scopes)
            if len(scopes) > 1:
                scope_summaries.append("all")
            for scope_summary in scope_summaries:
                summary_row = build_summary_row(detail_rows, scope_summary)
                if summary_row:
                    summary_rows.append(summary_row)

            if setup_id % args.progress_every == 0 or setup_id == len(setup_parameters):
                elapsed = time.time() - start_time
                print(f"Completed {setup_id}/{len(setup_parameters)} CLS setups in {elapsed:.1f}s")

    summary_rows = sorted(
        summary_rows,
        key=lambda row: (
            finite_sort_value(row, "objective_score"),
            finite_sort_value(row, "avg_r2"),
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

    print(f"Wrote CLS detail rows: {detailed_path}")
    print(f"Wrote CLS ranked summaries: {summary_path}")
    print(f"Wrote top {args.top_n} CLS setups: {best_path}")
    print(f"Trace entries loaded: {len(entries)}")
    print(f"CLS setups run: {len(setup_parameters)}")
    print(f"Detailed rows: {total_detail_rows}")
    print(f"Elapsed: {time.time() - start_time:.1f}s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep CLS preprocessing settings and save diagnostics.")
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
        "--scope",
        choices=["mean", "electrode", "both"],
        default="both",
        help="Evaluate mean curves, electrode curves, or both. Default: both.",
    )
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--progress-every", type=int, default=5)
    parser.add_argument("--limit-setups", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    run_sweep(parse_args())


if __name__ == "__main__":
    main()
