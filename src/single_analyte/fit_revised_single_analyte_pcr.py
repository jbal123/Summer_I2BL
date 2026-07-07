#!/usr/bin/env python3
"""Fit PCR calibration curves for the revised single-analyte CV/CV-GC data."""

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
import re
import tempfile
import time
from dataclasses import dataclass
from itertools import product
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

from day1_da_only_cls_core import CURRENT_SCALE, CURRENT_UNIT, ROOT, clean_value
from day1_da_only_pcr_core import (
    DEFAULT_N_COMPONENTS,
    DEFAULT_SCORE_TREND_DEGREE,
    DEFAULT_SMOOTH_POLYORDER,
    DEFAULT_SMOOTH_WINDOW,
    fit_pcr_models,
    format_float,
    format_series,
    json_default,
    parse_int_list,
)


DEFAULT_DATA_ROOT = ROOT / "Single_Analyte_Data_revised"
DEFAULT_OUTPUT_DIR = DEFAULT_DATA_ROOT / "revised_single_analyte_pcr_analysis"

NUMBER_RE = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")
COND_RE = re.compile(r"Cond(\d+)")
DA_RE = re.compile(r"DA(\d+(?:p\d+)?)")
AA_RE = re.compile(r"AA(\d+(?:p\d+)?)")
UA_RE = re.compile(r"UA(\d+(?:p\d+)?)")

ANALYTE_ORDER = ("DA", "AA", "UA")
ANALYTE_LABELS = {
    "DA": "Dopamine",
    "AA": "Ascorbic Acid",
    "UA": "Uric Acid",
}
ANALYTE_COLORS = {
    "DA": "#D55E00",
    "AA": "#0072B2",
    "UA": "#009E73",
}
ANALYTE_VALUE_RE = {
    "DA": DA_RE,
    "AA": AA_RE,
    "UA": UA_RE,
}

TECHNIQUE_CONFIGS = {
    "cv_normal": {
        "label": "CV normal",
        "filename_pattern": "cv_norm",
        "electrode_columns": {f"E{i}": i for i in range(1, 9)},
    },
    "cv_gc": {
        "label": "CV-GC generator",
        "filename_pattern": "cv_gc",
        "electrode_columns": {
            "E1 generator": 1,
            "E3 generator": 3,
            "E5 generator": 5,
            "E7 generator": 7,
        },
    },
}

PANEL_ORDER = (
    ("cv_normal", "anodic"),
    ("cv_normal", "cathodic"),
    ("cv_gc", "anodic"),
    ("cv_gc", "cathodic"),
)
SWEEP_LABELS = {
    "anodic": "Anodic (increasing V)",
    "cathodic": "Cathodic (decreasing V)",
}

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


@dataclass(frozen=True)
class CVFile:
    analyte: str
    folder: str
    condition: int
    da_uM: float | None
    aa_uM: float | None
    ua_uM: float | None
    concentration_uM: float
    technique: str
    path: Path


@dataclass(frozen=True)
class SweepSegment:
    start: int
    end: int
    sign: int


@dataclass(frozen=True)
class CycleSelection:
    matrix: np.ndarray
    start_row: int
    end_row: int
    selected_cycle: int
    total_complete_cycles: int


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "legend.frameon": False,
            "lines.linewidth": 1.1,
        }
    )


def parse_value(pattern: re.Pattern[str], text: str) -> float | None:
    match = pattern.search(text)
    if match is None:
        return None
    return float(match.group(1).replace("p", "."))


def parse_condition(path: Path) -> int | None:
    match = COND_RE.search(path.name)
    if match is None:
        return None
    return int(match.group(1))


def technique_for(path: Path) -> str | None:
    name = path.name.lower()
    if not name.endswith(".txt"):
        return None
    if "cv_norm" in name:
        return "cv_normal"
    if "cv_gc" in name:
        return "cv_gc"
    return None


def is_numeric_folder(name: str) -> bool:
    try:
        float(name)
    except ValueError:
        return False
    return True


def is_target_analyte_file(analyte: str, da: float | None, aa: float | None, ua: float | None) -> bool:
    da_v = 0.0 if da is None else da
    aa_v = 0.0 if aa is None else aa
    ua_v = 0.0 if ua is None else ua
    if analyte == "DA":
        return da is not None and da_v >= 0.0 and aa_v == 0.0 and ua_v == 0.0
    if analyte == "AA":
        return da_v == 0.0 and aa is not None and aa_v > 0.0 and ua_v == 0.0
    if analyte == "UA":
        return da_v == 0.0 and aa_v == 0.0 and ua is not None and ua_v >= 0.0
    return False


def current_polarity_multiplier(analyte: str) -> float:
    return -1.0 if analyte == "AA" else 1.0


def current_polarity_label(analyte: str) -> str:
    if current_polarity_multiplier(analyte) < 0:
        return "displayed/fitted current = -raw current (AA polarity normalized to legacy CV/CV-GC convention)"
    return "displayed/fitted current = raw current"


def discover_cv_files(data_root: Path, analyte: str) -> tuple[list[CVFile], list[Path]]:
    selected: list[CVFile] = []
    skipped: list[Path] = []
    analyte_root = data_root / analyte
    if not analyte_root.is_dir():
        return selected, skipped

    folders = sorted(
        (path for path in analyte_root.iterdir() if path.is_dir() and is_numeric_folder(path.name)),
        key=lambda path: float(path.name),
    )
    for folder in folders:
        for path in sorted(folder.glob("CV*.txt")):
            technique = technique_for(path)
            if technique is None:
                skipped.append(path)
                continue
            condition = parse_condition(path)
            da = parse_value(DA_RE, path.name)
            aa = parse_value(AA_RE, path.name)
            ua = parse_value(UA_RE, path.name)
            if condition is None or not is_target_analyte_file(analyte, da, aa, ua):
                skipped.append(path)
                continue
            if analyte == "AA" and condition > 10:
                skipped.append(path)
                continue
            concentration = parse_value(ANALYTE_VALUE_RE[analyte], path.name)
            if concentration is None:
                skipped.append(path)
                continue
            selected.append(
                CVFile(
                    analyte=analyte,
                    folder=folder.name,
                    condition=condition,
                    da_uM=da,
                    aa_uM=aa,
                    ua_uM=ua,
                    concentration_uM=concentration,
                    technique=technique,
                    path=path,
                )
            )

    selected.sort(key=lambda item: (item.condition, item.technique, item.path.name))
    return selected, skipped


def load_numeric_matrix(path: Path) -> np.ndarray:
    rows: list[list[float]] = []
    with path.open("r", errors="ignore") as handle:
        for line in handle:
            values = [float(value) for value in NUMBER_RE.findall(line)]
            if len(values) >= 2:
                rows.append(values)
    if not rows:
        raise ValueError(f"No numeric rows in {path}")
    width = max(len(row) for row in rows)
    matrix = np.full((len(rows), width), np.nan, dtype=float)
    for index, row in enumerate(rows):
        matrix[index, : len(row)] = row
    return matrix


def sweep_segments(potential: np.ndarray) -> list[SweepSegment]:
    deltas = np.diff(potential)
    nonzero = np.flatnonzero(np.abs(deltas) > 1e-12)
    if len(nonzero) == 0:
        return []

    segments: list[SweepSegment] = []
    start = 0
    previous_sign = 1 if deltas[nonzero[0]] > 0 else -1
    for idx in nonzero[1:]:
        sign = 1 if deltas[idx] > 0 else -1
        if sign != previous_sign:
            segments.append(SweepSegment(start=start, end=idx + 1, sign=previous_sign))
            start = idx
            previous_sign = sign
    segments.append(SweepSegment(start=start, end=len(potential), sign=previous_sign))
    return segments


def select_last_complete_cycle(matrix: np.ndarray) -> CycleSelection:
    segments = sweep_segments(matrix[:, 0])
    complete_cycles = len(segments) // 2
    if complete_cycles == 0:
        return CycleSelection(matrix=matrix, start_row=0, end_row=len(matrix), selected_cycle=1, total_complete_cycles=1)

    first_segment = 2 * (complete_cycles - 1)
    start = segments[first_segment].start
    end = segments[first_segment + 1].end
    return CycleSelection(
        matrix=matrix[start:end],
        start_row=start,
        end_row=end,
        selected_cycle=complete_cycles,
        total_complete_cycles=complete_cycles,
    )


def split_sweep_indices(potential: np.ndarray) -> dict[str, np.ndarray]:
    segments = sweep_segments(potential)
    if not segments:
        return {"anodic": np.arange(len(potential)), "cathodic": np.array([], dtype=int)}
    if len(segments) == 1:
        indices = np.arange(segments[0].start, segments[0].end)
        if segments[0].sign > 0:
            return {"anodic": indices, "cathodic": np.array([], dtype=int)}
        return {"anodic": np.array([], dtype=int), "cathodic": indices}

    first = np.arange(segments[0].start, segments[0].end)
    second = np.arange(segments[1].start, segments[1].end)
    if segments[0].sign > 0:
        return {"anodic": first, "cathodic": second}
    return {"anodic": second, "cathodic": first}


def sorted_trace(x_values: np.ndarray, y_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(x_values) & np.isfinite(y_values)
    x = x_values[mask]
    y = y_values[mask]
    order = np.argsort(x)
    return x[order], y[order]


def cycle_label(selection: CycleSelection) -> str:
    return (
        f"last complete run {selection.selected_cycle}/{selection.total_complete_cycles}; "
        f"source rows {selection.start_row + 1}-{selection.end_row}"
    )


def load_trace_entries(data_root: Path, analyte: str) -> tuple[list[dict[str, object]], list[dict[str, object]], list[Path]]:
    files, skipped = discover_cv_files(data_root, analyte)
    entries: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []
    file_metadata: dict[tuple[int, str], dict[str, object]] = {}
    polarity = current_polarity_multiplier(analyte)

    for item in files:
        matrix = load_numeric_matrix(item.path)
        selection = select_last_complete_cycle(matrix)
        cycle_matrix = selection.matrix
        potential = cycle_matrix[:, 0]
        sweeps = split_sweep_indices(potential)
        columns = TECHNIQUE_CONFIGS[item.technique]["electrode_columns"]
        relative_source = str(item.path.relative_to(ROOT))

        file_metadata[(item.condition, item.technique)] = {
            "condition": item.condition,
            "folder": item.folder,
            "concentration_uM": item.concentration_uM,
            "technique": item.technique,
            "source_file": relative_source,
            "cycle_selection": cycle_label(selection),
            "current_polarity": current_polarity_label(analyte),
        }

        for sweep, indices in sweeps.items():
            if len(indices) < 10:
                continue
            for electrode, column_index in columns.items():
                if column_index >= cycle_matrix.shape[1]:
                    continue
                x_values, y_values = sorted_trace(
                    potential[indices],
                    cycle_matrix[indices, column_index] * CURRENT_SCALE * polarity,
                )
                if len(x_values) <= 10:
                    continue
                entries.append(
                    {
                        "scope": "electrode",
                        "condition": item.condition,
                        "condition_label": f"Cond{item.condition}",
                        "folder": item.folder,
                        "da_uM": item.concentration_uM,
                        "analyte": analyte,
                        "technique": item.technique,
                        "sweep": sweep,
                        "electrode": electrode,
                        "x": x_values,
                        "y": y_values,
                        "source_file": relative_source,
                        "cycle_selection": cycle_label(selection),
                        "current_polarity": current_polarity_label(analyte),
                    }
                )

    condition_map: dict[int, dict[str, object]] = {}
    for metadata in file_metadata.values():
        row = condition_map.setdefault(
            int(metadata["condition"]),
            {
                "condition": metadata["condition"],
                "folder": metadata["folder"],
                "concentration_uM": metadata["concentration_uM"],
                "cv_normal_file": "",
                "cv_gc_file": "",
                "cycle_selection": metadata["cycle_selection"],
                "current_polarity": metadata["current_polarity"],
            },
        )
        if metadata["technique"] == "cv_normal":
            row["cv_normal_file"] = metadata["source_file"]
        elif metadata["technique"] == "cv_gc":
            row["cv_gc_file"] = metadata["source_file"]

    for condition in sorted(condition_map):
        manifest_rows.append(condition_map[condition])

    return entries, manifest_rows, skipped


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


def build_detail_rows(analyte: str, setup_id: int, params: dict[str, object], models: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for model in models:
        rows.append(
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
    return rows


def build_summary_row(analyte: str, setup_id: int, params: dict[str, object], detail_rows: list[dict[str, object]]) -> dict[str, object]:
    rmse = [float(row["rmse_uA"]) for row in detail_rows]
    mae = [float(row["mae_uA"]) for row in detail_rows]
    max_abs = [float(row["max_abs_residual_uA"]) for row in detail_rows]
    nrmse = [float(row["normalized_rmse"]) for row in detail_rows]
    r2 = [float(row["r2"]) for row in detail_rows]
    cum_var = [float(row["cumulative_explained_variance_ratio"]) for row in detail_rows]
    abs_pearson = [float(row["avg_abs_pc_pearson"]) for row in detail_rows]
    abs_spearman = [float(row["avg_abs_pc_spearman"]) for row in detail_rows]
    trend_r2 = [float(row["avg_pc_trend_r2"]) for row in detail_rows]

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
        "analyte": analyte,
        "setup_id": setup_id,
        "method": "pcr",
        "n_components": params["n_components"],
        "requested_n_components": params["n_components"],
        "score_trend_degree": params["score_trend_degree"],
        "score_trend_degree_effective": detail_rows[0]["score_trend_degree_effective"],
        "smooth_window": params["smooth_window"],
        "smooth_polyorder": detail_rows[0]["smooth_polyorder"],
        "grid_points": params["grid_points"],
        "scope_summary": "electrode",
        "n_detail_rows": len(detail_rows),
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


def parameter_grid(args: argparse.Namespace, concentration_count: int) -> list[dict[str, int]]:
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
        setup_parameters.append(
            {
                "smooth_window": smooth_window,
                "smooth_polyorder": smooth_polyorder,
                "grid_points": grid_points,
                "n_components": n_components,
                "score_trend_degree": trend_degree,
            }
        )
    if args.limit_setups is not None:
        setup_parameters = setup_parameters[: args.limit_setups]
    return setup_parameters


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(csv_safe_row(row, fieldnames))


def write_manifest(path: Path, analyte: str, manifest_rows: list[dict[str, object]], skipped: list[Path]) -> None:
    fieldnames = [
        "analyte",
        "condition",
        "folder",
        "concentration_uM",
        "cv_normal_file",
        "cv_gc_file",
        "cycle_selection",
        "current_polarity",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in manifest_rows:
            writer.writerow({"analyte": analyte, **row})
    skipped_path = path.with_name(path.stem + "_skipped_files.txt")
    with skipped_path.open("w") as handle:
        for skipped_path_item in skipped:
            handle.write(str(skipped_path_item.relative_to(ROOT)) + "\n")


def fit_analyte_sweep(
    analyte: str,
    entries: list[dict[str, object]],
    concentration_count: int,
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    setups = parameter_grid(args, concentration_count)
    if not setups:
        raise ValueError(f"No PCR setup parameters available for {analyte}")

    detailed_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    start = time.time()
    for setup_id, params in enumerate(setups, 1):
        models = fit_pcr_models(entries, params, scopes={"electrode"})
        detail_rows = build_detail_rows(analyte, setup_id, params, models)
        detailed_rows.extend(detail_rows)
        summary_rows.append(build_summary_row(analyte, setup_id, params, detail_rows))
        if setup_id % args.progress_every == 0 or setup_id == len(setups):
            print(f"{analyte}: completed {setup_id}/{len(setups)} PCR setups in {time.time() - start:.1f}s")

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
    best = summary_rows[0]

    write_csv(output_dir / analyte.lower() / "pcr_detailed.csv", detailed_rows, DETAIL_FIELDNAMES)
    write_csv(output_dir / analyte.lower() / "pcr_summary.csv", summary_rows, SUMMARY_FIELDNAMES)
    write_csv(output_dir / analyte.lower() / "best_setups.csv", summary_rows[: args.top_n], SUMMARY_FIELDNAMES)
    return detailed_rows, summary_rows, best


def params_from_summary(row: dict[str, object]) -> dict[str, int]:
    return {
        "n_components": int(float(row["requested_n_components"])),
        "score_trend_degree": int(float(row["score_trend_degree"])),
        "smooth_window": int(float(row["smooth_window"])),
        "smooth_polyorder": int(float(row["smooth_polyorder"])),
        "grid_points": int(float(row["grid_points"])),
    }


def style_curve_axis(ax: plt.Axes, show_xlabel: bool = True, show_ylabel: bool = True) -> None:
    if show_xlabel:
        ax.set_xlabel("Potential (V)")
    if show_ylabel:
        ax.set_ylabel(f"Current ({CURRENT_UNIT})")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", direction="in")
    ax.tick_params(axis="both", which="minor", direction="in")
    ax.minorticks_on()
    formatter = mticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-3, 3))
    ax.yaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=11))
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))


def style_score_axis(ax: plt.Axes) -> None:
    ax.set_xlabel("Concentration (uM)")
    ax.set_ylabel("PCA score")
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


def sorted_electrodes(models: list[dict[str, object]]) -> list[str]:
    def key(electrode: str) -> tuple[int, str]:
        match = re.search(r"E(\d+)", electrode)
        if match:
            return int(match.group(1)), electrode
        return 999, electrode

    return sorted({str(model["electrode"]) for model in models}, key=key)


def model_title(model: dict[str, object]) -> str:
    return f"{TECHNIQUE_CONFIGS[str(model['technique'])]['label']} | {SWEEP_LABELS[str(model['sweep'])]} | {model['electrode']}"


def add_metric_box(ax: plt.Axes, model: dict[str, object]) -> None:
    ax.text(
        0.02,
        0.98,
        (
            f"R2={float(model['r2']):.3f}\n"
            f"RMSE={float(model['rmse_uA']):.3g} {CURRENT_UNIT}\n"
            f"nRMSE={float(model['normalized_rmse']):.3g}\n"
            f"PC trend R2={float(model['avg_pc_trend_r2']):.3f}\n"
            f"cum var={float(model['cumulative_explained_variance_ratio']):.3f}"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.5,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78},
    )


def write_intro_page(
    pdf: PdfPages,
    analyte: str,
    best: dict[str, object],
    manifest_rows: list[dict[str, object]],
    skipped: list[Path],
    model_count: int,
) -> None:
    concentrations = [float(row["concentration_uM"]) for row in manifest_rows]
    lines = [
        f"{ANALYTE_LABELS[analyte]} ({analyte}) revised single-analyte PCR report",
        "",
        "Data scope:",
        f"- Data root: {DEFAULT_DATA_ROOT.relative_to(ROOT)}",
        "- CV normal: all available current channels E1-E8.",
        "- CV-GC: generator channels only, E1/E3/E5/E7.",
        "- Each CV/CV-GC file is reduced to the last complete cycle before splitting sweeps.",
        "- Anodic and cathodic fits are separate model groups.",
        f"- Current polarity: {current_polarity_label(analyte)}.",
        f"- Conditions used: {', '.join(str(int(row['condition'])) for row in manifest_rows)}.",
        f"- Concentrations used: {', '.join(clean_value(value) for value in concentrations)} uM.",
        f"- CV-like files skipped by filtering/exclusions: {len(skipped)}.",
        "",
        "Selected PCR setup:",
        f"- setup_id={int(float(best['setup_id']))}",
        f"- PCs={int(float(best['n_components']))}, score trend degree={int(float(best['score_trend_degree']))}",
        f"- smooth window={int(float(best['smooth_window']))}, smooth polyorder={int(float(best['smooth_polyorder']))}",
        f"- grid points={int(float(best['grid_points']))}",
        f"- fitted model groups={model_count}",
        "",
        "Selected setup metrics across fitted model groups:",
        f"- avg R2={float(best['avg_r2']):.4f}, median R2={float(best['median_r2']):.4f}, min R2={float(best['min_r2']):.4f}",
        f"- avg nRMSE={float(best['avg_normalized_rmse']):.4g}",
        f"- avg RMSE={float(best['avg_rmse_uA']):.4g} {CURRENT_UNIT}",
        f"- avg PC trend R2={float(best['avg_pc_trend_r2']):.4f}",
        f"- avg cumulative explained variance={float(best['avg_cumulative_explained_variance_ratio']):.4f}",
    ]
    if analyte == "AA":
        lines.insert(8, "- AA conditions 11 and 12 are excluded because they were flagged as shorted.")

    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.06, 0.94, lines[0], fontsize=17, fontweight="semibold", va="top")
    fig.text(0.06, 0.88, "\n".join(lines[2:]), fontsize=9.2, va="top", family="monospace")
    pdf.savefig(fig)
    plt.close(fig)


def plot_metric_overview(pdf: PdfPages, analyte: str, models: list[dict[str, object]]) -> None:
    models = sorted(models, key=lambda model: (str(model["technique"]), str(model["electrode"]), str(model["sweep"])))
    labels = [
        f"{model['technique'].replace('cv_', '')}\n{model['electrode']}\n{model['sweep'][:3]}"
        for model in models
    ]
    x = np.arange(len(models))
    fig, axes = plt.subplots(5, 1, figsize=(max(14, len(models) * 0.55), 12.0), sharex=True)
    fig.suptitle(f"{analyte} selected PCR setup metrics by fit group", fontweight="semibold")
    metric_specs = [
        ("r2", "Curve R2", "#0072B2"),
        ("normalized_rmse", "Normalized RMSE", "#D55E00"),
        ("rmse_uA", f"RMSE ({CURRENT_UNIT})", "#009E73"),
        ("avg_pc_trend_r2", "Avg PC trend R2", "#CC79A7"),
        ("cumulative_explained_variance_ratio", "Cumulative explained variance", "#666666"),
    ]
    for ax, (field, ylabel, color) in zip(axes, metric_specs):
        values = np.array([float(model[field]) for model in models], dtype=float)
        ax.bar(x, values, color=color)
        ax.set_ylabel(ylabel)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", direction="in")
        if field == "r2":
            ax.axhline(0, color="#777777", lw=0.7)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, rotation=90, fontsize=6)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    pdf.savefig(fig)
    plt.close(fig)


def plot_score_pages(pdf: PdfPages, analyte: str, models: list[dict[str, object]]) -> None:
    lookup = model_lookup(models)
    for electrode in sorted_electrodes(models):
        fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2))
        fig.suptitle(f"{analyte} PCA score trends | {electrode}", fontweight="semibold")
        for ax, (technique, sweep) in zip(axes.ravel(), PANEL_ORDER):
            model = lookup.get((electrode, technique, sweep))
            style_score_axis(ax)
            ax.set_title(f"{TECHNIQUE_CONFIGS[technique]['label']} | {SWEEP_LABELS[sweep]}")
            if model is None:
                ax.text(0.5, 0.5, "No fitted group", transform=ax.transAxes, ha="center", va="center")
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
                    s=20,
                    color=color,
                    label=f"PC{component} r={float(score_row['pearson_r']):.2f} fitR2={float(score_row['trend_r2']):.2f}",
                )
                ax.plot(x_fit, np.polyval(coeffs, x_fit), color=color, lw=1.1)
            add_metric_box(ax, model)
            ax.legend(loc="best", fontsize=5.8)
        fig.tight_layout(rect=(0, 0.02, 1, 0.94))
        pdf.savefig(fig)
        plt.close(fig)


def plot_overlay_pages(pdf: PdfPages, analyte: str, models: list[dict[str, object]]) -> None:
    lookup = model_lookup(models)
    for electrode in sorted_electrodes(models):
        fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.7))
        fig.suptitle(f"{analyte} actual vs PCR predicted overlays | {electrode}", fontweight="semibold")
        for ax, (technique, sweep) in zip(axes.ravel(), PANEL_ORDER):
            model = lookup.get((electrode, technique, sweep))
            style_curve_axis(ax)
            ax.set_title(f"{TECHNIQUE_CONFIGS[technique]['label']} | {SWEEP_LABELS[sweep]}")
            if model is None:
                ax.text(0.5, 0.5, "No fitted group", transform=ax.transAxes, ha="center", va="center")
                continue
            concentrations = np.asarray(model["concentrations"], dtype=float)
            colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(concentrations)))
            y_values_for_limits = []
            for index, color in enumerate(colors):
                actual = model["data_matrix"][:, index]
                predicted = model["predicted_matrix"][:, index]
                y_values_for_limits.extend([actual, predicted])
                ax.plot(model["x_grid"], actual, color=color, lw=0.9, alpha=0.78)
                ax.plot(model["x_grid"], predicted, color=color, lw=1.1, ls="--", alpha=0.95)
            ax.plot([], [], color="#444444", lw=1.0, label="actual")
            ax.plot([], [], color="#444444", lw=1.0, ls="--", label="PCR predicted")
            set_curve_y_limits(ax, y_values_for_limits)
            add_metric_box(ax, model)
            ax.legend(loc="best", fontsize=6)
        fig.tight_layout(rect=(0, 0.02, 1, 0.94))
        pdf.savefig(fig)
        plt.close(fig)


def condition_label(models: list[dict[str, object]], condition: int) -> str:
    for model in models:
        matches = np.where(np.asarray(model["conditions"], dtype=int) == int(condition))[0]
        if len(matches):
            concentration = float(np.asarray(model["concentrations"], dtype=float)[int(matches[0])])
            return f"Condition {condition} | {clean_value(concentration)} uM"
    return f"Condition {condition}"


def chunks(values: list[str], size: int) -> list[list[str]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def set_curve_y_limits(
    ax: plt.Axes,
    values: list[np.ndarray],
    padding_fraction: float = 0.08,
) -> None:
    finite = np.concatenate([np.asarray(value, dtype=float).ravel() for value in values])
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return
    y_min = float(np.min(finite))
    y_max = float(np.max(finite))
    y_span = y_max - y_min
    if y_span <= 0:
        y_span = max(abs(y_max), 1.0)
    padding = padding_fraction * y_span
    ax.set_ylim(y_min - padding, y_max + padding)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=11))
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))


def plot_single_condition_grid(
    pdf: PdfPages,
    analyte: str,
    condition: int,
    models: list[dict[str, object]],
    technique: str,
    electrodes: list[str],
    page_label: str = "",
) -> None:
    lookup = model_lookup(models)
    n_rows = len(electrodes)
    fig_height = max(7.2, 3.05 * n_rows + 1.25)
    fig, axes = plt.subplots(n_rows, 2, figsize=(12.4, fig_height), squeeze=False)
    label_suffix = f" | {page_label}" if page_label else ""
    fig.suptitle(
        f"{analyte} PCR condition fit | {condition_label(models, condition)} | {TECHNIQUE_CONFIGS[technique]['label']}{label_suffix}",
        fontweight="semibold",
    )
    for row_index, electrode in enumerate(electrodes):
        for col_index, sweep in enumerate(("anodic", "cathodic")):
            ax = axes[row_index, col_index]
            model = lookup.get((electrode, technique, sweep))
            style_curve_axis(ax, show_xlabel=row_index == n_rows - 1, show_ylabel=col_index == 0)
            ax.set_title(f"{electrode} | {SWEEP_LABELS[sweep]}", fontsize=7.5)
            if model is None:
                ax.text(0.5, 0.5, "No fitted group", transform=ax.transAxes, ha="center", va="center")
                continue
            matches = np.where(np.asarray(model["conditions"], dtype=int) == int(condition))[0]
            if len(matches) == 0:
                ax.text(0.5, 0.5, "No condition", transform=ax.transAxes, ha="center", va="center")
                continue
            index = int(matches[0])
            actual = model["data_matrix"][:, index]
            predicted = model["predicted_matrix"][:, index]
            residual = model["residual_matrix"][:, index]
            per_condition_r2 = float(model["per_condition_r2"][index])
            per_condition_rmse = float(model["per_condition_rmse_uA"][index])
            ax.plot(model["x_grid"], actual, color="#444444", lw=0.95, label="actual")
            ax.plot(model["x_grid"], predicted, color=ANALYTE_COLORS[analyte], lw=1.05, ls="--", label="PCR predicted")
            ax.plot(model["x_grid"], residual, color="#0072B2", lw=0.65, alpha=0.75, label="residual")
            set_curve_y_limits(ax, [actual, predicted, residual])
            ax.text(
                0.02,
                0.96,
                f"R2c={per_condition_r2:.3f}\nRMSEc={per_condition_rmse:.3g} {CURRENT_UNIT}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=5.8,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72},
            )
            if row_index == 0 and col_index == 0:
                ax.legend(loc="best", fontsize=5.8)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    pdf.savefig(fig)
    plt.close(fig)


def plot_condition_fit_pages(pdf: PdfPages, analyte: str, models: list[dict[str, object]]) -> None:
    conditions = sorted({int(condition) for model in models for condition in np.asarray(model["conditions"], dtype=int)})
    cv_normal_electrodes = [f"E{i}" for i in range(1, 9)]
    cv_gc_electrodes = ["E1 generator", "E3 generator", "E5 generator", "E7 generator"]
    for condition in conditions:
        for electrode_group in chunks(cv_normal_electrodes, 2):
            plot_single_condition_grid(
                pdf,
                analyte,
                condition,
                models,
                "cv_normal",
                electrode_group,
                "-".join(electrode_group),
            )
        for electrode_group in chunks(cv_gc_electrodes, 2):
            plot_single_condition_grid(
                pdf,
                analyte,
                condition,
                models,
                "cv_gc",
                electrode_group,
                " / ".join(electrode_group),
            )


def write_selected_metrics_csv(path: Path, analyte: str, models: list[dict[str, object]], best: dict[str, object]) -> None:
    rows = build_detail_rows(analyte, int(float(best["setup_id"])), params_from_summary(best), models)
    write_csv(path, rows, DETAIL_FIELDNAMES)


def write_selected_metadata(path: Path, analyte: str, best: dict[str, object], manifest_rows: list[dict[str, object]]) -> None:
    metadata = {
        "analyte": analyte,
        "analyte_label": ANALYTE_LABELS[analyte],
        "selected_setup": best,
        "conditions": manifest_rows,
        "current_polarity": current_polarity_label(analyte),
        "equation": "current(V,c)=mean_curve(V)+sum_j(polyval(score_coefficients_j,c)*loading_j(V))",
    }
    with path.open("w") as handle:
        json.dump(metadata, handle, indent=2, default=json_default)
        handle.write("\n")


def write_analyte_pdf(
    path: Path,
    analyte: str,
    models: list[dict[str, object]],
    best: dict[str, object],
    manifest_rows: list[dict[str, object]],
    skipped: list[Path],
) -> None:
    with PdfPages(path) as pdf:
        write_intro_page(pdf, analyte, best, manifest_rows, skipped, len(models))
        plot_metric_overview(pdf, analyte, models)
        plot_score_pages(pdf, analyte, models)
        plot_overlay_pages(pdf, analyte, models)
        plot_condition_fit_pages(pdf, analyte, models)


def run(args: argparse.Namespace) -> None:
    configure_plot_style()
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_paths: list[Path] = []
    summary_for_all: list[dict[str, object]] = []
    for analyte in ANALYTE_ORDER:
        analyte_output = output_dir / analyte.lower()
        analyte_output.mkdir(parents=True, exist_ok=True)
        entries, manifest_rows, skipped = load_trace_entries(data_root, analyte)
        if not entries:
            raise RuntimeError(f"No trace entries loaded for {analyte}")
        condition_count = len({int(row["condition"]) for row in manifest_rows})
        print(
            f"{analyte}: loaded {len(entries)} trace entries from {condition_count} conditions "
            f"({current_polarity_label(analyte)})"
        )
        write_manifest(analyte_output / "included_conditions.csv", analyte, manifest_rows, skipped)
        _detailed_rows, summary_rows, best = fit_analyte_sweep(
            analyte,
            entries,
            condition_count,
            args,
            output_dir,
        )
        best_params = params_from_summary(best)
        selected_models = fit_pcr_models(entries, best_params, scopes={"electrode"})
        selected_models = sorted(
            selected_models,
            key=lambda model: (str(model["technique"]), str(model["electrode"]), str(model["sweep"])),
        )
        write_selected_metrics_csv(analyte_output / "selected_setup_model_metrics.csv", analyte, selected_models, best)
        write_selected_metadata(analyte_output / "selected_setup_metadata.json", analyte, best, manifest_rows)

        pdf_path = data_root / f"revised_{analyte}_pcr_fit_report.pdf"
        write_analyte_pdf(pdf_path, analyte, selected_models, best, manifest_rows, skipped)
        pdf_paths.append(pdf_path)
        summary_for_all.append(best)
        print(
            f"{analyte}: selected setup {int(float(best['setup_id']))} | "
            f"PCs={int(float(best['n_components']))} trend={int(float(best['score_trend_degree']))} "
            f"smooth={int(float(best['smooth_window']))}/{int(float(best['smooth_polyorder']))} "
            f"grid={int(float(best['grid_points']))} | avg R2={float(best['avg_r2']):.4f}"
        )
        print(f"{analyte}: wrote PDF {pdf_path}")

    write_csv(output_dir / "selected_setup_summary.csv", summary_for_all, SUMMARY_FIELDNAMES)
    print("Wrote selected setup summary:", output_dir / "selected_setup_summary.csv")
    for pdf_path in pdf_paths:
        print("Report PDF:", pdf_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
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
        default="0,250,500",
        help='Aligned voltage grid point counts. Use 0 for native common grid. Example: "0,250,500".',
    )
    parser.add_argument(
        "--n-components",
        default=f"1-{DEFAULT_N_COMPONENTS + 2}",
        help='PCA component counts to sweep. Example: "1-5".',
    )
    parser.add_argument(
        "--score-trend-degrees",
        default=f"1,{DEFAULT_SCORE_TREND_DEGREE},3",
        help='Polynomial degrees for PC-score interpolation. Example: "1,2,3".',
    )
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--limit-setups", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
