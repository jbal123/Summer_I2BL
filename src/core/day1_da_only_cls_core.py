#!/usr/bin/env python3
"""Shared CLS utilities for the Day 1 DA-only CV workflow."""

from __future__ import annotations

# --- path bootstrap (auto-added during repo reorg): make sibling modules importable ---
import os as _os, sys as _sys
_SRC_DIR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _sub in ("core", "features", "day1_da_only", "single_analyte", "multi_analyte", "models"):
    _p = _os.path.join(_SRC_DIR, _sub)
    if _os.path.isdir(_p) and _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end path bootstrap ---


import csv
import math
import os
import re
import tempfile
from pathlib import Path

import numpy as np

try:
    from scipy.signal import savgol_filter
except Exception:  # pragma: no cover
    savgol_filter = None


CACHE_ROOT = Path(tempfile.gettempdir()) / "microneedlearrayml_matplotlib"
(CACHE_ROOT / "mplconfig").mkdir(parents=True, exist_ok=True)
(CACHE_ROOT / "xdg_cache").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT / "xdg_cache"))

REPO_ROOT = Path(__file__).resolve().parents[2]
# ROOT anchors the raw-data tree; manifests store condition paths relative to it.
ROOT = REPO_ROOT / "data"
RESULTS_ROOT = REPO_ROOT / "results"
DEFAULT_SUBSET_DIR = ROOT / "Day1_DA_Only_Conditions"
DEFAULT_MANIFEST = DEFAULT_SUBSET_DIR / "day1_da_only_conditions.csv"

CURRENT_SCALE = 1e6
CURRENT_UNIT = "uA"
SENSITIVITY_UNIT = "uA/uM"
DEFAULT_SMOOTH_WINDOW = 51
DEFAULT_SMOOTH_POLYORDER = 3
DEFAULT_GRID_POINTS = 0
DEFAULT_MODEL_ORDER = "linear"

MODEL_ORDER_DEGREES = {
    "linear": 1,
    "quadratic": 2,
    "cubic": 3,
}

NUMBER_PATTERN = re.compile(
    r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"
)

TECHNIQUES = {
    "cv_normal": {
        "label": "CV normal",
        "filename_pattern": "cv_norm",
        "electrode_columns": {"E1": 1, "E3": 2, "E5": 3, "E7": 4},
    },
    "cv_gc": {
        "label": "CV GC",
        "filename_pattern": "cv_gc",
        "electrode_columns": {"E1": 1, "E3": 3, "E5": 5, "E7": 7},
    },
}

SWEEPS = {
    "anodic": "Anodic",
    "cathodic": "Cathodic",
}

PANEL_ORDER = [
    ("cv_normal", "anodic"),
    ("cv_normal", "cathodic"),
    ("cv_gc", "anodic"),
    ("cv_gc", "cathodic"),
]


def clean_value(value: object) -> str:
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return str(value)
    if value_float.is_integer():
        return str(int(value_float))
    return str(value_float)


def slug_float(value: object) -> str:
    return clean_value(value).replace("-", "neg").replace(".", "p")


def format_float(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return ""
    return f"{value:.12g}"


def format_series(values: list[float] | np.ndarray) -> str:
    return ";".join(format_float(float(value)) for value in values)


def normalize_model_order(value: object = DEFAULT_MODEL_ORDER) -> str:
    if value is None:
        return DEFAULT_MODEL_ORDER
    text = str(value).strip().lower()
    if text in {"1", "degree1", "degree_1", "linear"}:
        return "linear"
    if text in {"2", "degree2", "degree_2", "quadratic"}:
        return "quadratic"
    if text in {"3", "degree3", "degree_3", "cubic"}:
        return "cubic"
    raise ValueError(f"Unsupported CLS model order: {value!r}")


def model_order_degree(value: object = DEFAULT_MODEL_ORDER) -> int:
    return MODEL_ORDER_DEGREES[normalize_model_order(value)]


def calibration_powers_for_order(value: object = DEFAULT_MODEL_ORDER) -> np.ndarray:
    degree = model_order_degree(value)
    return np.arange(1, degree + 1, dtype=int)


def calibration_term_names(value: object = DEFAULT_MODEL_ORDER) -> list[str]:
    return [f"c^{power}" for power in calibration_powers_for_order(value)]


def calibration_design_matrix(
    concentrations: np.ndarray,
    value: object = DEFAULT_MODEL_ORDER,
) -> tuple[np.ndarray, np.ndarray]:
    powers = calibration_powers_for_order(value)
    design = np.vstack([concentrations ** int(power) for power in powers])
    return design, powers


def parse_int_list(text: str) -> list[int]:
    values = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            step = 1 if end >= start else -1
            values.extend(range(start, end + step, step))
        else:
            values.append(int(part))
    return sorted(dict.fromkeys(values))


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


def read_manifest(manifest_path: Path = DEFAULT_MANIFEST) -> list[dict[str, str]]:
    rows = read_csv_dicts(manifest_path)
    required = {"Condition", "Dopamine (uM)", "Subset Folder"}
    missing = required.difference(rows[0].keys() if rows else [])
    if missing:
        raise ValueError(f"Manifest is missing columns: {', '.join(sorted(missing))}")
    return sorted(rows, key=lambda row: float(row["Dopamine (uM)"]))


def find_technique_file(condition_folder: Path, filename_pattern: str) -> Path | None:
    matches = sorted(
        file_path
        for file_path in condition_folder.glob("*.txt")
        if filename_pattern in file_path.name.lower()
    )
    return matches[0] if matches else None


def load_numeric_rows(file_path: Path) -> list[list[float]]:
    rows = []
    with file_path.open("r", errors="ignore") as handle:
        for line in handle:
            stripped = line.strip()
            if not NUMBER_PATTERN.match(stripped):
                continue
            values = [float(value) for value in NUMBER_PATTERN.findall(stripped)]
            if len(values) >= 2:
                rows.append(values)
    if not rows:
        raise ValueError(f"No numeric rows found in {file_path}")
    return rows


def split_cv_sweeps(rows: list[list[float]]) -> dict[str, list[list[float]]]:
    x_values = [row[0] for row in rows]
    deltas = [x_values[index + 1] - x_values[index] for index in range(len(x_values) - 1)]
    nonzero_deltas = [(index, delta) for index, delta in enumerate(deltas) if abs(delta) > 1e-12]
    if not nonzero_deltas:
        return {"anodic": rows, "cathodic": []}

    first_sign = 1 if nonzero_deltas[0][1] > 0 else -1
    split_index = None
    for index, delta in nonzero_deltas[1:]:
        current_sign = 1 if delta > 0 else -1
        if current_sign != first_sign:
            split_index = index
            break

    if split_index is None:
        forward_rows = rows
        reverse_rows = []
    else:
        forward_rows = rows[: split_index + 1]
        reverse_rows = rows[split_index:]

    if first_sign > 0:
        return {"anodic": forward_rows, "cathodic": reverse_rows}
    return {"anodic": reverse_rows, "cathodic": forward_rows}


def sort_trace(x_values: np.ndarray, y_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(x_values)
    return x_values[order], y_values[order]


def extract_electrode_traces(
    sweep_rows: list[list[float]],
    electrode_columns: dict[str, int],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    traces = {}
    for electrode, column_index in electrode_columns.items():
        x_values = []
        y_values = []
        for row in sweep_rows:
            if column_index >= len(row):
                continue
            x_values.append(row[0])
            y_values.append(row[column_index] * CURRENT_SCALE)
        if len(x_values) > 10:
            traces[electrode] = sort_trace(np.array(x_values), np.array(y_values))
    return traces


def mean_trace_from_electrodes(
    traces: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray] | None:
    if not traces:
        return None
    first_x = next(iter(traces.values()))[0]
    y_arrays = []
    for x_values, y_values in traces.values():
        if len(x_values) == len(first_x) and np.allclose(x_values, first_x):
            y_arrays.append(y_values)
        else:
            y_arrays.append(np.interp(first_x, x_values, y_values))
    return first_x, np.mean(np.vstack(y_arrays), axis=0)


def load_trace_entries(
    conditions: list[dict[str, str]],
    subset_dir: Path = DEFAULT_SUBSET_DIR,
) -> list[dict[str, object]]:
    entries = []
    for condition in conditions:
        condition_id = int(float(condition["Condition"]))
        da_uM = float(condition["Dopamine (uM)"])
        condition_folder = ROOT / condition["Subset Folder"]
        if not condition_folder.is_dir():
            condition_folder = subset_dir / f"Condition_{condition_id}"

        for technique_key, technique_config in TECHNIQUES.items():
            source_file = find_technique_file(condition_folder, str(technique_config["filename_pattern"]))
            if source_file is None:
                print(f"Missing {technique_config['label']} file for Condition {condition_id}")
                continue
            rows = load_numeric_rows(source_file)
            sweeps = split_cv_sweeps(rows)
            for sweep_key, sweep_rows in sweeps.items():
                traces = extract_electrode_traces(sweep_rows, technique_config["electrode_columns"])
                for electrode, (x_values, y_values) in traces.items():
                    entries.append(
                        {
                            "scope": "electrode",
                            "condition": condition_id,
                            "da_uM": da_uM,
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
                            "condition": condition_id,
                            "da_uM": da_uM,
                            "technique": technique_key,
                            "sweep": sweep_key,
                            "electrode": "mean",
                            "x": mean_x,
                            "y": mean_y,
                            "source_file": str(source_file.relative_to(ROOT)),
                        }
                    )
    return entries


def adjusted_smooth_window(length: int, requested_window: int, polyorder: int) -> int:
    if requested_window <= 1 or length <= polyorder + 2:
        return 0
    window = min(requested_window, length if length % 2 == 1 else length - 1)
    if window % 2 == 0:
        window -= 1
    if window <= polyorder:
        window = polyorder + 2
        if window % 2 == 0:
            window += 1
    if window > length:
        return 0
    return window


def smooth_signal(y_values: np.ndarray, requested_window: int, polyorder: int) -> tuple[np.ndarray, int]:
    window = adjusted_smooth_window(len(y_values), requested_window, polyorder)
    if window == 0:
        return y_values.copy(), 0
    if savgol_filter is not None:
        return savgol_filter(y_values, window_length=window, polyorder=polyorder), window
    pad = window // 2
    padded = np.pad(y_values, (pad, pad), mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode="valid"), window


def group_records(records: list[dict[str, object]], keys: tuple[str, ...]) -> dict[tuple[object, ...], list[dict[str, object]]]:
    grouped = {}
    for record in records:
        key = tuple(record[key_name] for key_name in keys)
        grouped.setdefault(key, []).append(record)
    return grouped


def safe_r2(actual: np.ndarray, predicted: np.ndarray) -> float:
    total = float(np.sum((actual - np.mean(actual)) ** 2))
    if total <= 0:
        return math.nan
    residual = float(np.sum((actual - predicted) ** 2))
    return float(1.0 - residual / total)


def align_group_records(
    records: list[dict[str, object]],
    params: dict[str, object],
) -> dict[str, object]:
    sorted_records = sorted(records, key=lambda record: float(record["da_uM"]))
    smooth_window = int(params.get("smooth_window", DEFAULT_SMOOTH_WINDOW))
    smooth_polyorder = 0 if smooth_window <= 1 else int(params.get("smooth_polyorder", DEFAULT_SMOOTH_POLYORDER))
    requested_grid_points = int(params.get("grid_points", DEFAULT_GRID_POINTS))

    x_mins = [float(np.min(record["x"])) for record in sorted_records]
    x_maxs = [float(np.max(record["x"])) for record in sorted_records]
    grid_min = max(x_mins)
    grid_max = min(x_maxs)
    if grid_max <= grid_min:
        raise ValueError("Trace voltage ranges do not overlap enough for CLS alignment")

    first_x = np.asarray(sorted_records[0]["x"], dtype=float)
    same_grid = all(
        len(record["x"]) == len(first_x) and np.allclose(record["x"], first_x)
        for record in sorted_records
    )
    if requested_grid_points > 0:
        x_grid = np.linspace(grid_min, grid_max, requested_grid_points)
    elif same_grid:
        overlap_mask = (first_x >= grid_min) & (first_x <= grid_max)
        x_grid = first_x[overlap_mask]
    else:
        min_points = min(len(record["x"]) for record in sorted_records)
        x_grid = np.linspace(grid_min, grid_max, min_points)

    aligned_columns = []
    raw_columns = []
    aligned_records = []
    actual_smooth_windows = []
    for record in sorted_records:
        x_values = np.asarray(record["x"], dtype=float)
        y_values = np.asarray(record["y"], dtype=float)
        y_target, actual_smooth_window = smooth_signal(y_values, smooth_window, smooth_polyorder)
        actual_smooth_windows.append(actual_smooth_window)
        if len(x_values) == len(x_grid) and np.allclose(x_values, x_grid):
            y_aligned = y_target.copy()
            raw_aligned = y_values.copy()
        else:
            y_aligned = np.interp(x_grid, x_values, y_target)
            raw_aligned = np.interp(x_grid, x_values, y_values)
        aligned_columns.append(y_aligned)
        raw_columns.append(raw_aligned)
        aligned_records.append(
            {
                **record,
                "x_aligned": x_grid,
                "y_aligned": y_aligned,
                "raw_y_aligned": raw_aligned,
                "actual_smooth_window": actual_smooth_window,
                "smooth_polyorder_effective": smooth_polyorder if actual_smooth_window else 0,
            }
        )

    return {
        "records": aligned_records,
        "x_grid": x_grid,
        "data_matrix": np.column_stack(aligned_columns),
        "raw_data_matrix": np.column_stack(raw_columns),
        "conditions": np.array([int(record["condition"]) for record in sorted_records], dtype=int),
        "concentrations": np.array([float(record["da_uM"]) for record in sorted_records], dtype=float),
        "actual_smooth_window": max(actual_smooth_windows) if actual_smooth_windows else 0,
        "smooth_polyorder_effective": smooth_polyorder if any(actual_smooth_windows) else 0,
    }


def fit_cls_matrix(
    data_matrix: np.ndarray,
    concentrations: np.ndarray,
    model_order: object = DEFAULT_MODEL_ORDER,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit D = S.T @ X for one analyte.

    data_matrix is D with shape (n_voltages, n_concentrations).
    concentrations is C flattened to shape (n_concentrations,). For linear,
    X = [C]. For quadratic, X = [C, C^2]. For cubic, X = [C, C^2, C^3].
    """
    design, powers = calibration_design_matrix(concentrations, model_order)
    if design.shape[1] < design.shape[0]:
        raise ValueError(
            f"CLS {normalize_model_order(model_order)} requires at least "
            f"{design.shape[0]} calibration concentrations"
        )
    if np.linalg.matrix_rank(design.T) < design.shape[0]:
        raise ValueError("CLS calibration design is rank deficient")
    term_vectors, *_ = np.linalg.lstsq(design.T, data_matrix.T, rcond=None)
    predicted_matrix = term_vectors.T @ design
    residual_matrix = data_matrix - predicted_matrix
    return term_vectors, predicted_matrix, residual_matrix, powers


def cls_metrics(data_matrix: np.ndarray, predicted_matrix: np.ndarray) -> dict[str, object]:
    residual_matrix = data_matrix - predicted_matrix
    rmse = float(np.sqrt(np.mean(residual_matrix ** 2)))
    mae = float(np.mean(np.abs(residual_matrix)))
    max_abs = float(np.max(np.abs(residual_matrix)))
    data_span = float(np.max(data_matrix) - np.min(data_matrix))
    nrmse = rmse / data_span if data_span > 0 else math.nan
    r2 = safe_r2(data_matrix, predicted_matrix)

    per_condition_rmse = np.sqrt(np.mean(residual_matrix ** 2, axis=0))
    per_condition_mae = np.mean(np.abs(residual_matrix), axis=0)
    per_condition_r2 = np.array(
        [
            safe_r2(data_matrix[:, index], predicted_matrix[:, index])
            for index in range(data_matrix.shape[1])
        ],
        dtype=float,
    )
    per_voltage_rmse = np.sqrt(np.mean(residual_matrix ** 2, axis=1))
    per_voltage_mae = np.mean(np.abs(residual_matrix), axis=1)
    return {
        "rmse_uA": rmse,
        "mae_uA": mae,
        "max_abs_residual_uA": max_abs,
        "normalized_rmse": nrmse,
        "r2": r2,
        "per_condition_rmse_uA": per_condition_rmse,
        "per_condition_mae_uA": per_condition_mae,
        "per_condition_r2": per_condition_r2,
        "per_voltage_rmse_uA": per_voltage_rmse,
        "per_voltage_mae_uA": per_voltage_mae,
    }


def fit_cls_group(
    records: list[dict[str, object]],
    params: dict[str, object],
) -> dict[str, object]:
    if len(records) < 2:
        raise ValueError("CLS group requires at least two concentration records")
    aligned = align_group_records(records, params)
    model_order = normalize_model_order(params.get("model_order", DEFAULT_MODEL_ORDER))
    term_vectors, predicted_matrix, residual_matrix, powers = fit_cls_matrix(
        aligned["data_matrix"],
        aligned["concentrations"],
        model_order,
    )
    metrics = cls_metrics(aligned["data_matrix"], predicted_matrix)
    example = aligned["records"][0]
    return {
        "method": "cls",
        "model_order": model_order,
        "calibration_degree": int(len(powers)),
        "calibration_powers": powers,
        "calibration_term_names": calibration_term_names(model_order),
        "scope": example["scope"],
        "technique": example["technique"],
        "technique_label": TECHNIQUES[str(example["technique"])]["label"],
        "sweep": example["sweep"],
        "sweep_label": SWEEPS[str(example["sweep"])],
        "electrode": example["electrode"],
        "x_grid": aligned["x_grid"],
        "conditions": aligned["conditions"],
        "concentrations": aligned["concentrations"],
        "data_matrix": aligned["data_matrix"],
        "raw_data_matrix": aligned["raw_data_matrix"],
        "predicted_matrix": predicted_matrix,
        "residual_matrix": residual_matrix,
        "calibration_vectors": term_vectors,
        "sensitivity_uA_per_uM": term_vectors[0],
        "quadratic_sensitivity_uA_per_uM2": term_vectors[1] if len(term_vectors) > 1 else None,
        "cubic_sensitivity_uA_per_uM3": term_vectors[2] if len(term_vectors) > 2 else None,
        "records": aligned["records"],
        "n_voltages": int(aligned["data_matrix"].shape[0]),
        "n_concentrations": int(aligned["data_matrix"].shape[1]),
        "smooth_window": int(params.get("smooth_window", DEFAULT_SMOOTH_WINDOW)),
        "smooth_polyorder": int(params.get("smooth_polyorder", DEFAULT_SMOOTH_POLYORDER)),
        "actual_smooth_window": int(aligned["actual_smooth_window"]),
        "smooth_polyorder_effective": int(aligned["smooth_polyorder_effective"]),
        "grid_points": int(params.get("grid_points", DEFAULT_GRID_POINTS)),
        **metrics,
    }


def fit_cls_models(
    entries: list[dict[str, object]],
    params: dict[str, object],
    scopes: set[str] | None = None,
) -> list[dict[str, object]]:
    grouped = group_records(entries, ("scope", "technique", "sweep", "electrode"))
    models = []
    for (_scope, _technique, _sweep, _electrode), records in sorted(grouped.items()):
        if scopes is not None and str(_scope) not in scopes:
            continue
        models.append(fit_cls_group(records, params))
    return models


def build_prediction_model(
    cls_models: list[dict[str, object]],
    params: dict[str, object],
    scope: str = "mean",
) -> dict[str, object]:
    curves = {}
    model_order = normalize_model_order(params.get("model_order", DEFAULT_MODEL_ORDER))
    for model in cls_models:
        if model["scope"] != scope:
            continue
        curve_key = f"{model['technique']}_{model['sweep']}"
        curves[curve_key] = {
            "technique": model["technique"],
            "technique_label": model["technique_label"],
            "sweep": model["sweep"],
            "sweep_label": model["sweep_label"],
            "electrode": model["electrode"],
            "x_grid": model["x_grid"],
            "training_da_uM": model["concentrations"],
            "conditions": model["conditions"],
            "calibration_powers": model["calibration_powers"],
            "calibration_term_names": model["calibration_term_names"],
            "calibration_vectors": model["calibration_vectors"],
            "sensitivity_uA_per_uM": model["sensitivity_uA_per_uM"],
            "quadratic_sensitivity_uA_per_uM2": model["quadratic_sensitivity_uA_per_uM2"],
            "cubic_sensitivity_uA_per_uM3": model["cubic_sensitivity_uA_per_uM3"],
            "metrics": {
                "rmse_uA": model["rmse_uA"],
                "mae_uA": model["mae_uA"],
                "max_abs_residual_uA": model["max_abs_residual_uA"],
                "normalized_rmse": model["normalized_rmse"],
                "r2": model["r2"],
            },
        }
    return {
        "method": "cls",
        "model_order": model_order,
        "calibration_degree": model_order_degree(model_order),
        "equation": "D = S.T @ X; linear X=[C], quadratic X=[C, C^2], cubic X=[C, C^2, C^3]",
        "params": params,
        "current_unit": CURRENT_UNIT,
        "sensitivity_unit": SENSITIVITY_UNIT,
        "curves": curves,
    }


def predict_curves(
    model: dict[str, object],
    concentration_uM: float,
    points: int | None = None,
) -> dict[str, dict[str, object]]:
    predicted = {}
    model_order = normalize_model_order(model.get("model_order", model.get("params", {}).get("model_order", DEFAULT_MODEL_ORDER)))
    for curve_key, curve in model["curves"].items():
        x_grid = np.asarray(curve["x_grid"], dtype=float)
        calibration_vectors = np.asarray(curve.get("calibration_vectors", [curve["sensitivity_uA_per_uM"]]), dtype=float)
        if points is None or int(points) <= 0 or int(points) == len(x_grid):
            potential_v = x_grid
            vectors_grid = calibration_vectors
        else:
            potential_v = np.linspace(float(np.min(x_grid)), float(np.max(x_grid)), int(points))
            vectors_grid = np.vstack(
                [np.interp(potential_v, x_grid, vector) for vector in calibration_vectors]
            )
        powers = np.asarray(curve.get("calibration_powers", calibration_powers_for_order(model_order)), dtype=int)
        factors = np.array([float(concentration_uM) ** int(power) for power in powers], dtype=float)
        current_uA = vectors_grid.T @ factors
        predicted[curve_key] = {
            "curve": curve_key,
            "technique": curve["technique"],
            "technique_label": curve["technique_label"],
            "sweep": curve["sweep"],
            "sweep_label": curve["sweep_label"],
            "potential_v": potential_v,
            "current_uA": current_uA,
            "model_order": model_order,
            "calibration_powers": powers,
            "calibration_vectors": vectors_grid,
            "sensitivity_uA_per_uM": vectors_grid[0],
            "quadratic_sensitivity_uA_per_uM2": vectors_grid[1] if len(vectors_grid) > 1 else None,
            "cubic_sensitivity_uA_per_uM3": vectors_grid[2] if len(vectors_grid) > 2 else None,
        }
    return predicted


def write_prediction_csv(predicted: dict[str, dict[str, object]], concentration: float, output_path: Path) -> None:
    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["requested_da_uM", "curve", "technique", "sweep", "potential_v", "current_uA"])
        for curve_key, curve in predicted.items():
            for potential_v, current_uA in zip(curve["potential_v"], curve["current_uA"]):
                writer.writerow(
                    [
                        concentration,
                        curve_key,
                        curve["technique"],
                        curve["sweep"],
                        float(potential_v),
                        float(current_uA),
                    ]
                )


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"Cannot JSON encode {type(value)}")
