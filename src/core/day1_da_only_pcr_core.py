#!/usr/bin/env python3
"""Shared PCR/PCA utilities for the Day 1 DA-only CV workflow."""

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
from pathlib import Path

import numpy as np

from day1_da_only_cls_core import (
    CURRENT_UNIT,
    DEFAULT_GRID_POINTS,
    DEFAULT_MANIFEST,
    DEFAULT_SMOOTH_POLYORDER,
    DEFAULT_SMOOTH_WINDOW,
    DEFAULT_SUBSET_DIR,
    PANEL_ORDER,
    SWEEPS,
    TECHNIQUES,
    align_group_records,
    clean_value,
    cls_metrics,
    format_float,
    format_series,
    group_records,
    json_default,
    load_trace_entries,
    parse_int_list,
    read_csv_dicts,
    read_manifest,
    safe_r2,
    slug_float,
)


DEFAULT_N_COMPONENTS = 3
DEFAULT_SCORE_TREND_DEGREE = 2


def normalize_positive_int(value: object, name: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if result < 1:
        raise ValueError(f"{name} must be >= 1")
    return result


def pearson_corr(x_values: np.ndarray, y_values: np.ndarray) -> float:
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2:
        return math.nan
    x_std = float(np.std(x))
    y_std = float(np.std(y))
    if x_std <= 0 or y_std <= 0:
        return math.nan
    return float(np.corrcoef(x, y)[0, 1])


def rank_values(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    index = 0
    while index < len(values):
        end = index + 1
        while end < len(values) and values[order[end]] == values[order[index]]:
            end += 1
        ranks[order[index:end]] = 0.5 * (index + end - 1) + 1.0
        index = end
    return ranks


def spearman_corr(x_values: np.ndarray, y_values: np.ndarray) -> float:
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2:
        return math.nan
    return pearson_corr(rank_values(x), rank_values(y))


def effective_pcr_components(n_samples: int, n_voltages: int, requested: int) -> int:
    requested = normalize_positive_int(requested, "n_components")
    max_components = min(max(n_samples - 1, 1), n_voltages)
    return min(requested, max_components)


def effective_trend_degree(concentrations: np.ndarray, requested: int) -> int:
    requested = normalize_positive_int(requested, "score_trend_degree")
    unique_count = len(np.unique(np.asarray(concentrations, dtype=float)))
    max_degree = max(1, min(unique_count - 1, len(concentrations) - 1))
    return min(requested, max_degree)


def orient_pca_to_concentration(
    scores: np.ndarray,
    loadings: np.ndarray,
    concentrations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    scores = scores.copy()
    loadings = loadings.copy()
    for component_index in range(scores.shape[1]):
        corr = pearson_corr(concentrations, scores[:, component_index])
        if np.isfinite(corr) and corr < 0:
            scores[:, component_index] *= -1.0
            loadings[component_index, :] *= -1.0
    return scores, loadings


def fit_score_trends(
    concentrations: np.ndarray,
    scores: np.ndarray,
    requested_degree: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    degree = effective_trend_degree(concentrations, requested_degree)
    coefficients = []
    predicted_columns = []
    for component_index in range(scores.shape[1]):
        component_coefficients = np.polyfit(concentrations, scores[:, component_index], degree)
        coefficients.append(component_coefficients)
        predicted_columns.append(np.polyval(component_coefficients, concentrations))
    return np.vstack(coefficients), np.column_stack(predicted_columns), degree


def score_correlation_rows(
    concentrations: np.ndarray,
    scores: np.ndarray,
    predicted_scores: np.ndarray,
    explained_variance_ratio: np.ndarray,
    trend_coefficients: np.ndarray,
) -> list[dict[str, object]]:
    rows = []
    for component_index in range(scores.shape[1]):
        actual = scores[:, component_index]
        predicted = predicted_scores[:, component_index]
        rows.append(
            {
                "component": component_index + 1,
                "pearson_r": pearson_corr(concentrations, actual),
                "spearman_r": spearman_corr(concentrations, actual),
                "trend_r2": safe_r2(actual, predicted),
                "trend_rmse": float(np.sqrt(np.mean((actual - predicted) ** 2))),
                "explained_variance_ratio": float(explained_variance_ratio[component_index]),
                "scores": actual.copy(),
                "predicted_scores": predicted.copy(),
                "trend_coefficients_desc": trend_coefficients[component_index].copy(),
            }
        )
    return rows


def summarize_score_correlations(score_rows: list[dict[str, object]]) -> dict[str, float]:
    if not score_rows:
        return {
            "avg_abs_pc_pearson": math.nan,
            "avg_abs_pc_spearman": math.nan,
            "avg_pc_trend_r2": math.nan,
        }
    pearson = [
        abs(float(row["pearson_r"]))
        for row in score_rows
        if np.isfinite(float(row["pearson_r"]))
    ]
    spearman = [
        abs(float(row["spearman_r"]))
        for row in score_rows
        if np.isfinite(float(row["spearman_r"]))
    ]
    trend_r2 = [
        float(row["trend_r2"])
        for row in score_rows
        if np.isfinite(float(row["trend_r2"]))
    ]
    return {
        "avg_abs_pc_pearson": float(np.mean(pearson)) if pearson else math.nan,
        "avg_abs_pc_spearman": float(np.mean(spearman)) if spearman else math.nan,
        "avg_pc_trend_r2": float(np.mean(trend_r2)) if trend_r2 else math.nan,
    }


def fit_pcr_group(
    records: list[dict[str, object]],
    params: dict[str, object],
) -> dict[str, object]:
    if len(records) < 2:
        raise ValueError("PCR group requires at least two concentration records")

    aligned = align_group_records(records, params)
    data_matrix = np.asarray(aligned["data_matrix"], dtype=float)
    concentrations = np.asarray(aligned["concentrations"], dtype=float)
    x_grid = np.asarray(aligned["x_grid"], dtype=float)

    n_voltages, n_samples = data_matrix.shape
    requested_components = normalize_positive_int(
        params.get("n_components", DEFAULT_N_COMPONENTS),
        "n_components",
    )
    n_components = effective_pcr_components(n_samples, n_voltages, requested_components)
    requested_trend_degree = normalize_positive_int(
        params.get("score_trend_degree", DEFAULT_SCORE_TREND_DEGREE),
        "score_trend_degree",
    )

    x_matrix = data_matrix.T
    mean_curve = np.mean(x_matrix, axis=0)
    centered = x_matrix - mean_curve
    _u, singular_values, vt = np.linalg.svd(centered, full_matrices=False)

    loadings = vt[:n_components, :].copy()
    scores = centered @ loadings.T
    scores, loadings = orient_pca_to_concentration(scores, loadings, concentrations)
    trend_coefficients, predicted_scores, trend_degree = fit_score_trends(
        concentrations,
        scores,
        requested_trend_degree,
    )

    predicted_matrix = (mean_curve + predicted_scores @ loadings).T
    residual_matrix = data_matrix - predicted_matrix
    metrics = cls_metrics(data_matrix, predicted_matrix)

    singular_energy = singular_values ** 2
    total_energy = float(np.sum(singular_energy))
    if total_energy > 0:
        full_explained_ratio = singular_energy / total_energy
    else:
        full_explained_ratio = np.full_like(singular_energy, math.nan, dtype=float)
    explained_ratio = full_explained_ratio[:n_components]
    cumulative_explained = float(np.nansum(explained_ratio))
    score_rows = score_correlation_rows(
        concentrations,
        scores,
        predicted_scores,
        explained_ratio,
        trend_coefficients,
    )
    score_summary = summarize_score_correlations(score_rows)
    example = aligned["records"][0]

    return {
        "method": "pcr",
        "scope": example["scope"],
        "technique": example["technique"],
        "technique_label": TECHNIQUES[str(example["technique"])]["label"],
        "sweep": example["sweep"],
        "sweep_label": SWEEPS[str(example["sweep"])],
        "electrode": example["electrode"],
        "x_grid": x_grid,
        "conditions": aligned["conditions"],
        "concentrations": concentrations,
        "data_matrix": data_matrix,
        "raw_data_matrix": aligned["raw_data_matrix"],
        "predicted_matrix": predicted_matrix,
        "residual_matrix": residual_matrix,
        "pca_mean_curve": mean_curve,
        "pca_loadings": loadings,
        "pca_scores": scores,
        "predicted_scores": predicted_scores,
        "score_trend_coefficients": trend_coefficients,
        "score_correlations": score_rows,
        "singular_values": singular_values,
        "explained_variance_ratio": explained_ratio,
        "cumulative_explained_variance_ratio": cumulative_explained,
        "n_components": n_components,
        "requested_n_components": requested_components,
        "score_trend_degree": requested_trend_degree,
        "score_trend_degree_effective": trend_degree,
        "records": aligned["records"],
        "n_voltages": int(n_voltages),
        "n_concentrations": int(n_samples),
        "smooth_window": int(params.get("smooth_window", DEFAULT_SMOOTH_WINDOW)),
        "smooth_polyorder": int(params.get("smooth_polyorder", DEFAULT_SMOOTH_POLYORDER)),
        "actual_smooth_window": int(aligned["actual_smooth_window"]),
        "smooth_polyorder_effective": int(aligned["smooth_polyorder_effective"]),
        "grid_points": int(params.get("grid_points", DEFAULT_GRID_POINTS)),
        **metrics,
        **score_summary,
    }


def fit_pcr_models(
    entries: list[dict[str, object]],
    params: dict[str, object],
    scopes: set[str] | None = None,
) -> list[dict[str, object]]:
    grouped = group_records(entries, ("scope", "technique", "sweep", "electrode"))
    models = []
    for (_scope, _technique, _sweep, _electrode), records in sorted(grouped.items()):
        if scopes is not None and str(_scope) not in scopes:
            continue
        models.append(fit_pcr_group(records, params))
    return models


def build_prediction_model(
    pcr_models: list[dict[str, object]],
    params: dict[str, object],
    scope: str = "mean",
) -> dict[str, object]:
    curves = {}
    for model in pcr_models:
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
            "pca_mean_curve": model["pca_mean_curve"],
            "pca_loadings": model["pca_loadings"],
            "pca_scores": model["pca_scores"],
            "predicted_scores": model["predicted_scores"],
            "score_trend_coefficients": model["score_trend_coefficients"],
            "score_correlations": model["score_correlations"],
            "explained_variance_ratio": model["explained_variance_ratio"],
            "cumulative_explained_variance_ratio": model["cumulative_explained_variance_ratio"],
            "n_components": model["n_components"],
            "score_trend_degree_effective": model["score_trend_degree_effective"],
            "metrics": {
                "rmse_uA": model["rmse_uA"],
                "mae_uA": model["mae_uA"],
                "max_abs_residual_uA": model["max_abs_residual_uA"],
                "normalized_rmse": model["normalized_rmse"],
                "r2": model["r2"],
                "avg_abs_pc_pearson": model["avg_abs_pc_pearson"],
                "avg_abs_pc_spearman": model["avg_abs_pc_spearman"],
                "avg_pc_trend_r2": model["avg_pc_trend_r2"],
            },
        }
    return {
        "method": "pcr",
        "equation": "X ~= mean_curve + PCA_scores(C) @ PCA_loadings; each retained score is fit as a polynomial in concentration",
        "params": params,
        "current_unit": CURRENT_UNIT,
        "curves": curves,
    }


def predict_curves(
    model: dict[str, object],
    concentration_uM: float,
    points: int | None = None,
) -> dict[str, dict[str, object]]:
    predicted = {}
    for curve_key, curve in model["curves"].items():
        x_grid = np.asarray(curve["x_grid"], dtype=float)
        mean_curve = np.asarray(curve["pca_mean_curve"], dtype=float)
        loadings = np.asarray(curve["pca_loadings"], dtype=float)
        trend_coefficients = np.asarray(curve["score_trend_coefficients"], dtype=float)
        if trend_coefficients.ndim == 1:
            trend_coefficients = trend_coefficients.reshape(1, -1)

        if points is None or int(points) <= 0 or int(points) == len(x_grid):
            potential_v = x_grid
            mean_grid = mean_curve
            loading_grid = loadings
        else:
            potential_v = np.linspace(float(np.min(x_grid)), float(np.max(x_grid)), int(points))
            mean_grid = np.interp(potential_v, x_grid, mean_curve)
            loading_grid = np.vstack(
                [np.interp(potential_v, x_grid, loading) for loading in loadings]
            )

        score_values = np.array(
            [np.polyval(coefficients, float(concentration_uM)) for coefficients in trend_coefficients],
            dtype=float,
        )
        current_uA = mean_grid + score_values @ loading_grid
        predicted[curve_key] = {
            "curve": curve_key,
            "technique": curve["technique"],
            "technique_label": curve["technique_label"],
            "sweep": curve["sweep"],
            "sweep_label": curve["sweep_label"],
            "potential_v": potential_v,
            "current_uA": current_uA,
            "method": "pcr",
            "n_components": int(curve["n_components"]),
            "score_trend_degree_effective": int(curve["score_trend_degree_effective"]),
            "pca_scores": score_values,
        }
    return predicted


def write_prediction_csv(
    predicted: dict[str, dict[str, object]],
    concentration: float,
    output_path: Path,
) -> None:
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


__all__ = [
    "CURRENT_UNIT",
    "DEFAULT_GRID_POINTS",
    "DEFAULT_MANIFEST",
    "DEFAULT_N_COMPONENTS",
    "DEFAULT_SCORE_TREND_DEGREE",
    "DEFAULT_SMOOTH_POLYORDER",
    "DEFAULT_SMOOTH_WINDOW",
    "DEFAULT_SUBSET_DIR",
    "PANEL_ORDER",
    "SWEEPS",
    "TECHNIQUES",
    "build_prediction_model",
    "clean_value",
    "fit_pcr_models",
    "format_float",
    "format_series",
    "json_default",
    "load_trace_entries",
    "parse_int_list",
    "predict_curves",
    "read_csv_dicts",
    "read_manifest",
    "slug_float",
    "write_prediction_csv",
]
