#!/usr/bin/env python3
"""Step 8: evaluation metrics and plotting.

Metrics are computed for both the superposition baseline and the corrected
prediction so the improvement over baseline (the primary claim) is explicit.
"""

from __future__ import annotations

import numpy as np

# Default per-analyte peak windows (volts). Tune to your electrode if needed.
DEFAULT_PEAK_WINDOWS = {
    "AA": (-0.05, 0.10),
    "DA": (0.10, 0.25),
    "UA": (0.25, 0.45),
}


def evaluate_curve_reconstruction(
    predicted: np.ndarray,
    actual: np.ndarray,
    voltage_grid: np.ndarray,
    peak_windows: dict[str, tuple[float, float]] | None = None,
) -> dict[str, float]:
    predicted = np.asarray(predicted, dtype=float)
    actual = np.asarray(actual, dtype=float)
    residual = actual - predicted
    rmse_global = float(np.sqrt(np.mean(residual ** 2)))
    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    r2_global = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    results = {"rmse_global": rmse_global, "r2_global": r2_global}

    if peak_windows:
        for name, (v_lo, v_hi) in peak_windows.items():
            mask = (voltage_grid >= v_lo) & (voltage_grid <= v_hi)
            if not mask.any():
                results[f"rmse_{name}"] = float("nan")
                results[f"r2_{name}"] = float("nan")
                continue
            r_peak = residual[mask]
            a_peak = actual[mask]
            results[f"rmse_{name}"] = float(np.sqrt(np.mean(r_peak ** 2)))
            ss_res_p = float(np.sum(r_peak ** 2))
            ss_tot_p = float(np.sum((a_peak - a_peak.mean()) ** 2))
            results[f"r2_{name}"] = 1.0 - ss_res_p / ss_tot_p if ss_tot_p > 0 else float("nan")

    return results


def aggregate_metrics(per_condition: list[dict[str, float]]) -> dict[str, float]:
    """Mean of each metric across conditions, ignoring non-finite values."""
    if not per_condition:
        return {}
    keys = per_condition[0].keys()
    out = {}
    for key in keys:
        values = [row[key] for row in per_condition if np.isfinite(row.get(key, np.nan))]
        out[key] = float(np.mean(values)) if values else float("nan")
    return out


def improvement_summary(
    baseline_metrics: dict[str, float],
    corrected_metrics: dict[str, float],
) -> dict[str, float]:
    """Fractional RMSE reduction (positive = correction helps)."""
    out = {}
    for key in baseline_metrics:
        if not key.startswith("rmse"):
            continue
        base = baseline_metrics.get(key, float("nan"))
        corr = corrected_metrics.get(key, float("nan"))
        if np.isfinite(base) and base > 0 and np.isfinite(corr):
            out[f"{key}_reduction_frac"] = (base - corr) / base
        else:
            out[f"{key}_reduction_frac"] = float("nan")
    return out


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def plot_condition(ax, voltage, actual, baseline, corrected, title, *, residual_std=None, unit="uA"):
    ax.plot(voltage, actual, color="#000000", lw=1.2, label="actual mixture")
    ax.plot(voltage, baseline, color="#D55E00", lw=1.2, ls="--", label="superposition")
    ax.plot(voltage, corrected, color="#0072B2", lw=1.4, label="corrected")
    if residual_std is not None and np.any(residual_std > 0):
        ax.fill_between(
            voltage,
            corrected - residual_std,
            corrected + residual_std,
            color="#0072B2",
            alpha=0.18,
            lw=0,
            label="±1σ residual",
        )
    ax.set_title(title, fontsize=8)
    ax.set_xlabel("Potential (V)", fontsize=7)
    ax.set_ylabel(f"Current ({unit})", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


__all__ = [
    "DEFAULT_PEAK_WINDOWS",
    "evaluate_curve_reconstruction",
    "aggregate_metrics",
    "improvement_summary",
    "plot_condition",
]
