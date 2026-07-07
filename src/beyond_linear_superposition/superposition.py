#!/usr/bin/env python3
"""Steps 3-4: linear superposition baseline and training-residual computation."""

from __future__ import annotations

import numpy as np

from pcr_model import AnalytePCRModel


def superposition_prediction(
    c_da: float,
    c_aa: float,
    c_ua: float,
    models: dict[str, AnalytePCRModel],
) -> np.ndarray:
    """Naive linear superposition: sum of the three per-analyte PCR curves.

    This is the baseline that the residual model corrects. Analytes missing a
    fitted model (e.g. too few isolates) contribute zero.
    """
    total = None
    for analyte, conc in (("DA", c_da), ("AA", c_aa), ("UA", c_ua)):
        model = models.get(analyte)
        if model is None:
            continue
        curve = model.predict_curve(conc)
        total = curve if total is None else total + curve
    if total is None:
        raise ValueError("No per-analyte models available for superposition")
    return total


def compute_residuals(
    conditions: np.ndarray,
    real_curves: np.ndarray,
    models: dict[str, AnalytePCRModel],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute superposition baselines and residuals for mixture conditions.

    conditions: (n_mixtures, 3) columns [c_DA, c_AA, c_UA].
    real_curves: (n_mixtures, n_voltage_points), preprocessed.
    Returns (baselines, residuals) both shaped like ``real_curves``.
    """
    conditions = np.asarray(conditions, dtype=float)
    real_curves = np.asarray(real_curves, dtype=float)
    baselines = np.vstack(
        [superposition_prediction(row[0], row[1], row[2], models) for row in conditions]
    )
    residuals = real_curves - baselines
    return baselines, residuals


__all__ = ["superposition_prediction", "compute_residuals"]
