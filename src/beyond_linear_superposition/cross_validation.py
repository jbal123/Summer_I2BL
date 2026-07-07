#!/usr/bin/env python3
"""Step 7: leave-one-UA-level-out cross-validation.

UA is the hardest concentration axis, so we test generalization across it
rather than using a random split. Each fold holds out all mixtures at one UA
level, trains the residual model on the rest, and evaluates on the held-out
level. The COW reference and per-analyte PCR models are fixed across folds
(they depend only on isolate data, per the spec's implementation notes).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from residual_model import fit_gpr_residual_model, fit_polynomial_residual_model


def ua_levels(conditions: np.ndarray) -> list[float]:
    return sorted({float(v) for v in np.asarray(conditions, dtype=float)[:, 2]})


@dataclass
class Fold:
    held_out_ua: float
    train_index: np.ndarray
    test_index: np.ndarray


def leave_one_ua_out_folds(conditions: np.ndarray) -> list[Fold]:
    conditions = np.asarray(conditions, dtype=float)
    ua_col = conditions[:, 2]
    folds = []
    for level in ua_levels(conditions):
        test_mask = np.isclose(ua_col, level)
        train_mask = ~test_mask
        if not test_mask.any() or not train_mask.any():
            continue
        folds.append(
            Fold(
                held_out_ua=level,
                train_index=np.where(train_mask)[0],
                test_index=np.where(test_mask)[0],
            )
        )
    return folds


def fit_residual_model(
    conditions: np.ndarray,
    residuals: np.ndarray,
    model_type: str,
    poly_degree: int = 2,
    poly_alpha: float = 1e-3,
    gpr_components: int = 5,
):
    if model_type == "polynomial":
        return fit_polynomial_residual_model(
            conditions, residuals, degree=poly_degree, alpha=poly_alpha
        )
    if model_type == "gpr":
        return fit_gpr_residual_model(conditions, residuals, n_components=gpr_components)
    raise ValueError(f"Unknown residual model type: {model_type!r}")


__all__ = ["ua_levels", "Fold", "leave_one_ua_out_folds", "fit_residual_model"]
