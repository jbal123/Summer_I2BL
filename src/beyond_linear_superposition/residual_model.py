#!/usr/bin/env python3
"""Step 5: residual models mapping (c_DA, c_AA, c_UA) -> residual curve.

Two options, in order of preference for this dataset size:
  5a. Polynomial interaction regression (try first).
  5b. Gaussian-process regression on residual PCA scores (if polynomial underfits).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures


# --------------------------------------------------------------------------- #
# 5a. Polynomial interaction regression
# --------------------------------------------------------------------------- #
@dataclass
class PolynomialResidualModel:
    poly: PolynomialFeatures
    model: Ridge
    degree: int
    alpha: float

    def predict(self, c_da: float, c_aa: float, c_ua: float) -> np.ndarray:
        X = self.poly.transform([[c_da, c_aa, c_ua]])
        return self.model.predict(X)[0]


def fit_polynomial_residual_model(
    conditions: np.ndarray,
    residuals: np.ndarray,
    degree: int = 2,
    alpha: float = 1e-3,
) -> PolynomialResidualModel:
    """Low-degree polynomial in the three concentrations (pairwise + three-way
    interaction terms), fit jointly across all voltage points with Ridge."""
    conditions = np.asarray(conditions, dtype=float)
    residuals = np.asarray(residuals, dtype=float)
    poly = PolynomialFeatures(degree=degree, include_bias=True)
    X = poly.fit_transform(conditions)
    model = Ridge(alpha=alpha)
    model.fit(X, residuals)
    return PolynomialResidualModel(poly=poly, model=model, degree=degree, alpha=alpha)


# --------------------------------------------------------------------------- #
# 5b. Gaussian process regression on residual PCA scores
# --------------------------------------------------------------------------- #
@dataclass
class GPRResidualModel:
    pca: PCA
    gps: list[GaussianProcessRegressor]
    c_min: np.ndarray
    c_range: np.ndarray

    def _normalize(self, c_da: float, c_aa: float, c_ua: float) -> np.ndarray:
        x = (np.array([[c_da, c_aa, c_ua]], dtype=float) - self.c_min) / self.c_range
        return x

    def predict(self, c_da: float, c_aa: float, c_ua: float) -> np.ndarray:
        x = self._normalize(c_da, c_aa, c_ua)
        score_preds = np.array([gp.predict(x)[0] for gp in self.gps])
        return self.pca.inverse_transform(score_preds.reshape(1, -1))[0]

    def predict_with_uncertainty(
        self, c_da: float, c_aa: float, c_ua: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (mean_residual, per-voltage residual std)."""
        x = self._normalize(c_da, c_aa, c_ua)
        means, stds = [], []
        for gp in self.gps:
            mean, std = gp.predict(x, return_std=True)
            means.append(float(mean[0]))
            stds.append(float(std[0]))
        mean_scores = np.array(means)
        std_scores = np.array(stds)
        mean_residual = self.pca.inverse_transform(mean_scores.reshape(1, -1))[0]
        components = self.pca.components_  # (n_components, n_voltage_points)
        residual_std = np.sqrt((std_scores[:, None] ** 2 * components ** 2).sum(axis=0))
        return mean_residual, residual_std


def fit_gpr_residual_model(
    conditions: np.ndarray,
    residuals: np.ndarray,
    n_components: int = 5,
) -> GPRResidualModel:
    """Compress residuals with PCA, fit one GP per retained component.

    Concentrations are min-max normalized to [0, 1] using the training set only,
    so the kernel length-scales are interpretable.
    """
    conditions = np.asarray(conditions, dtype=float)
    residuals = np.asarray(residuals, dtype=float)

    n_comp = min(n_components, residuals.shape[0], residuals.shape[1])
    pca = PCA(n_components=n_comp)
    pca.fit(residuals)
    scores = pca.transform(residuals)

    c_min = conditions.min(axis=0)
    c_range = conditions.max(axis=0) - c_min
    c_range[c_range == 0] = 1.0  # guard against constant columns
    X = (conditions - c_min) / c_range

    gps = []
    for k in range(n_comp):
        kernel = (
            ConstantKernel(1.0) * RBF(length_scale=[0.3, 0.3, 0.3]) + WhiteKernel(1e-3)
        )
        gp = GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=5, normalize_y=True
        )
        gp.fit(X, scores[:, k])
        gps.append(gp)

    return GPRResidualModel(pca=pca, gps=gps, c_min=c_min, c_range=c_range)


def predict_residual(
    residual_model: object,
    c_da: float,
    c_aa: float,
    c_ua: float,
) -> np.ndarray:
    """Dispatch to whichever residual model type was supplied."""
    if isinstance(residual_model, (PolynomialResidualModel, GPRResidualModel)):
        return residual_model.predict(c_da, c_aa, c_ua)
    raise TypeError(f"Unknown residual model type: {type(residual_model)!r}")


__all__ = [
    "PolynomialResidualModel",
    "GPRResidualModel",
    "fit_polynomial_residual_model",
    "fit_gpr_residual_model",
    "predict_residual",
]
