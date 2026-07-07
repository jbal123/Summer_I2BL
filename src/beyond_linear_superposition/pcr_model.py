#!/usr/bin/env python3
"""Step 2: per-analyte PCR model (standalone, per the spec).

One ``AnalytePCRModel`` instance per analyte (DA, AA, UA), fit on preprocessed
isolate curves. After fitting, ``predict_curve(c)`` reconstructs the isolated
single-analyte CV curve at concentration ``c`` (uM) on the common voltage grid.
"""

from __future__ import annotations

import numpy as np
from numpy.polynomial import polynomial as P
from sklearn.decomposition import PCA


class AnalytePCRModel:
    """PCA on isolate curves + polynomial concentration->score regression."""

    def __init__(self, n_components: int = 5, score_degree: int = 2):
        self.n_components = n_components
        self.score_degree = score_degree
        self.pca: PCA | None = None
        self.mean_curve: np.ndarray | None = None
        self.coefs: list[np.ndarray] = []
        self.n_components_effective: int = 0
        self.training_concentrations: np.ndarray | None = None

    def fit(self, concentrations: np.ndarray, curves: np.ndarray) -> "AnalytePCRModel":
        """concentrations: (n_samples,); curves: (n_samples, n_voltage_points),
        preprocessed (AsLSSR + COW)."""
        concentrations = np.asarray(concentrations, dtype=float)
        curves = np.asarray(curves, dtype=float)
        if curves.ndim != 2:
            raise ValueError("curves must be 2-D (n_samples, n_voltage_points)")
        n_samples = curves.shape[0]
        if n_samples < 2:
            raise ValueError("AnalytePCRModel requires at least two isolate curves")

        self.training_concentrations = concentrations
        self.mean_curve = curves.mean(axis=0)
        centered = curves - self.mean_curve

        # PCA cannot retain more components than (n_samples - 1).
        n_comp = min(self.n_components, n_samples - 1, curves.shape[1])
        self.n_components_effective = n_comp
        self.pca = PCA(n_components=n_comp)
        scores = self.pca.fit_transform(centered)  # (n_samples, n_comp)

        # Polynomial degree is capped by the number of distinct concentrations.
        unique = len(np.unique(concentrations))
        degree = max(1, min(self.score_degree, unique - 1, n_samples - 1))
        self.coefs = [
            P.polyfit(concentrations, scores[:, k], deg=degree) for k in range(n_comp)
        ]
        return self

    def predict_scores(self, concentration: float) -> np.ndarray:
        return np.array([P.polyval(float(concentration), c) for c in self.coefs])

    def predict_curve(self, concentration: float) -> np.ndarray:
        """Reconstruct the isolated-analyte CV curve at ``concentration`` (uM)."""
        if self.pca is None or self.mean_curve is None:
            raise RuntimeError("AnalytePCRModel must be fit before prediction")
        scores = self.predict_scores(concentration)
        return self.mean_curve + self.pca.inverse_transform(scores.reshape(1, -1))[0]


def fit_analyte_models(
    isolate_curves: dict[str, list[tuple[float, np.ndarray]]],
    n_components: int = 5,
    score_degree: int = 2,
) -> dict[str, AnalytePCRModel]:
    """Fit one AnalytePCRModel per analyte from preprocessed isolate curves."""
    models: dict[str, AnalytePCRModel] = {}
    for analyte, samples in isolate_curves.items():
        if len(samples) < 2:
            continue
        concentrations = np.array([c for c, _ in samples], dtype=float)
        curves = np.vstack([curve for _, curve in samples])
        models[analyte] = AnalytePCRModel(
            n_components=n_components, score_degree=score_degree
        ).fit(concentrations, curves)
    return models


__all__ = ["AnalytePCRModel", "fit_analyte_models"]
