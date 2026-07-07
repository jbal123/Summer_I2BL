#!/usr/bin/env python3
"""Step 6: full prediction pipeline = superposition baseline + residual correction.

Also bundles the per-panel objects (PCR models, COW reference, residual model)
so a single fitted pipeline can be serialized and reused at inference time
exactly as it was fit on training data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pcr_model import AnalytePCRModel
from residual_model import (
    GPRResidualModel,
    PolynomialResidualModel,
    predict_residual,
)
from superposition import superposition_prediction


def predict_mixture_curve(
    c_da: float,
    c_aa: float,
    c_ua: float,
    models: dict[str, AnalytePCRModel],
    residual_model: object | None,
) -> np.ndarray:
    """Superposition baseline + predicted residual correction.

    If ``residual_model`` is None, returns the bare superposition baseline.
    """
    baseline = superposition_prediction(c_da, c_aa, c_ua, models)
    if residual_model is None:
        return baseline
    return baseline + predict_residual(residual_model, c_da, c_aa, c_ua)


@dataclass
class FittedPanelPipeline:
    """Everything needed to predict mixtures for one (technique, sweep) panel."""

    technique: str
    sweep: str
    grid: np.ndarray
    cow_reference: np.ndarray
    preprocess_config: dict
    pcr_models: dict[str, AnalytePCRModel]
    residual_model: object | None
    residual_model_type: str

    def predict(self, c_da: float, c_aa: float, c_ua: float) -> np.ndarray:
        return predict_mixture_curve(
            c_da, c_aa, c_ua, self.pcr_models, self.residual_model
        )

    def predict_baseline(self, c_da: float, c_aa: float, c_ua: float) -> np.ndarray:
        return superposition_prediction(c_da, c_aa, c_ua, self.pcr_models)

    def predict_with_uncertainty(
        self, c_da: float, c_aa: float, c_ua: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (corrected_curve, residual_std). std is zeros for non-GPR models."""
        baseline = self.predict_baseline(c_da, c_aa, c_ua)
        if isinstance(self.residual_model, GPRResidualModel):
            mean_res, std_res = self.residual_model.predict_with_uncertainty(c_da, c_aa, c_ua)
            return baseline + mean_res, std_res
        if isinstance(self.residual_model, PolynomialResidualModel):
            return baseline + self.residual_model.predict(c_da, c_aa, c_ua), np.zeros_like(baseline)
        return baseline, np.zeros_like(baseline)


__all__ = ["predict_mixture_curve", "FittedPanelPipeline"]
