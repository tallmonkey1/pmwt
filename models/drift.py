r"""Regularized dynamic drift estimation (SPEC §2.1).

The drift model :math:`\mu_t` is implemented as a small regularized model over slow features.
Drift is deliberately low-weight; for short-dated premium selling the distribution's
shape and tails dominate.

This module provides a :class:`RidgeDriftEstimator` that uses Ridge regression on
historical features to forecast the next-step drift.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..core.errors import ValidationError

__all__ = ["DriftEstimator", "RidgeDriftEstimator"]


class DriftEstimator:
    """Abstract interface for drift estimation."""

    def estimate_drift(self, features: NDArray[np.float64], targets: NDArray[np.float64]) -> float:
        """Estimate the annualized drift from historical features and targets."""
        raise NotImplementedError


class RidgeDriftEstimator(DriftEstimator):
    """Annualized drift estimator using Ridge regression for regularization.

    Parameters
    ----------
    alpha:
        Regularization strength (L2 penalty).
    annualization_factor:
        Factor to annualize the estimated drift (e.g. 252 for daily data).
    """

    def __init__(self, alpha: float = 1.0, annualization_factor: float = 252.0) -> None:
        self._alpha = alpha
        self._annual = annualization_factor
        self._coef: NDArray[np.float64] | None = None
        self._intercept: float = 0.0

    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        """Fit the Ridge model to historical features ``x`` and returns ``y``."""
        if x.shape[0] != y.shape[0]:
            raise ValidationError("x and y must have same number of rows", context={})
        
        # Simple Ridge closed-form: w = (X'X + alpha*I)^-1 X'y
        n_features = x.shape[1]
        x_mean = np.mean(x, axis=0)
        y_mean = np.mean(y)
        x_centered = x - x_mean
        y_centered = y - y_mean
        
        xtx = x_centered.T @ x_centered
        xty = x_centered.T @ y_centered
        
        self._coef = np.linalg.solve(xtx + self._alpha * np.eye(n_features), xty)
        self._intercept = float(y_mean - x_mean @ self._coef)

    def predict(self, x: NDArray[np.float64]) -> float:
        """Predict the next-step drift (annualized)."""
        if self._coef is None:
            return 0.0
        # Return annualized drift
        pred = float(x @ self._coef + self._intercept)
        return pred * self._annual
