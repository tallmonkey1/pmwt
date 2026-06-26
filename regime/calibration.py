r"""Probability calibration for regime classification (SPEC §2.6).

A model can be accurate yet *miscalibrated*: when it says "80% chance of low-vol" it should
be right 80% of the time. The trade gate keys off calibrated probabilities, so calibration
is not optional -- an over-confident model would trade through risk it claims not to see.

This module implements **temperature scaling** (Guo et al., 2017): a single scalar ``T``
divides the logits before the softmax, fit by minimizing multiclass log-loss on a held-out
set. It is the simplest calibration method that provably preserves the arg-max prediction
(so accuracy is unchanged) while correcting over/under-confidence. Temperature is optimized
by a robust 1-D golden-section search on the (convex) log-loss, requiring no autograd.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar

from ..core.errors import ModelStateError, ValidationError
from ..core.validation import check_array_finite

__all__ = ["TemperatureScaler"]

_EPS = 1e-12


def _to_logits(probabilities: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert probabilities to logits (log-probabilities), clipped for stability."""
    clipped = np.clip(probabilities, _EPS, 1.0)
    return np.log(clipped)


def _softmax(logits: NDArray[np.float64]) -> NDArray[np.float64]:
    """Numerically-stable row-wise softmax."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return np.asarray(exp / exp.sum(axis=1, keepdims=True), dtype=np.float64)


@dataclass
class TemperatureScaler:
    """Single-parameter temperature scaling for multiclass probabilities.

    Fit on a validation set of (predicted probabilities, true labels); applies the learned
    temperature to new probability vectors. ``temperature_ > 1`` softens over-confident
    predictions; ``< 1`` sharpens under-confident ones.
    """

    temperature_: float | None = None

    @property
    def is_fitted(self) -> bool:
        """True once a temperature has been fit."""
        return self.temperature_ is not None

    def fit(
        self,
        probabilities: NDArray[np.float64],
        labels: NDArray[np.int_],
        *,
        max_temperature: float = 100.0,
    ) -> TemperatureScaler:
        """Fit the temperature by minimizing multiclass log-loss.

        Parameters
        ----------
        probabilities:
            Predicted class probabilities, shape ``(N, K)`` (rows sum to ~1).
        labels:
            True class indices, shape ``(N,)`` with values in ``[0, K)``.
        max_temperature:
            Upper bound for the search bracket.
        """
        probs = np.asarray(probabilities, dtype=np.float64)
        y = np.asarray(labels)
        check_array_finite(probs, name="probabilities")
        if probs.ndim != 2 or probs.shape[0] < 1:
            raise ValidationError(
                "probabilities must be 2-D (N, K)", context={"shape": probs.shape}
            )
        if y.shape != (probs.shape[0],):
            raise ValidationError(
                "labels must have shape (N,)", context={"labels": y.shape, "probs": probs.shape}
            )
        if y.min() < 0 or y.max() >= probs.shape[1]:
            raise ValidationError("labels out of range for the number of classes", context={})

        logits = _to_logits(probs)
        n, _ = probs.shape
        rows = np.arange(n)

        def neg_log_likelihood(log_temp: float) -> float:
            temperature = float(np.exp(log_temp))
            scaled = _softmax(logits / temperature)
            picked = np.clip(scaled[rows, y], _EPS, 1.0)
            return float(-np.mean(np.log(picked)))

        # Optimize in log-temperature space for a well-conditioned, positive temperature.
        result = minimize_scalar(
            neg_log_likelihood,
            bounds=(np.log(1e-2), np.log(max_temperature)),
            method="bounded",
        )
        self.temperature_ = float(np.exp(result.x))
        return self

    def transform(self, probabilities: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply the fitted temperature to a probability matrix."""
        if self.temperature_ is None:
            raise ModelStateError("TemperatureScaler must be fit before transform", context={})
        probs = np.asarray(probabilities, dtype=np.float64)
        if probs.ndim != 2:
            raise ValidationError(
                "probabilities must be 2-D (N, K)", context={"shape": probs.shape}
            )
        logits = _to_logits(probs)
        return _softmax(logits / self.temperature_)

    def state_dict(self) -> dict[str, float]:
        """Return a serializable snapshot."""
        if self.temperature_ is None:
            raise ModelStateError("cannot serialize an unfitted scaler", context={})
        return {"temperature": self.temperature_}

    @classmethod
    def from_state_dict(cls, state: dict[str, float]) -> TemperatureScaler:
        """Reconstruct a scaler from :meth:`state_dict`."""
        return cls(temperature_=float(state["temperature"]))
