r"""Calibration and skill metrics for probabilistic regime forecasts (SPEC §2.6).

The spec is explicit: "No 'high accuracy' claim ships without these numbers." This module
provides the standard proper scores and calibration diagnostics, so any regime model's skill
is reported honestly rather than asserted:

* **Multiclass Brier score** -- mean squared error between the predicted probability vector
  and the one-hot truth; a strictly proper score (lower is better).
* **Log-loss** (cross-entropy) -- the other canonical strictly-proper score.
* **Reliability (calibration) curve** -- binned predicted-vs-empirical frequency for a
  chosen class; the basis of the reliability diagram.
* **Expected Calibration Error (ECE)** -- the average gap between confidence and accuracy
  over confidence bins; a single-number calibration summary used by the promotion gates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..core.errors import ValidationError
from ..core.validation import check_array_finite

__all__ = [
    "ReliabilityCurve",
    "brier_score",
    "expected_calibration_error",
    "log_loss",
    "reliability_curve",
]

_EPS = 1e-12


def _validate_probs_labels(
    probabilities: NDArray[np.float64], labels: NDArray[np.int_]
) -> tuple[NDArray[np.float64], NDArray[np.int_]]:
    probs = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(labels)
    check_array_finite(probs, name="probabilities")
    if probs.ndim != 2 or probs.shape[0] < 1:
        raise ValidationError("probabilities must be 2-D (N, K)", context={"shape": probs.shape})
    if y.shape != (probs.shape[0],):
        raise ValidationError("labels must have shape (N,)", context={"labels": y.shape})
    if y.min() < 0 or y.max() >= probs.shape[1]:
        raise ValidationError("labels out of range", context={"n_classes": probs.shape[1]})
    if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-3):
        raise ValidationError("probability rows must sum to 1", context={})
    return probs, y


def brier_score(probabilities: NDArray[np.float64], labels: NDArray[np.int_]) -> float:
    """Return the multiclass Brier score (mean squared error vs. one-hot truth)."""
    probs, y = _validate_probs_labels(probabilities, labels)
    n, k = probs.shape
    onehot = np.zeros((n, k), dtype=np.float64)
    onehot[np.arange(n), y] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def log_loss(probabilities: NDArray[np.float64], labels: NDArray[np.int_]) -> float:
    """Return the multiclass log-loss (cross-entropy)."""
    probs, y = _validate_probs_labels(probabilities, labels)
    picked = np.clip(probs[np.arange(probs.shape[0]), y], _EPS, 1.0)
    return float(-np.mean(np.log(picked)))


def expected_calibration_error(
    probabilities: NDArray[np.float64],
    labels: NDArray[np.int_],
    *,
    n_bins: int = 10,
) -> float:
    r"""Return the Expected Calibration Error over confidence bins.

    For each sample we take the predicted top-class confidence and whether the top-class
    prediction was correct. Samples are bucketed by confidence; ECE is the sample-weighted
    average ``|confidence - accuracy|`` across buckets. Zero means perfect calibration.
    """
    probs, y = _validate_probs_labels(probabilities, labels)
    if n_bins < 1:
        raise ValidationError("n_bins must be >= 1", context={"n_bins": n_bins})
    confidence = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == y).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = probs.shape[0]
    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        # Include the right edge in the final bin.
        in_bin = (
            (confidence > lo) & (confidence <= hi)
            if b > 0
            else (confidence >= lo) & (confidence <= hi)
        )
        count = int(np.count_nonzero(in_bin))
        if count == 0:
            continue
        avg_conf = float(np.mean(confidence[in_bin]))
        avg_acc = float(np.mean(correct[in_bin]))
        ece += (count / n) * abs(avg_conf - avg_acc)
    return ece


@dataclass(frozen=True, slots=True)
class ReliabilityCurve:
    """Binned reliability data for one class (for reliability diagrams)."""

    bin_confidence: NDArray[np.float64]
    bin_accuracy: NDArray[np.float64]
    bin_count: NDArray[np.int_]


def reliability_curve(
    probabilities: NDArray[np.float64],
    labels: NDArray[np.int_],
    *,
    class_index: int,
    n_bins: int = 10,
) -> ReliabilityCurve:
    """Return the reliability curve for a single class.

    For the chosen class, bins the predicted probability and reports, per bin, the mean
    predicted probability and the empirical frequency of that class. A perfectly calibrated
    model lies on the diagonal.
    """
    probs, y = _validate_probs_labels(probabilities, labels)
    if not 0 <= class_index < probs.shape[1]:
        raise ValidationError("class_index out of range", context={"class_index": class_index})
    if n_bins < 1:
        raise ValidationError("n_bins must be >= 1", context={"n_bins": n_bins})

    p = probs[:, class_index]
    is_class = (y == class_index).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    conf = np.full(n_bins, np.nan)
    acc = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=np.int_)
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        in_bin = (p > lo) & (p <= hi) if b > 0 else (p >= lo) & (p <= hi)
        count = int(np.count_nonzero(in_bin))
        counts[b] = count
        if count > 0:
            conf[b] = float(np.mean(p[in_bin]))
            acc[b] = float(np.mean(is_class[in_bin]))
    return ReliabilityCurve(bin_confidence=conf, bin_accuracy=acc, bin_count=counts)
