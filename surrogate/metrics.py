r"""Distributional accuracy and calibration metrics for the surrogate.

These quantify how well the surrogate distribution matches the Monte-Carlo ground truth.
They drive both training diagnostics and the runtime fallback guardrail (SPEC §2.5).

* **Quantile loss** -- the training objective, also a clean scalar accuracy summary.
* **CRPS** (Continuous Ranked Probability Score) -- the standard proper score for a full
  predictive distribution; computed here from a dense quantile grid via its
  quantile-integral form, which equals twice the integral of the pinball loss over levels.
* **Wasserstein-1 distance** between two quantile functions -- the integral of the absolute
  difference of their inverse CDFs, a direct, units-of-return measure of distributional
  discrepancy used as the fallback trigger.
* **PIT calibration** -- the probability-integral-transform histogram; a perfectly
  calibrated model has uniform PIT values. We summarize the deviation from uniformity as a
  single coverage-error statistic.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..core.errors import ValidationError
from ..core.validation import check_array_finite

__all__ = [
    "calibration_error",
    "crps_from_quantiles",
    "pit_values",
    "quantile_loss_numpy",
    "wasserstein1_from_quantiles",
]


def _validate_quantiles(
    levels: NDArray[np.float64], values: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    lv = np.asarray(levels, dtype=np.float64)
    vv = np.asarray(values, dtype=np.float64)
    check_array_finite(lv, name="quantile_levels")
    check_array_finite(vv, name="quantile_values")
    if lv.ndim != 1 or vv.ndim != 1 or lv.size != vv.size or lv.size < 2:
        raise ValidationError(
            "levels and values must be 1-D of equal length >= 2",
            context={"levels": lv.shape, "values": vv.shape},
        )
    if np.any(lv <= 0.0) or np.any(lv >= 1.0) or not np.all(np.diff(lv) > 0.0):
        raise ValidationError("levels must be strictly increasing within (0, 1)", context={})
    return lv, vv


def quantile_loss_numpy(
    predicted: NDArray[np.float64],
    target: NDArray[np.float64],
    levels: NDArray[np.float64],
) -> float:
    """Return the mean pinball loss between two quantile vectors (NumPy, for evaluation)."""
    lv, pred = _validate_quantiles(levels, predicted)
    _, tgt = _validate_quantiles(levels, target)
    errors = tgt - pred
    loss = np.maximum(lv * errors, (lv - 1.0) * errors)
    return float(np.mean(loss))


def crps_from_quantiles(
    predicted: NDArray[np.float64],
    target_samples: NDArray[np.float64],
    levels: NDArray[np.float64],
) -> float:
    r"""Approximate the CRPS of a quantile-defined forecast against an empirical sample.

    Uses the identity that CRPS equals twice the integral over levels of the pinball loss of
    the predicted quantiles evaluated against the sample. Concretely, for each level we
    average the pinball loss over the sample, then integrate across levels by the trapezoid
    rule. Lower is better; CRPS has the units of the variable (log-return).
    """
    lv, pred = _validate_quantiles(levels, predicted)
    samples = np.asarray(target_samples, dtype=np.float64)
    check_array_finite(samples, name="target_samples")
    if samples.ndim != 1 or samples.size < 2:
        raise ValidationError("target_samples must be 1-D with length >= 2", context={})

    # Per-level mean pinball loss against the empirical sample.
    per_level = np.empty(lv.size, dtype=np.float64)
    for i, (tau, q) in enumerate(zip(lv, pred, strict=True)):
        errors = samples - q
        per_level[i] = np.mean(np.maximum(tau * errors, (tau - 1.0) * errors))
    return float(2.0 * np.trapezoid(per_level, lv))


def wasserstein1_from_quantiles(
    quantiles_a: NDArray[np.float64],
    quantiles_b: NDArray[np.float64],
    levels: NDArray[np.float64],
) -> float:
    r"""Return the Wasserstein-1 distance between two quantile functions on a shared grid.

    :math:`W_1 = \int_0^1 |Q_a(p) - Q_b(p)|\, dp`, approximated by the trapezoid rule over
    the level grid. This is the runtime surrogate-vs-MC discrepancy measure.
    """
    lv, qa = _validate_quantiles(levels, quantiles_a)
    _, qb = _validate_quantiles(levels, quantiles_b)
    return float(np.trapezoid(np.abs(qa - qb), lv))


def pit_values(
    samples: NDArray[np.float64],
    quantiles: NDArray[np.float64],
    levels: NDArray[np.float64],
) -> NDArray[np.float64]:
    r"""Return the probability-integral-transform values of ``samples`` under the forecast.

    Each sample is mapped through the forecast CDF (the inverse of the quantile function),
    obtained by interpolating ``levels`` against ``quantiles``. For a well-calibrated
    forecast the PIT values are uniform on ``[0, 1]``.
    """
    lv, qv = _validate_quantiles(levels, quantiles)
    s = np.asarray(samples, dtype=np.float64)
    check_array_finite(s, name="samples")
    if s.ndim != 1 or s.size < 1:
        raise ValidationError("samples must be a non-empty 1-D array", context={})
    return np.interp(s, qv, lv, left=0.0, right=1.0)


def calibration_error(
    samples: NDArray[np.float64],
    quantiles: NDArray[np.float64],
    levels: NDArray[np.float64],
) -> float:
    r"""Return a scalar calibration error: mean absolute coverage deviation.

    For each nominal level ``tau`` we measure the empirical coverage -- the fraction of
    samples at or below the predicted ``tau``-quantile -- and average ``|coverage - tau|``.
    Zero means perfectly calibrated; this is the single number used to flag calibration
    drift.
    """
    lv, qv = _validate_quantiles(levels, quantiles)
    s = np.asarray(samples, dtype=np.float64)
    check_array_finite(s, name="samples")
    if s.ndim != 1 or s.size < 1:
        raise ValidationError("samples must be a non-empty 1-D array", context={})
    empirical_coverage = np.array([np.mean(s <= q) for q in qv])
    return float(np.mean(np.abs(empirical_coverage - lv)))
