r"""Forward-variance curve (:math:`\xi_0`) estimation.

The forward-variance curve is the model's term structure of variance and, under the pricing
measure, is what option markets quote through the implied-variance term structure
(SPEC §2.4). This module provides two estimators:

* :func:`estimate_xi0_level` -- a single flat level from a realized-variance history (the
  robust default when only historical underlying data is available). It is the mean
  realized variance, which is the maximum-likelihood / method-of-moments estimate of the
  constant forward-variance level.

* :func:`estimate_xi0_curve` -- a piecewise-linear curve from an at-the-money implied-vol
  term structure. For maturities :math:`T_i` with ATM implied vols :math:`\sigma_i`, the
  forward variance over each interval is the *difference of total implied variances*,

  .. math::

      \xi_0\big([T_{i-1}, T_i]\big) = \frac{\sigma_i^2 T_i - \sigma_{i-1}^2 T_{i-1}}
                                            {T_i - T_{i-1}},

  i.e. the slope of the total-variance term structure, which is exactly the forward
  variance implied by no-arbitrage. The result is returned as a
  :class:`~options_engine.models.rbergomi.ForwardVariance` ready for the simulator.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..core.errors import CalibrationError, ValidationError
from ..core.validation import check_array_finite
from ..models.rbergomi import ForwardVariance
from .results import ParameterEstimate

__all__ = ["estimate_xi0_curve", "estimate_xi0_level"]


def estimate_xi0_level(realized_variance_series: NDArray[np.float64]) -> ParameterEstimate:
    r"""Estimate a flat forward-variance level from a realized-variance history.

    Parameters
    ----------
    realized_variance_series:
        Series of (annualized) realized-variance observations. Must be strictly positive.

    Returns
    -------
    ParameterEstimate
        The mean realized variance as the ``xi0`` level, with the standard error of the
        mean.
    """
    rv = np.asarray(realized_variance_series, dtype=np.float64)
    check_array_finite(rv, name="realized_variance_series")
    if rv.ndim != 1 or rv.size < 2:
        raise CalibrationError(
            "need at least two realized-variance observations", context={"size": int(rv.size)}
        )
    if np.any(rv <= 0.0):
        raise ValidationError("realized variance must be strictly positive", context={})

    level = float(np.mean(rv))
    std_error = float(np.std(rv, ddof=1) / np.sqrt(rv.size))
    return ParameterEstimate(
        name="xi0_level", value=level, std_error=std_error, n_observations=int(rv.size)
    )


def estimate_xi0_curve(
    maturities: NDArray[np.float64],
    atm_implied_vols: NDArray[np.float64],
) -> ForwardVariance:
    r"""Build a piecewise-linear forward-variance curve from an ATM IV term structure.

    Parameters
    ----------
    maturities:
        Strictly increasing, strictly positive maturities in years.
    atm_implied_vols:
        At-the-money implied volatilities at each maturity (strictly positive). Same length
        as ``maturities``.

    Returns
    -------
    ForwardVariance
        A curve whose value over each maturity interval equals the no-arbitrage forward
        variance implied by the total-variance term structure.

    Raises
    ------
    CalibrationError
        If the total-variance term structure is not non-decreasing (which would imply a
        negative forward variance -- a calendar-spread arbitrage).
    """
    t = np.asarray(maturities, dtype=np.float64)
    sigma = np.asarray(atm_implied_vols, dtype=np.float64)
    check_array_finite(t, name="maturities")
    check_array_finite(sigma, name="atm_implied_vols")
    if t.ndim != 1 or sigma.ndim != 1 or t.size != sigma.size:
        raise ValidationError(
            "maturities and atm_implied_vols must be 1-D arrays of equal length",
            context={"t": t.shape, "sigma": sigma.shape},
        )
    if t.size < 1:
        raise ValidationError("need at least one maturity", context={})
    if np.any(t <= 0.0):
        raise ValidationError("maturities must be strictly positive", context={})
    if np.any(sigma <= 0.0):
        raise ValidationError("implied vols must be strictly positive", context={})
    if t.size > 1 and not np.all(np.diff(t) > 0.0):
        raise ValidationError("maturities must be strictly increasing", context={})

    total_variance = sigma**2 * t  # w(T) = sigma^2 * T

    # Forward variance over [0, T_0] is the average variance to the first maturity.
    knot_times = np.concatenate(([0.0], t))
    knot_values = np.empty(knot_times.size, dtype=np.float64)
    knot_values[0] = total_variance[0] / t[0]  # average variance on the first interval

    prev_w = 0.0
    prev_t = 0.0
    for i in range(t.size):
        dw = total_variance[i] - prev_w
        dtau = t[i] - prev_t
        fwd = dw / dtau
        if fwd <= 0.0:
            raise CalibrationError(
                "non-positive forward variance (calendar arbitrage in IV term structure)",
                context={"interval_end": float(t[i]), "forward_variance": float(fwd)},
            )
        knot_values[i + 1] = fwd
        prev_w = total_variance[i]
        prev_t = t[i]

    # The first knot value should equal the first forward variance for a clean t=0 anchor.
    knot_values[0] = knot_values[1]
    return ForwardVariance(knot_times=knot_times, knot_values=knot_values)
