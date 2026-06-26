r"""Realized-variance and jump-robust volatility estimators.

These transform observed price/return series into the volatility proxies that the
parameter estimators consume. The two workhorses are:

* **Realized variance (RV)** -- the sum of squared intraday log-returns over a window. It
  consistently estimates integrated variance *plus* the contribution of any jumps.
* **Bipower variation (BV)** -- a jump-robust estimator (Barndorff-Nielsen & Shephard,
  2004) that consistently estimates integrated variance *excluding* jumps:

  .. math::

      BV = \frac{\pi}{2}\, \frac{n}{n-1}\sum_{i=2}^{n} |r_i|\,|r_{i-1}|.

The gap ``RV - BV`` is the basis of the jump test (:mod:`options_engine.calibration.jumps`):
under the no-jump null it converges to zero.

All functions are vectorized, validated, and annualization-aware so the downstream
estimators receive consistent, comparable volatility proxies.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..core.errors import ValidationError
from ..core.timegrid import TRADING_DAYS_PER_YEAR
from ..core.validation import check_array_finite

__all__ = [
    "bipower_variation",
    "daily_realized_variance",
    "log_returns",
    "log_variance_proxy",
    "realized_variance",
]

# (pi / 2) is the asymptotic scaling constant for bipower variation.
_BV_SCALE = np.pi / 2.0


def log_returns(prices: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return log-returns ``log(P_t / P_{t-1})`` from a positive price series."""
    p = np.asarray(prices, dtype=np.float64)
    check_array_finite(p, name="prices")
    if p.ndim != 1 or p.size < 2:
        raise ValidationError(
            "prices must be a 1-D series with at least two points", context={"size": int(p.size)}
        )
    if np.any(p <= 0.0):
        raise ValidationError("prices must be strictly positive", context={})
    return np.diff(np.log(p))


def realized_variance(returns: NDArray[np.float64]) -> float:
    """Return the realized variance (sum of squared returns) over the window."""
    r = np.asarray(returns, dtype=np.float64)
    check_array_finite(r, name="returns")
    if r.size == 0:
        raise ValidationError("returns must be non-empty", context={})
    return float(np.sum(r**2))


def bipower_variation(returns: NDArray[np.float64]) -> float:
    r"""Return the (jump-robust) bipower variation over the window.

    Requires at least two returns. Estimates integrated variance while being asymptotically
    unaffected by finite-activity jumps, because the product of adjacent absolute returns
    suppresses the impact of any single large jump.
    """
    r = np.asarray(returns, dtype=np.float64)
    check_array_finite(r, name="returns")
    n = r.size
    if n < 2:
        raise ValidationError(
            "bipower variation needs at least two returns", context={"n": int(n)}
        )
    abs_r = np.abs(r)
    raw = float(np.sum(abs_r[1:] * abs_r[:-1]))
    # Small-sample bias correction factor n / (n - 1).
    return _BV_SCALE * (n / (n - 1.0)) * raw


def daily_realized_variance(
    intraday_returns: NDArray[np.float64],
    *,
    steps_per_day: int,
    annualize: bool = True,
    trading_days_per_year: int = TRADING_DAYS_PER_YEAR,
) -> NDArray[np.float64]:
    """Aggregate intraday returns into a per-day realized-variance series.

    Parameters
    ----------
    intraday_returns:
        Flat array of intraday log-returns whose length is a multiple of ``steps_per_day``.
    steps_per_day:
        Number of intraday return observations per trading day.
    annualize:
        If True, multiply each daily RV by ``trading_days_per_year`` so the series is in
        annualized-variance units (matching the model's ``xi_0`` convention).

    Returns
    -------
    numpy.ndarray
        One realized-variance value per trading day.
    """
    r = np.asarray(intraday_returns, dtype=np.float64)
    check_array_finite(r, name="intraday_returns")
    if steps_per_day < 1:
        raise ValidationError(
            "steps_per_day must be >= 1", context={"steps_per_day": steps_per_day}
        )
    if r.size == 0 or r.size % steps_per_day != 0:
        raise ValidationError(
            "intraday_returns length must be a positive multiple of steps_per_day",
            context={"size": int(r.size), "steps_per_day": steps_per_day},
        )
    n_days = r.size // steps_per_day
    daily = np.sum(r.reshape(n_days, steps_per_day) ** 2, axis=1)
    if annualize:
        daily = daily * trading_days_per_year
    return np.asarray(daily, dtype=np.float64)


def log_variance_proxy(variance: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return ``log(variance)`` after validating strict positivity.

    The Hurst estimator operates on log-variance (equivalently ``2 * log-vol``). Realized
    variance can be exactly zero in a flat window, which would produce ``-inf``; we reject
    non-positive values explicitly so the caller fixes the upstream aggregation rather than
    silently propagating infinities.
    """
    v = np.asarray(variance, dtype=np.float64)
    check_array_finite(v, name="variance")
    if v.size == 0:
        raise ValidationError("variance series must be non-empty", context={})
    if np.any(v <= 0.0):
        raise ValidationError(
            "variance must be strictly positive to take a log proxy",
            context={"min": float(v.min())},
        )
    return np.log(v)
