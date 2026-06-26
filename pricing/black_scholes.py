r"""Black-Scholes-Merton analytic pricing and Greeks.

The rBergomi model has no closed-form option prices, so the engine prices with Monte Carlo
(see :mod:`options_engine.pricing.monte_carlo`). Black-Scholes is nonetheless essential as:

* an **independent validation oracle** -- in the ``eta -> 0`` limit the rBergomi variance is
  deterministic and rBergomi MC prices must converge to Black-Scholes (tested in Phase 2/3);
* a **fast Greeks engine** for hedging and the implied-volatility surface; and
* the standard quoting language (implied volatility) used by the market-maker simulator.

All formulas are vectorized over array inputs and validated. The convention is a
continuously-compounded risk-free rate ``r`` and continuous dividend yield ``q`` (defaulting
to zero), with forward ``F = S e^{(r - q) T}``.

For a call,

.. math::

    C = S e^{-qT} N(d_1) - K e^{-rT} N(d_2), \qquad
    d_{1,2} = \frac{\ln(S/K) + (r - q \pm \tfrac12 \sigma^2) T}{\sigma \sqrt{T}}.

The zero-volatility and zero-time-to-expiry limits are handled explicitly (the formulas
degenerate to the discounted intrinsic value), so the functions never return NaN.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from ..core.enums import OptionRight
from ..core.errors import ValidationError
from ..core.validation import check_non_negative

__all__ = ["BlackScholesGreeks", "greeks", "implied_volatility", "price"]

# Inputs at or below this magnitude are treated as the degenerate (deterministic) limit.
_EPS = 1e-12


def _validate_inputs(
    spot: NDArray[np.float64],
    strike: NDArray[np.float64],
    expiry: NDArray[np.float64],
    vol: NDArray[np.float64],
) -> None:
    """Validate the common Black-Scholes inputs (shared by price and Greeks)."""
    if np.any(spot <= 0.0):
        raise ValidationError("spot must be strictly positive", context={})
    if np.any(strike <= 0.0):
        raise ValidationError("strike must be strictly positive", context={})
    if np.any(expiry < 0.0):
        raise ValidationError("expiry must be non-negative", context={})
    if np.any(vol < 0.0):
        raise ValidationError("volatility must be non-negative", context={})
    for name, arr in (("spot", spot), ("strike", strike), ("expiry", expiry), ("vol", vol)):
        if not np.all(np.isfinite(arr)):
            raise ValidationError(f"{name} contains non-finite values", context={})


def _d1_d2(
    spot: NDArray[np.float64],
    strike: NDArray[np.float64],
    expiry: NDArray[np.float64],
    vol: NDArray[np.float64],
    rate: float,
    dividend: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return ``(d1, d2, sigma_sqrt_T)`` with safe handling of the degenerate limit.

    Where ``sigma * sqrt(T)`` is ~0 the option is (discounted) intrinsic; we set ``d1, d2``
    to +/- infinity according to moneyness so that ``N(d)`` collapses to the correct 0/1
    indicator, and return a zero ``sigma_sqrt_T`` that callers use to zero-out vega/gamma.
    """
    sigma_sqrt_t = vol * np.sqrt(expiry)
    forward = spot * np.exp((rate - dividend) * expiry)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(forward / strike) + 0.5 * sigma_sqrt_t**2) / sigma_sqrt_t
        d2 = d1 - sigma_sqrt_t
    degenerate = sigma_sqrt_t <= _EPS
    if np.any(degenerate):
        # In the deterministic limit, sign of (forward - strike) decides ITM/OTM.
        itm = forward > strike
        fill = np.where(itm, np.inf, -np.inf)
        d1 = np.where(degenerate, fill, d1)
        d2 = np.where(degenerate, fill, d2)
    return d1, d2, sigma_sqrt_t


def price(
    spot: NDArray[np.float64] | float,
    strike: NDArray[np.float64] | float,
    expiry: NDArray[np.float64] | float,
    vol: NDArray[np.float64] | float,
    right: OptionRight,
    *,
    rate: float = 0.0,
    dividend: float = 0.0,
) -> NDArray[np.float64]:
    r"""Return the Black-Scholes price of a European option, vectorized.

    Parameters
    ----------
    spot, strike, expiry, vol:
        Broadcastable arrays (or scalars) of spot price, strike, time to expiry (years), and
        annualized volatility. Must be non-negative (spot/strike strictly positive).
    right:
        Call or put.
    rate, dividend:
        Continuously-compounded risk-free rate and dividend yield.

    Returns
    -------
    numpy.ndarray
        The option price(s), always finite and non-negative.
    """
    s = np.atleast_1d(np.asarray(spot, dtype=np.float64))
    k = np.atleast_1d(np.asarray(strike, dtype=np.float64))
    t = np.atleast_1d(np.asarray(expiry, dtype=np.float64))
    v = np.atleast_1d(np.asarray(vol, dtype=np.float64))
    _validate_inputs(s, k, t, v)

    d1, d2, _ = _d1_d2(s, k, t, v, rate, dividend)
    disc = np.exp(-rate * t)
    disc_div = np.exp(-dividend * t)
    if right is OptionRight.CALL:
        result = s * disc_div * norm.cdf(d1) - k * disc * norm.cdf(d2)
    elif right is OptionRight.PUT:
        result = k * disc * norm.cdf(-d2) - s * disc_div * norm.cdf(-d1)
    else:  # pragma: no cover - exhaustive over the enum
        raise ValidationError("unknown option right", context={"right": right})
    # Numerical guard: prices are non-negative by no-arbitrage; clip tiny negatives.
    return np.asarray(np.maximum(result, 0.0), dtype=np.float64)


@dataclass(frozen=True, slots=True)
class BlackScholesGreeks:
    """Container for the standard option Greeks (per unit of underlying, vectorized).

    Conventions:

    * ``delta`` -- d(price)/d(spot).
    * ``gamma`` -- d^2(price)/d(spot)^2.
    * ``vega``  -- d(price)/d(vol), per 1.00 (i.e. 100 vol points) change; divide by 100 for
      per-vol-point.
    * ``theta`` -- d(price)/d(t) **per year** (negative for long options); divide by 365 for
      per-calendar-day.
    * ``rho``   -- d(price)/d(rate), per 1.00 change in rate.
    """

    delta: NDArray[np.float64]
    gamma: NDArray[np.float64]
    vega: NDArray[np.float64]
    theta: NDArray[np.float64]
    rho: NDArray[np.float64]


def greeks(
    spot: NDArray[np.float64] | float,
    strike: NDArray[np.float64] | float,
    expiry: NDArray[np.float64] | float,
    vol: NDArray[np.float64] | float,
    right: OptionRight,
    *,
    rate: float = 0.0,
    dividend: float = 0.0,
) -> BlackScholesGreeks:
    """Return the analytic Black-Scholes Greeks of a European option, vectorized.

    Inputs follow the same conventions as :func:`price`. In the degenerate
    (zero-vol or zero-time) limit, gamma/vega/theta vanish and delta becomes the 0/1
    indicator of moneyness, consistent with the intrinsic-value payoff.
    """
    s = np.atleast_1d(np.asarray(spot, dtype=np.float64))
    k = np.atleast_1d(np.asarray(strike, dtype=np.float64))
    t = np.atleast_1d(np.asarray(expiry, dtype=np.float64))
    v = np.atleast_1d(np.asarray(vol, dtype=np.float64))
    _validate_inputs(s, k, t, v)

    d1, d2, sigma_sqrt_t = _d1_d2(s, k, t, v, rate, dividend)
    disc = np.exp(-rate * t)
    disc_div = np.exp(-dividend * t)
    pdf_d1 = norm.pdf(d1)
    nondegenerate = sigma_sqrt_t > _EPS

    # Gamma and vega are identical for calls and puts.
    with np.errstate(divide="ignore", invalid="ignore"):
        gamma = disc_div * pdf_d1 / (s * sigma_sqrt_t)
        vega = s * disc_div * pdf_d1 * np.sqrt(t)
    gamma = np.where(nondegenerate, gamma, 0.0)
    vega = np.where(nondegenerate, vega, 0.0)

    sqrt_t = np.sqrt(t)
    with np.errstate(divide="ignore", invalid="ignore"):
        theta_common = -(s * disc_div * pdf_d1 * v) / (2.0 * sqrt_t)
    theta_common = np.where(nondegenerate, theta_common, 0.0)

    if right is OptionRight.CALL:
        delta = disc_div * norm.cdf(d1)
        theta = (
            theta_common - rate * k * disc * norm.cdf(d2) + dividend * s * disc_div * norm.cdf(d1)
        )
        rho = k * t * disc * norm.cdf(d2)
    elif right is OptionRight.PUT:
        delta = -disc_div * norm.cdf(-d1)
        theta = (
            theta_common
            + rate * k * disc * norm.cdf(-d2)
            - dividend * s * disc_div * norm.cdf(-d1)
        )
        rho = -k * t * disc * norm.cdf(-d2)
    else:  # pragma: no cover - exhaustive over the enum
        raise ValidationError("unknown option right", context={"right": right})

    return BlackScholesGreeks(
        delta=np.asarray(delta, dtype=np.float64),
        gamma=np.asarray(gamma, dtype=np.float64),
        vega=np.asarray(vega, dtype=np.float64),
        theta=np.asarray(theta, dtype=np.float64),
        rho=np.asarray(rho, dtype=np.float64),
    )


def implied_volatility(
    target_price: float,
    spot: float,
    strike: float,
    expiry: float,
    right: OptionRight,
    *,
    rate: float = 0.0,
    dividend: float = 0.0,
    tol: float = 1e-8,
    max_iter: int = 100,
) -> float:
    r"""Return the Black-Scholes implied volatility for a target price.

    Uses a bracketed hybrid of Newton's method and bisection (safeguarded Newton): Newton
    for speed where vega is healthy, bisection as a guaranteed-convergent fallback. This is
    robust even for deep OTM options where vega is tiny -- exactly the regime the wings of an
    iron condor live in.

    Parameters
    ----------
    target_price:
        Observed/target option price. Must respect the no-arbitrage bounds, otherwise a
        :class:`ValidationError` is raised (no implied vol exists outside them).

    Raises
    ------
    ValidationError
        If the target price violates the no-arbitrage bounds or inputs are invalid.
    """
    check_non_negative(target_price, name="target_price")
    if spot <= 0.0 or strike <= 0.0 or expiry <= 0.0:
        raise ValidationError(
            "spot, strike and expiry must be strictly positive",
            context={"spot": spot, "strike": strike, "expiry": expiry},
        )

    disc = np.exp(-rate * expiry)
    disc_div = np.exp(-dividend * expiry)
    forward = spot * disc_div
    if right is OptionRight.CALL:
        lower_bound = max(forward - strike * disc, 0.0)
        upper_bound = forward
    else:
        lower_bound = max(strike * disc - forward, 0.0)
        upper_bound = strike * disc
    # Small tolerance band for floating-point at the bounds.
    if target_price < lower_bound - 1e-10 or target_price > upper_bound + 1e-10:
        raise ValidationError(
            "target price violates no-arbitrage bounds; no implied volatility exists",
            context={
                "target": target_price,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
            },
        )

    def objective(sigma: float) -> float:
        modelled = float(
            price(spot, strike, expiry, sigma, right, rate=rate, dividend=dividend)[0]
        )
        return modelled - target_price

    low, high = 1e-9, 5.0
    f_low, f_high = objective(low), objective(high)
    # Expand the upper bracket if the price is extremely high (very high vol).
    expand = 0
    while f_low * f_high > 0.0 and high < 1e3 and expand < 20:
        high *= 2.0
        f_high = objective(high)
        expand += 1
    if f_low * f_high > 0.0:
        # Price is at a boundary (e.g. exactly intrinsic) -> vol is ~0.
        return 0.0 if abs(f_low) <= abs(f_high) else high

    sigma = 0.2  # sensible starting guess
    for _ in range(max_iter):
        f = objective(sigma)
        if abs(f) < tol:
            return sigma
        # Maintain the bracket.
        if f_low * f < 0.0:
            high = sigma
        else:
            low, f_low = sigma, f
        vega = float(
            greeks(spot, strike, expiry, sigma, right, rate=rate, dividend=dividend).vega[0]
        )
        if vega > 1e-10:
            step = f / vega
            newton = sigma - step
            if low < newton < high:
                sigma = newton
                continue
        sigma = 0.5 * (low + high)  # bisection fallback
    return sigma
