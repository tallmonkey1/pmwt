r"""Aggregate Black-Scholes Greeks for multi-leg structures (e.g. the iron condor).

The strategy and risk layers (SPEC §4.3, §5) need the *net* Greeks of a position to manage
delta/gamma/vega/theta exposure and to drive hedging decisions. This module computes those
net Greeks from the analytic single-option Greeks, summing signed contributions across legs
at a common spot, volatility, and rate.

Using analytic Greeks here (rather than Monte-Carlo bump-and-revalue) is deliberate: they
are exact, fast, and noise-free, which matters for stable hedging. The rBergomi-specific
risk still enters through the *volatility* supplied per leg (e.g. from the model's implied
surface), so this is the right separation of concerns.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..core.errors import ValidationError
from ..core.validation import check_positive
from . import black_scholes as bs
from .instruments import IronCondor, OptionLeg

__all__ = ["NetGreeks", "iron_condor_greeks", "leg_net_greeks"]


@dataclass(frozen=True, slots=True)
class NetGreeks:
    """Net position Greeks (scalars), per unit underlying times the contract multiplier."""

    price: float
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float

    def __add__(self, other: NetGreeks) -> NetGreeks:
        """Sum two net-Greek bundles componentwise (for portfolio aggregation)."""
        return NetGreeks(
            price=self.price + other.price,
            delta=self.delta + other.delta,
            gamma=self.gamma + other.gamma,
            vega=self.vega + other.vega,
            theta=self.theta + other.theta,
            rho=self.rho + other.rho,
        )


def leg_net_greeks(
    leg: OptionLeg,
    *,
    spot: float,
    vol: float,
    rate: float = 0.0,
    dividend: float = 0.0,
    multiplier: float = 1.0,
) -> NetGreeks:
    """Return the signed net Greeks of a single option leg.

    The leg's signed quantity and the contract multiplier scale every Greek, so the result
    is in account terms (per the whole leg position).
    """
    check_positive(spot, name="spot")
    check_positive(multiplier, name="multiplier")
    opt = leg.option
    scale = leg.quantity * multiplier
    px = float(
        bs.price(spot, opt.strike, opt.expiry, vol, opt.right, rate=rate, dividend=dividend)[0]
    )
    g = bs.greeks(spot, opt.strike, opt.expiry, vol, opt.right, rate=rate, dividend=dividend)
    return NetGreeks(
        price=scale * px,
        delta=scale * float(g.delta[0]),
        gamma=scale * float(g.gamma[0]),
        vega=scale * float(g.vega[0]),
        theta=scale * float(g.theta[0]),
        rho=scale * float(g.rho[0]),
    )


def iron_condor_greeks(
    condor: IronCondor,
    *,
    spot: float,
    leg_vols: NDArray[np.float64] | tuple[float, float, float, float],
    rate: float = 0.0,
    dividend: float = 0.0,
    multiplier: float = 1.0,
) -> NetGreeks:
    r"""Return the net Greeks of an iron condor with per-leg volatilities.

    Parameters
    ----------
    condor:
        The iron-condor structure.
    leg_vols:
        Four volatilities, one per leg in the order returned by :meth:`IronCondor.legs`
        (put-long, put-short, call-short, call-long). Distinct per-leg vols capture the
        volatility skew/smile across strikes, which is essential for correct wing pricing.
    rate, dividend, multiplier:
        Standard pricing conventions and the contract multiplier.
    """
    vols = np.asarray(leg_vols, dtype=np.float64)
    legs = condor.legs()
    if vols.shape != (len(legs),):
        raise ValidationError(
            "leg_vols must provide exactly one volatility per leg",
            context={"expected": len(legs), "got": int(vols.size)},
        )
    total = NetGreeks(price=0.0, delta=0.0, gamma=0.0, vega=0.0, theta=0.0, rho=0.0)
    for leg, vol in zip(legs, vols, strict=True):
        total = total + leg_net_greeks(
            leg, spot=spot, vol=float(vol), rate=rate, dividend=dividend, multiplier=multiplier
        )
    return total
