r"""Terminal payoff functions for options and the iron condor.

Payoffs are deterministic functions of the terminal underlying price and are the basis for
both Monte-Carlo pricing (expected discounted payoff) and realized-P&L accounting. They are
fully vectorized over arrays of terminal prices so they can be applied to an entire
Monte-Carlo sample at once.

All payoffs are *per unit of underlying* (i.e. per share); contract multipliers and
position sizing are applied later by the strategy/execution layers, keeping this module a
pure, reusable mathematical core.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..core.enums import OptionRight
from ..core.errors import ValidationError
from ..core.validation import check_array_finite
from .instruments import EuropeanOption, IronCondor, OptionLeg, BullPutSpread

__all__ = [
    "iron_condor_payoff",
    "iron_condor_pnl",
    "leg_payoff",
    "option_payoff",
    "bull_put_spread_payoff",
]
...
def bull_put_spread_payoff(
    spread: BullPutSpread, terminal_spot: NDArray[np.float64]
) -> NDArray[np.float64]:
    r"""Return the terminal payoff (excluding the entry credit) of the bull put spread.

    This is the sum of the two signed leg payoffs. For the short spread the result is
    non-positive.
    """
    spot = np.asarray(terminal_spot, dtype=np.float64)
    total = np.zeros(spot.shape, dtype=np.float64)
    for leg in spread.legs():
        total = total + leg_payoff(leg, spot)
    return total


def option_payoff(
    option: EuropeanOption, terminal_spot: NDArray[np.float64]
) -> NDArray[np.float64]:
    r"""Return the terminal payoff of one long option over a sample of terminal spots.

    Call: :math:`\max(S_T - K, 0)`; put: :math:`\max(K - S_T, 0)`.
    """
    spot = np.asarray(terminal_spot, dtype=np.float64)
    check_array_finite(spot, name="terminal_spot")
    if np.any(spot < 0.0):
        raise ValidationError("terminal spot must be non-negative", context={})
    if option.right is OptionRight.CALL:
        return np.maximum(spot - option.strike, 0.0)
    return np.maximum(option.strike - spot, 0.0)


def leg_payoff(leg: OptionLeg, terminal_spot: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the signed terminal payoff of an option leg (``quantity * payoff``)."""
    return leg.quantity * option_payoff(leg.option, terminal_spot)


def iron_condor_payoff(
    condor: IronCondor, terminal_spot: NDArray[np.float64]
) -> NDArray[np.float64]:
    r"""Return the terminal payoff (excluding the entry credit) of the iron condor.

    This is the sum of the four signed leg payoffs. For the short condor the result is
    non-positive (you owe intrinsic value when the underlying breaches a short strike); the
    net trade P&L combines this with the credit received at entry -- see
    :func:`iron_condor_pnl`.
    """
    spot = np.asarray(terminal_spot, dtype=np.float64)
    total = np.zeros(spot.shape, dtype=np.float64)
    for leg in condor.legs():
        total = total + leg_payoff(leg, spot)
    return total


def iron_condor_pnl(
    condor: IronCondor,
    terminal_spot: NDArray[np.float64],
    *,
    entry_credit: float,
    multiplier: float = 1.0,
) -> NDArray[np.float64]:
    r"""Return the per-condor net P&L at expiry given the credit collected at entry.

    Parameters
    ----------
    condor:
        The iron-condor structure.
    terminal_spot:
        Sample of terminal underlying prices.
    entry_credit:
        Net premium received when opening one condor (per unit underlying, positive for a
        credit). This is the maximum profit.
    multiplier:
        Contract multiplier (e.g. 100 for standard listed options). The returned P&L is per
        condor including the multiplier.

    Notes
    -----
    Net P&L per condor = ``(entry_credit + terminal_payoff) * multiplier``, where
    ``terminal_payoff`` is non-positive for the short structure. The worst case equals
    ``(entry_credit - max_spread_width) * multiplier`` and the best case is
    ``entry_credit * multiplier`` -- both verified in the test suite.
    """
    if multiplier <= 0.0:
        raise ValidationError("multiplier must be positive", context={"multiplier": multiplier})
    payoff = iron_condor_payoff(condor, terminal_spot)
    return (entry_credit + payoff) * multiplier
