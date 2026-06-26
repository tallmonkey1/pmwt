r"""Position sizing: fractional Kelly with hard risk caps (SPEC §6).

The strategy sizes each defined-risk condor by the Kelly criterion on the trade's own
win/loss economics, scaled by a *fractional-Kelly* factor and then **clamped** by the hard
per-trade and per-day risk budgets and available margin. This is the spec's "aggressive
within a hard envelope": Kelly sets the ambition, the caps guarantee survival.

For a binary-outcome bet that wins ``b`` per unit risked with probability ``p`` and loses the
full unit with probability ``1 - p``, the Kelly fraction of *risk capital* is

.. math::

    f^* = \frac{p\,b - (1 - p)}{b} = p - \frac{1 - p}{b}.

An iron condor is well-approximated as such a bet: ``win`` = net credit, ``loss`` =
``max_loss_per_condor`` = spread width minus credit, so ``b = win / loss``. We apply
fractional Kelly, floor at zero (never bet a negative edge), and translate the resulting
risk-capital fraction into an integer number of condors that respects every cap.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..core.config import RiskConfig
from ..core.errors import ValidationError
from ..core.validation import check_positive, check_probability

__all__ = [
    "SizingInputs",
    "SizingResult",
    "empirical_kelly_fraction",
    "kelly_fraction",
    "size_position",
]


def empirical_kelly_fraction(
    pnl_per_unit_risk: NDArray[np.float64], *, max_fraction: float = 1.0, n_grid: int = 101
) -> float:
    r"""Return the growth-optimal Kelly fraction from a P&L-per-unit-risk sample.

    The binary :func:`kelly_fraction` assumes the loss is always the full unit, which badly
    *under-sizes* structures like iron condors whose losses are usually partial. The honest,
    general criterion maximizes the expected log growth rate over the *actual* outcome
    distribution:

    .. math::

        f^* = \arg\max_{f \in [0, f_\max]} \; \mathbb{E}\big[\log(1 + f\, R)\big],

    where ``R`` is the trade's P&L per unit of risk capital (so a total loss is ``R = -1``).
    We evaluate the concave objective on a grid in ``[0, max_fraction]`` and return the
    maximizer, floored at 0 (a non-positive-edge sample sizes to zero). Using the empirical
    sample makes no distributional assumption and naturally accounts for the condor's
    asymmetric, partial-loss payoff.

    Parameters
    ----------
    pnl_per_unit_risk:
        Sample of trade P&L divided by the per-unit risk capital (max loss). Values should be
        ``>= -1`` for a defined-risk trade; values are clipped at ``-1 + eps`` for safety.
    max_fraction:
        Upper bound on the returned fraction (full Kelly cap before the fractional scaler).
    n_grid:
        Number of grid points for the 1-D maximization.
    """
    r = np.asarray(pnl_per_unit_risk, dtype=np.float64)
    if r.ndim != 1 or r.size < 2:
        raise ValidationError(
            "pnl_per_unit_risk must be a 1-D sample of length >= 2", context={"size": int(r.size)}
        )
    if not np.all(np.isfinite(r)):
        raise ValidationError("pnl_per_unit_risk contains non-finite values", context={})
    check_positive(max_fraction, name="max_fraction")
    if n_grid < 3:
        raise ValidationError("n_grid must be >= 3", context={"n_grid": n_grid})

    # No positive expectation => no bet.
    if float(np.mean(r)) <= 0.0:
        return 0.0

    fractions = np.linspace(0.0, max_fraction, n_grid)
    best_f = 0.0
    best_growth = 0.0  # growth at f = 0 is log(1) = 0
    for f in fractions[1:]:
        wealth = 1.0 + f * r
        if np.any(wealth <= 0.0):
            # This fraction risks ruin on the worst sampled outcome; stop increasing f.
            break
        growth = float(np.mean(np.log(wealth)))
        if growth > best_growth:
            best_growth = growth
            best_f = float(f)
    return best_f


def kelly_fraction(*, win_probability: float, payoff_ratio: float) -> float:
    r"""Return the (full) Kelly fraction of risk capital for a binary bet.

    Parameters
    ----------
    win_probability:
        Probability of the winning outcome ``p`` in ``[0, 1]``.
    payoff_ratio:
        ``b`` = amount won per unit risked on a win (``> 0``).

    Returns
    -------
    float
        The Kelly fraction, floored at 0 (a non-positive-edge bet sizes to zero).
    """
    p = check_probability(win_probability, name="win_probability")
    b = check_positive(payoff_ratio, name="payoff_ratio")
    f = p - (1.0 - p) / b
    return max(0.0, f)


@dataclass(frozen=True, slots=True)
class SizingInputs:
    """Inputs required to size a single condor trade."""

    account_equity: float
    win_probability: float
    net_credit: float  # per condor, per unit underlying
    max_loss_per_condor: float  # per condor, per unit underlying (positive)
    multiplier: float
    available_margin: float
    risked_today: float = 0.0  # capital already newly risked today (account currency)
    #: Optional P&L-per-unit-risk sample enabling the (more accurate) empirical Kelly. When
    #: provided, growth-optimal sizing is used instead of the conservative binary formula.
    pnl_per_unit_risk: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        check_positive(self.account_equity, name="account_equity")
        check_probability(self.win_probability, name="win_probability")
        check_positive(self.net_credit, name="net_credit")
        check_positive(self.max_loss_per_condor, name="max_loss_per_condor")
        check_positive(self.multiplier, name="multiplier")
        if self.available_margin < 0.0:
            raise ValidationError("available_margin must be non-negative", context={})
        if self.risked_today < 0.0:
            raise ValidationError("risked_today must be non-negative", context={})


@dataclass(frozen=True, slots=True)
class SizingResult:
    """The sizing decision with a full audit of which constraint bound it."""

    quantity: int
    kelly_fraction: float
    capital_at_risk: float
    binding_constraint: str


def size_position(inputs: SizingInputs, *, risk: RiskConfig) -> SizingResult:
    """Return the number of condors to trade under fractional Kelly and the hard caps.

    The size is the minimum across four limits: fractional-Kelly target, the per-trade risk
    cap, the remaining per-day risk budget, and available margin. The binding constraint is
    reported for auditability. A zero result is a valid, expected outcome (e.g. no edge or no
    remaining budget) -- the strategy simply does not trade.
    """
    risk_per_condor = inputs.max_loss_per_condor * inputs.multiplier
    if risk_per_condor <= 0.0:  # pragma: no cover - guarded by positive max_loss
        raise ValidationError("risk per condor must be positive", context={})

    # Prefer the empirical (growth-optimal) Kelly when a P&L sample is supplied; it correctly
    # accounts for the condor's partial-loss payoff. Otherwise fall back to the conservative
    # binary formula from the win-probability and payoff ratio.
    if inputs.pnl_per_unit_risk is not None:
        full_kelly = empirical_kelly_fraction(inputs.pnl_per_unit_risk)
    else:
        payoff_ratio = inputs.net_credit / inputs.max_loss_per_condor
        full_kelly = kelly_fraction(
            win_probability=inputs.win_probability, payoff_ratio=payoff_ratio
        )
    fractional = full_kelly * risk.kelly_fraction

    # 1. Kelly target capital, in account currency.
    kelly_capital = fractional * inputs.account_equity
    # 2. Per-trade hard cap.
    per_trade_capital = risk.max_risk_fraction_per_trade * inputs.account_equity
    # 3. Remaining per-day budget.
    daily_budget = risk.max_risk_fraction_per_day * inputs.account_equity
    remaining_daily = max(0.0, daily_budget - inputs.risked_today)
    # 4. Available margin.
    margin_capital = inputs.available_margin

    limits = {
        "kelly": kelly_capital,
        "per_trade_cap": per_trade_capital,
        "daily_budget": remaining_daily,
        "margin": margin_capital,
    }
    binding_constraint = min(limits, key=lambda k: limits[k])
    allowed_capital = limits[binding_constraint]

    quantity = int(allowed_capital // risk_per_condor)
    if quantity < 1:
        return SizingResult(
            quantity=0,
            kelly_fraction=full_kelly,
            capital_at_risk=0.0,
            binding_constraint=binding_constraint,
        )

    return SizingResult(
        quantity=quantity,
        kelly_fraction=full_kelly,
        capital_at_risk=quantity * risk_per_condor,
        binding_constraint=binding_constraint,
    )
