r"""Multi-reason exit logic for open iron-condor positions (SPEC §5).

The spec requires redundant exit triggers so management never depends on a single signal.
This module evaluates an open position against several independent conditions and returns the
first that fires (ordered by urgency):

1. **Stop-loss** -- unrealized loss reaches a multiple of the credit collected (defensive
   close before the defined-risk maximum is hit).
2. **Profit target** -- a configured fraction of maximum profit has been captured (lock in
   theta decay rather than holding for the last few cents).
3. **Regime breach** -- the regime is no longer calm (the conditions that justified the
   trade have gone away).
4. **Time stop** -- close before the gamma cliff near expiry, where pin/assignment risk and
   adverse convexity spike.

The decision is honest about what it can do: it minimizes *expected* loss given current
information; it cannot (and does not claim to) avoid loss with certainty (SPEC §0).
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass

from ..core.enums import VolRegime
from ..core.errors import ValidationError
from ..core.validation import check_non_negative, check_positive, check_unit_interval
from ..regime.detector import RegimeNowcast
from .account import OpenPosition

__all__ = ["ExitConfig", "ExitDecision", "evaluate_exit"]


@dataclass(frozen=True, slots=True)
class ExitConfig:
    """Configuration for position exit triggers."""

    #: Close when captured profit reaches this fraction of maximum profit (e.g. 0.5 = 50%).
    profit_target_fraction: float = 0.5
    #: Close when the unrealized loss reaches this multiple of the credit collected.
    stop_loss_credit_multiple: float = 2.0
    #: Close when time to expiry falls below this many days (gamma-cliff avoidance).
    time_stop_days: float = 1.0
    #: Exit if the probability of a calm (low-vol) regime falls below this level.
    min_regime_low_prob: float = 0.40
    #: Whether the regime-breach exit is active.
    enable_regime_exit: bool = True

    def __post_init__(self) -> None:
        check_unit_interval(
            self.profit_target_fraction, name="profit_target_fraction", inclusive=False
        )
        check_positive(self.stop_loss_credit_multiple, name="stop_loss_credit_multiple")
        check_non_negative(self.time_stop_days, name="time_stop_days")
        check_unit_interval(self.min_regime_low_prob, name="min_regime_low_prob")


@dataclass(frozen=True, slots=True)
class ExitDecision:
    """The exit evaluator's verdict, with an auditable reason code."""

    exit_position: bool
    reason: str
    trigger: (
        str  # machine-readable: "stop_loss" | "profit_target" | "regime" | "time_stop" | "hold"
    )

    @property
    def hold(self) -> bool:
        """True if the position should be held."""
        return not self.exit_position


def evaluate_exit(
    position: OpenPosition,
    *,
    now: _dt.datetime,
    unrealized_pnl: float,
    regime: RegimeNowcast | None = None,
    config: ExitConfig | None = None,
    trading_days_per_year: int = 252,
) -> ExitDecision:
    r"""Return the exit decision for an open position given current marks and regime.

    Parameters
    ----------
    position:
        The open iron-condor position.
    now:
        Timezone-aware current time.
    unrealized_pnl:
        Current mark-to-market P&L of the *whole* position (account currency; losses
        negative).
    regime:
        Current regime nowcast (optional; required only for the regime-breach exit).
    config:
        Exit configuration; defaults are used if omitted.
    trading_days_per_year:
        Annualization used to convert the condor's year-fraction expiry into a day count.

    Returns
    -------
    ExitDecision
        Whether to exit and why. Triggers are checked in urgency order.
    """
    if now.tzinfo is None or now.tzinfo.utcoffset(now) is None:
        raise ValidationError("now must be timezone-aware", context={})
    cfg = config or ExitConfig()

    max_profit = position.max_profit
    credit_collected = position.entry_credit * position.quantity * position.multiplier

    # 1. Stop-loss: unrealized loss exceeds the credit multiple.
    loss = -unrealized_pnl  # positive when losing
    if loss >= cfg.stop_loss_credit_multiple * credit_collected:
        return ExitDecision(
            exit_position=True,
            reason=(
                f"stop-loss: loss {loss:.2f} >= "
                f"{cfg.stop_loss_credit_multiple:.1f}x credit {credit_collected:.2f}"
            ),
            trigger="stop_loss",
        )

    # 2. Profit target: captured a sufficient fraction of max profit.
    if max_profit > 0.0 and unrealized_pnl >= cfg.profit_target_fraction * max_profit:
        return ExitDecision(
            exit_position=True,
            reason=(
                f"profit target: captured {unrealized_pnl:.2f} >= "
                f"{cfg.profit_target_fraction:.0%} of max {max_profit:.2f}"
            ),
            trigger="profit_target",
        )

    # 3. Regime breach: conditions that justified the trade have deteriorated.
    if cfg.enable_regime_exit and regime is not None:
        low_prob = regime.current_prob(VolRegime.LOW)
        if low_prob < cfg.min_regime_low_prob:
            return ExitDecision(
                exit_position=True,
                reason=f"regime breach: P(low) {low_prob:.3f} < {cfg.min_regime_low_prob:.2f}",
                trigger="regime",
            )

    # 4. Time stop: avoid the gamma cliff near expiry.
    days_to_expiry = position.condor.expiry * trading_days_per_year - _elapsed_trading_days(
        position.entry_time, now, trading_days_per_year
    )
    if days_to_expiry <= cfg.time_stop_days:
        return ExitDecision(
            exit_position=True,
            reason=f"time stop: {days_to_expiry:.2f} days to expiry <= {cfg.time_stop_days}",
            trigger="time_stop",
        )

    return ExitDecision(exit_position=False, reason="all hold conditions met", trigger="hold")


def _elapsed_trading_days(
    start: _dt.datetime, now: _dt.datetime, trading_days_per_year: int
) -> float:
    """Return calendar-time elapsed expressed in trading days (fractional).

    Uses an actual/365.25 calendar fraction scaled to the trading-day count, which is the
    standard convention for converting holding time to the same units as a year-fraction
    expiry. (Intraday precision is preserved, which matters for the MFD mode.)
    """
    if now < start:
        return 0.0
    elapsed_seconds = (now - start).total_seconds()
    years = elapsed_seconds / (365.25 * 24 * 3600)
    return years * trading_days_per_year
