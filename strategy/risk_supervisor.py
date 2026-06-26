r"""Deterministic hard risk supervisor with kill-switch (SPEC §4.5).

This is the non-negotiable safety overlay that sits *outside* every decision component
(strategy rules today, the RL agent in later phases). It is purely deterministic and has the
final say: it can veto any new position and can trigger a global kill-switch that flattens
and halts trading. The learned/optimized logic can never widen its own risk budget because
the budget lives here, not in the policy.

Checks enforced before allowing a new position (SPEC §4.5, §6):

* **Kill-switch:** if the trailing drawdown breaches the configured limit, no new risk is
  permitted and the supervisor signals that open positions should be flattened.
* **Per-trade risk cap:** the position's defined-risk loss may not exceed the per-trade cap.
* **Per-day risk budget:** new risk plus risk already taken today may not exceed the daily
  budget.
* **Margin / leverage:** projected margin usage may not exceed available margin or the gross
  leverage ceiling.
* **Concentration:** the number of open positions may not exceed a hard maximum.

Every veto returns a structured, machine-readable reason for the audit trail.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..core.config import RiskConfig
from ..core.errors import ValidationError
from ..core.validation import check_non_negative
from .account import Account, OpenPosition

__all__ = ["RiskCheckResult", "RiskSupervisor", "RiskSupervisorConfig"]


@dataclass(frozen=True, slots=True)
class RiskSupervisorConfig:
    """Supervisor limits not already covered by the core :class:`RiskConfig`."""

    #: Maximum number of simultaneously open positions.
    max_open_positions: int = 20

    def __post_init__(self) -> None:
        if self.max_open_positions < 1:
            raise ValidationError(
                "max_open_positions must be >= 1",
                context={"max_open_positions": self.max_open_positions},
            )


@dataclass(frozen=True, slots=True)
class RiskCheckResult:
    """Outcome of a pre-trade risk check or a portfolio kill-switch evaluation."""

    approved: bool
    reason: str
    kill_switch_triggered: bool = False


class RiskSupervisor:
    """Deterministic pre-trade risk checks and the trailing-drawdown kill-switch.

    Parameters
    ----------
    risk:
        Core risk configuration (caps, drawdown limit, leverage ceiling).
    config:
        Supervisor-specific limits.
    """

    def __init__(self, *, risk: RiskConfig, config: RiskSupervisorConfig | None = None) -> None:
        if not isinstance(risk, RiskConfig):
            raise ValidationError("risk must be a RiskConfig", context={})
        self._risk = risk
        self._config = config or RiskSupervisorConfig()

    @property
    def risk(self) -> RiskConfig:
        """The core risk configuration."""
        return self._risk

    def check_kill_switch(
        self, account: Account, *, unrealized_pnl: float = 0.0
    ) -> RiskCheckResult:
        """Return whether the trailing-drawdown kill-switch is triggered.

        When triggered, *no new risk* may be opened and open positions should be flattened
        and trading halted (the runner enforces the flatten/halt).
        """
        drawdown = account.drawdown(unrealized_pnl)
        if drawdown >= self._risk.max_drawdown_kill_switch:
            return RiskCheckResult(
                approved=False,
                reason=(
                    f"kill-switch: drawdown {drawdown:.3f} >= "
                    f"limit {self._risk.max_drawdown_kill_switch:.3f}"
                ),
                kill_switch_triggered=True,
            )
        return RiskCheckResult(approved=True, reason="drawdown within limit")

    def approve_new_position(
        self,
        account: Account,
        candidate: OpenPosition,
        *,
        unrealized_pnl: float = 0.0,
        risked_today: float = 0.0,
        available_margin: float,
    ) -> RiskCheckResult:
        """Return whether a proposed new position passes all hard risk checks.

        Parameters
        ----------
        account:
            Current account state.
        candidate:
            The proposed position (already sized).
        unrealized_pnl:
            Current mark-to-market P&L of open positions (for drawdown/equity).
        risked_today:
            Capital already newly risked today (account currency).
        available_margin:
            Margin currently available to support new risk.
        """
        check_non_negative(risked_today, name="risked_today")
        check_non_negative(available_margin, name="available_margin")

        # 0. Kill-switch dominates everything.
        kill = self.check_kill_switch(account, unrealized_pnl=unrealized_pnl)
        if kill.kill_switch_triggered:
            return kill

        equity = account.equity(unrealized_pnl)
        if equity <= 0.0:
            return RiskCheckResult(
                approved=False,
                reason="non-positive equity; trading halted",
                kill_switch_triggered=True,
            )

        position_risk = candidate.margin_requirement

        # 1. Per-trade cap.
        per_trade_cap = self._risk.max_risk_fraction_per_trade * equity
        if position_risk > per_trade_cap + 1e-9:
            return RiskCheckResult(
                approved=False,
                reason=f"per-trade risk {position_risk:.2f} exceeds cap {per_trade_cap:.2f}",
            )

        # 2. Per-day budget.
        daily_budget = self._risk.max_risk_fraction_per_day * equity
        if risked_today + position_risk > daily_budget + 1e-9:
            return RiskCheckResult(
                approved=False,
                reason=(
                    f"daily risk {risked_today + position_risk:.2f} exceeds "
                    f"budget {daily_budget:.2f}"
                ),
            )

        # 3. Margin availability.
        if position_risk > available_margin + 1e-9:
            return RiskCheckResult(
                approved=False,
                reason=(
                    f"insufficient margin: need {position_risk:.2f}, "
                    f"have {available_margin:.2f}"
                ),
            )

        # 4. Leverage ceiling: projected gross margin / equity must not exceed the ceiling.
        projected_margin = account.total_margin + position_risk
        leverage = projected_margin / equity
        if leverage > self._risk.max_leverage + 1e-9:
            return RiskCheckResult(
                approved=False,
                reason=f"leverage {leverage:.2f} exceeds ceiling {self._risk.max_leverage:.2f}",
            )

        # 5. Concentration.
        if account.position_count() >= self._config.max_open_positions:
            return RiskCheckResult(
                approved=False,
                reason=f"max open positions ({self._config.max_open_positions}) reached",
            )

        return RiskCheckResult(approved=True, reason="all risk checks passed")
