r"""Backtest result containers (SPEC §9, §11).

These immutable records are the audited output of a backtest: the equity curve, the per-step
log, the trade ledger, and the computed performance metrics. They are deliberately rich
enough to reconstruct *why* the strategy did what it did (observability / audit trail,
SPEC §11) and to feed the promotion gates (SPEC §9).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from ..core.errors import ValidationError
from .metrics import PerformanceMetrics

__all__ = ["BacktestResult", "StepRecord", "TradeRecord"]


@dataclass(frozen=True, slots=True)
class StepRecord:
    """One decision step's audit entry."""

    step: int
    equity: float
    reward: float
    traded: bool
    limit_breached: bool
    strategic_action: str
    regime_low_prob: float


@dataclass(frozen=True, slots=True)
class TradeRecord:
    """One opened-and-settled trade's economics."""

    step: int
    pnl: float
    credit: float
    transaction_cost: float
    quantity: int


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """The full result of a backtest run."""

    equity_curve: NDArray[np.float64]
    returns: NDArray[np.float64]
    steps: tuple[StepRecord, ...]
    trades: tuple[TradeRecord, ...]
    metrics: PerformanceMetrics
    starting_equity: float
    ending_equity: float
    kill_switch_triggered: bool
    metadata: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        eq = np.asarray(self.equity_curve, dtype=np.float64)
        ret = np.asarray(self.returns, dtype=np.float64)
        if eq.ndim != 1 or eq.size < 1:
            raise ValidationError("equity_curve must be a non-empty 1-D series", context={})
        if ret.ndim != 1:
            raise ValidationError("returns must be 1-D", context={})
        object.__setattr__(self, "equity_curve", eq)
        object.__setattr__(self, "returns", ret)

    @property
    def n_trades(self) -> int:
        """Number of trades executed."""
        return len(self.trades)

    @property
    def total_pnl(self) -> float:
        """Net P&L across the backtest (ending minus starting equity)."""
        return self.ending_equity - self.starting_equity
