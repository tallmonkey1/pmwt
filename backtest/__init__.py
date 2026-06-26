"""Backtest engines, performance metrics, and promotion gates (SPEC §9).

Runs a policy (RL agent or rule) through the trading environment, computes rigorous
performance/risk statistics (including the deflated Sharpe), validates with purged walk-forward
cross-validation, and enforces the HISTORY -> PAPER -> LIVE promotion gates.

Public surface:

* Metrics: :func:`compute_performance_metrics`, :class:`PerformanceMetrics`,
  :func:`sharpe_ratio`, :func:`sortino_ratio`, :func:`max_drawdown`,
  :func:`deflated_sharpe_ratio`, :func:`conditional_value_at_risk`.
* Engine: :func:`run_backtest`, :class:`BacktestConfig`, :class:`PolicyProtocol`.
* Results: :class:`BacktestResult`, :class:`StepRecord`, :class:`TradeRecord`.
* Validation: :func:`purged_walk_forward_splits`, :class:`WalkForwardSplit`.
* Promotion: :func:`evaluate_promotion`, :class:`PromotionThresholds`,
  :class:`PromotionReport`, :class:`GateOutcome`.
"""

from __future__ import annotations

from .engine import BacktestConfig, PolicyProtocol, run_backtest
from .metrics import (
    PerformanceMetrics,
    compute_performance_metrics,
    conditional_value_at_risk,
    deflated_sharpe_ratio,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
)
from .promotion import (
    GateOutcome,
    PromotionReport,
    PromotionThresholds,
    evaluate_promotion,
)
from .results import BacktestResult, StepRecord, TradeRecord
from .validation import WalkForwardSplit, purged_walk_forward_splits

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "GateOutcome",
    "PerformanceMetrics",
    "PolicyProtocol",
    "PromotionReport",
    "PromotionThresholds",
    "StepRecord",
    "TradeRecord",
    "WalkForwardSplit",
    "compute_performance_metrics",
    "conditional_value_at_risk",
    "deflated_sharpe_ratio",
    "evaluate_promotion",
    "max_drawdown",
    "purged_walk_forward_splits",
    "run_backtest",
    "sharpe_ratio",
    "sortino_ratio",
]
