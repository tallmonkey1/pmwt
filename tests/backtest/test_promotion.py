"""Tests for the promotion gates."""

from __future__ import annotations

import numpy as np
import pytest

from options_engine.backtest.metrics import compute_performance_metrics
from options_engine.backtest.promotion import (
    PromotionThresholds,
    evaluate_promotion,
)
from options_engine.backtest.results import BacktestResult
from options_engine.core.errors import ValidationError


def _result(
    *, mean: float, vol: float, n: int = 500, kill: bool = False, seed: int = 0
) -> BacktestResult:
    rng = np.random.default_rng(seed)
    rets = rng.normal(mean, vol, n)
    eq = 10000.0 * np.cumprod(1.0 + rets)
    eq = np.concatenate([[10000.0], eq])
    metrics = compute_performance_metrics(returns=rets, equity_curve=eq)
    return BacktestResult(
        equity_curve=eq,
        returns=rets,
        steps=(),
        trades=(),
        metrics=metrics,
        starting_equity=10000.0,
        ending_equity=float(eq[-1]),
        kill_switch_triggered=kill,
    )


class TestEvaluatePromotion:
    def test_approves_strong_strategy(self) -> None:
        res = _result(mean=0.0015, vol=0.004)  # high Sharpe, low drawdown
        report = evaluate_promotion(
            in_sample=res, n_trials=1, thresholds=PromotionThresholds(max_cvar_95=0.02)
        )
        assert report.approved
        assert all(g.passed for g in report.gates)

    def test_rejects_weak_strategy(self) -> None:
        res = _result(mean=0.0001, vol=0.02)  # poor Sharpe
        report = evaluate_promotion(in_sample=res, n_trials=100)
        assert not report.approved
        assert any(g.name == "deflated_sharpe" and not g.passed for g in report.failed_gates())

    def test_rejects_on_kill_switch(self) -> None:
        res = _result(mean=0.0015, vol=0.004, kill=True)
        report = evaluate_promotion(
            in_sample=res, n_trials=1, thresholds=PromotionThresholds(max_cvar_95=0.02)
        )
        assert not report.approved
        assert any(g.name == "kill_switch" and not g.passed for g in report.failed_gates())

    def test_walk_forward_consistency_gate(self) -> None:
        strong = _result(mean=0.0015, vol=0.004, seed=1)
        # Mostly-losing OOS folds should fail the consistency gate.
        folds = [
            _result(mean=-0.001, vol=0.01, seed=10),
            _result(mean=-0.001, vol=0.01, seed=11),
            _result(mean=0.002, vol=0.01, seed=12),
        ]
        report = evaluate_promotion(
            in_sample=strong,
            n_trials=1,
            thresholds=PromotionThresholds(max_cvar_95=0.05),
            walk_forward_results=folds,
        )
        assert any(g.name == "walk_forward_consistency" and not g.passed for g in report.gates)

    def test_crash_replay_robustness_gate(self) -> None:
        strong = _result(mean=0.0015, vol=0.004, seed=1)
        # A stressed scenario with a severe drawdown should fail the robustness gate.
        stressed = _result(mean=-0.01, vol=0.03, seed=20)
        report = evaluate_promotion(
            in_sample=strong,
            n_trials=1,
            thresholds=PromotionThresholds(max_cvar_95=0.05, max_stress_drawdown=0.10),
            stress_results=[stressed],
        )
        assert any(g.name == "crash_replay_robustness" and not g.passed for g in report.gates)

    def test_thresholds_validation(self) -> None:
        with pytest.raises(ValidationError):
            PromotionThresholds(min_deflated_sharpe=1.5)
        with pytest.raises(ValidationError):
            PromotionThresholds(max_drawdown=0.0)
