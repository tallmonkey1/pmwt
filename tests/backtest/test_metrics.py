"""Tests for backtest performance metrics, validated against hand-computed values."""

from __future__ import annotations

import numpy as np
import pytest

from options_engine.backtest.metrics import (
    compute_performance_metrics,
    conditional_value_at_risk,
    deflated_sharpe_ratio,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
)
from options_engine.core.errors import ValidationError


class TestSharpe:
    def test_zero_volatility_is_zero(self) -> None:
        assert sharpe_ratio(np.full(20, 0.01)) == 0.0

    def test_matches_definition(self) -> None:
        r = np.array([-0.009, 0.011] * 50)
        expected = np.sqrt(252) * np.mean(r) / np.std(r, ddof=1)
        assert sharpe_ratio(r) == pytest.approx(expected)

    def test_rejects_short_series(self) -> None:
        with pytest.raises(ValidationError):
            sharpe_ratio(np.array([0.01]))


class TestSortino:
    def test_no_downside_is_zero(self) -> None:
        assert sortino_ratio(np.full(10, 0.01)) == 0.0

    def test_positive_for_good_returns(self) -> None:
        rng = np.random.default_rng(0)
        r = rng.normal(0.001, 0.005, 500)
        assert sortino_ratio(r) > 0.0


class TestMaxDrawdown:
    def test_hand_computed(self) -> None:
        eq = np.array([100.0, 120.0, 90.0, 110.0])
        assert max_drawdown(eq) == pytest.approx(0.25)

    def test_monotone_curve_zero_drawdown(self) -> None:
        eq = np.array([100.0, 101.0, 102.0])
        assert max_drawdown(eq) == pytest.approx(0.0)

    def test_rejects_nonpositive(self) -> None:
        with pytest.raises(ValidationError):
            max_drawdown(np.array([100.0, 0.0]))


class TestCVaR:
    def test_hand_computed(self) -> None:
        r = np.array([-0.05, -0.04, -0.03, -0.02, -0.01, 0.01, 0.02, 0.03, 0.04, 0.05])
        # Worst 20% = [-0.05, -0.04]; CVaR = 0.045.
        assert conditional_value_at_risk(r, alpha=0.2) == pytest.approx(0.045)

    def test_rejects_bad_alpha(self) -> None:
        with pytest.raises(ValidationError):
            conditional_value_at_risk(np.array([0.01, -0.01]), alpha=1.5)


class TestDeflatedSharpe:
    def test_probability_in_unit_interval(self) -> None:
        rng = np.random.default_rng(0)
        r = rng.normal(0.001, 0.005, 500)
        dsr = deflated_sharpe_ratio(r, n_trials=10)
        assert 0.0 <= dsr <= 1.0

    def test_decreasing_in_trials(self) -> None:
        # More trials => more multiple-testing penalty => lower DSR.
        rng = np.random.default_rng(1)
        r = rng.normal(0.001, 0.005, 500)
        assert deflated_sharpe_ratio(r, n_trials=1) >= deflated_sharpe_ratio(r, n_trials=1000)

    def test_strong_strategy_high_dsr(self) -> None:
        rng = np.random.default_rng(2)
        r = rng.normal(0.002, 0.004, 1000)  # very high Sharpe
        assert deflated_sharpe_ratio(r, n_trials=1) > 0.95

    def test_rejects_bad_n_trials(self) -> None:
        with pytest.raises(ValidationError):
            deflated_sharpe_ratio(np.array([0.01, -0.01, 0.02]), n_trials=0)


class TestComputePerformanceMetrics:
    def _series(self):
        rng = np.random.default_rng(0)
        r = rng.normal(0.001, 0.01, 250)
        eq = 10000.0 * np.cumprod(1.0 + r)
        eq = np.concatenate([[10000.0], eq])
        return r, eq

    def test_bundle_fields(self) -> None:
        r, eq = self._series()
        m = compute_performance_metrics(returns=r, equity_curve=eq)
        assert m.n_periods == 250
        assert np.isfinite(m.sharpe)
        assert 0.0 <= m.max_drawdown <= 1.0
        assert 0.0 <= m.win_rate <= 1.0

    def test_profit_factor_with_trades(self) -> None:
        r, eq = self._series()
        trade_pnls = np.array([100.0, -50.0, 200.0, -25.0])
        m = compute_performance_metrics(returns=r, equity_curve=eq, trade_pnls=trade_pnls)
        # gross profit 300 / gross loss 75 = 4.0.
        assert m.profit_factor == pytest.approx(4.0)
        assert m.win_rate == pytest.approx(0.5)

    def test_total_return(self) -> None:
        r, eq = self._series()
        m = compute_performance_metrics(returns=r, equity_curve=eq)
        assert m.total_return == pytest.approx(eq[-1] / eq[0] - 1.0)
