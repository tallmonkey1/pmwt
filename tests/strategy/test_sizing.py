"""Tests for fractional-Kelly sizing with hard caps."""

from __future__ import annotations

import numpy as np
import pytest

from options_engine.core.config import RiskConfig
from options_engine.core.errors import ValidationError
from options_engine.strategy.sizing import (
    SizingInputs,
    empirical_kelly_fraction,
    kelly_fraction,
    size_position,
)


class TestEmpiricalKellyFraction:
    def test_zero_for_negative_expectation(self) -> None:
        # A losing bet sizes to zero.
        rng = np.random.default_rng(0)
        pnl = rng.normal(-0.1, 0.5, size=10_000)
        assert empirical_kelly_fraction(pnl) == 0.0

    def test_positive_for_favorable_sample(self) -> None:
        # Frequent wins, rare partial losses, positive expectation => positive Kelly.
        # E[pnl] = 0.8*0.5 - 0.2*1.0 = 0.20 > 0.
        rng = np.random.default_rng(1)
        wins = rng.random(10_000) < 0.8
        pnl = np.where(wins, 0.5, -1.0)
        f = empirical_kelly_fraction(pnl)
        assert 0.0 < f <= 1.0

    def test_exceeds_binary_kelly_for_partial_loss(self) -> None:
        # The whole point: for partial losses, empirical Kelly > binary Kelly.
        rng = np.random.default_rng(2)
        wins = rng.random(20_000) < 0.6
        # Partial loss of 0.5 (not the full unit) when losing.
        pnl = np.where(wins, 0.5, -0.5)
        emp = empirical_kelly_fraction(pnl)
        binary = kelly_fraction(win_probability=0.6, payoff_ratio=0.5 / 0.5)
        assert emp > binary

    def test_never_risks_ruin(self) -> None:
        # With a -1 (total loss) outcome possible, the fraction must stay < 1 to avoid ruin.
        rng = np.random.default_rng(3)
        wins = rng.random(10_000) < 0.8
        pnl = np.where(wins, 0.3, -1.0)
        assert empirical_kelly_fraction(pnl) < 1.0

    def test_rejects_short_sample(self) -> None:
        with pytest.raises(ValidationError):
            empirical_kelly_fraction(np.array([0.5]))


class TestKellyFraction:
    def test_known_value(self) -> None:
        # p=0.6, b=1 => f = 0.6 - 0.4/1 = 0.2.
        assert kelly_fraction(win_probability=0.6, payoff_ratio=1.0) == pytest.approx(0.2)

    def test_negative_edge_floored_to_zero(self) -> None:
        # p=0.5, b=0.5 => 0.5 - 0.5/0.5 = -0.5 -> floored to 0.
        assert kelly_fraction(win_probability=0.5, payoff_ratio=0.5) == 0.0

    def test_high_edge(self) -> None:
        assert kelly_fraction(win_probability=0.9, payoff_ratio=1.0) == pytest.approx(0.8)

    def test_rejects_bad_inputs(self) -> None:
        with pytest.raises(ValidationError):
            kelly_fraction(win_probability=1.5, payoff_ratio=1.0)
        with pytest.raises(ValidationError):
            kelly_fraction(win_probability=0.6, payoff_ratio=0.0)


class TestSizePosition:
    def _inputs(self, **kw) -> SizingInputs:
        defaults = {
            "account_equity": 100_000.0,
            "win_probability": 0.9,
            "net_credit": 1.0,
            "max_loss_per_condor": 2.0,
            "multiplier": 100.0,
            "available_margin": 100_000.0,
            "risked_today": 0.0,
        }
        defaults.update(kw)
        return SizingInputs(**defaults)  # type: ignore[arg-type]

    def test_sizes_positive_with_edge(self) -> None:
        result = size_position(self._inputs(), risk=RiskConfig())
        assert result.quantity >= 1
        assert result.capital_at_risk > 0.0

    def test_zero_for_no_edge(self) -> None:
        # Negative-edge trade sizes to zero (Kelly binding).
        result = size_position(
            self._inputs(win_probability=0.5, net_credit=0.5, max_loss_per_condor=2.0),
            risk=RiskConfig(),
        )
        assert result.quantity == 0

    def test_per_trade_cap_binds(self) -> None:
        # Very high Kelly but a tiny per-trade cap should bind the per-trade limit.
        risk = RiskConfig(max_risk_fraction_per_trade=0.01, max_risk_fraction_per_day=0.5)
        result = size_position(self._inputs(win_probability=0.95), risk=risk)
        # Capital at risk must respect the 1% per-trade cap.
        assert result.capital_at_risk <= 0.01 * 100_000.0 + 200.0  # within one condor's risk

    def test_daily_budget_binds(self) -> None:
        risk = RiskConfig(max_risk_fraction_per_trade=0.05, max_risk_fraction_per_day=0.05)
        # Already risked most of the daily budget (4900 of 5000).
        result = size_position(self._inputs(win_probability=0.95, risked_today=4_900.0), risk=risk)
        assert result.binding_constraint == "daily_budget"
        assert result.capital_at_risk <= 100.0 + 200.0

    def test_margin_binds(self) -> None:
        result = size_position(
            self._inputs(win_probability=0.95, available_margin=400.0), risk=RiskConfig()
        )
        assert result.binding_constraint == "margin"
        assert result.quantity <= 2

    def test_rejects_bad_inputs(self) -> None:
        with pytest.raises(ValidationError):
            SizingInputs(
                account_equity=-1.0,
                win_probability=0.6,
                net_credit=1.0,
                max_loss_per_condor=2.0,
                multiplier=100.0,
                available_margin=1.0,
            )
