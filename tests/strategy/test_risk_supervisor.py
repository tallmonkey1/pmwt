"""Tests for the deterministic risk supervisor and kill-switch."""

from __future__ import annotations

import datetime as dt

import pytest

from options_engine.core.config import RiskConfig
from options_engine.core.errors import ValidationError
from options_engine.pricing.instruments import IronCondor
from options_engine.strategy.account import Account, OpenPosition
from options_engine.strategy.risk_supervisor import (
    RiskSupervisor,
    RiskSupervisorConfig,
)

UTC = dt.UTC


def _position(pid: str, *, qty: int = 1, width: float = 5.0, credit: float = 1.0) -> OpenPosition:
    condor = IronCondor(90.0, 95.0, 95.0 + width, 95.0 + 2 * width, 0.05, quantity=qty)
    return OpenPosition(
        position_id=pid,
        condor=condor,
        entry_credit=credit,
        quantity=qty,
        multiplier=100.0,
        entry_time=dt.datetime(2024, 1, 1, tzinfo=UTC),
        entry_spot=100.0,
    )


class TestKillSwitch:
    def test_triggers_on_drawdown(self) -> None:
        risk = RiskConfig(max_drawdown_kill_switch=0.20)
        sup = RiskSupervisor(risk=risk)
        acct = Account.open(starting_cash=100_000.0)
        result = sup.check_kill_switch(acct, unrealized_pnl=-25_000.0)  # 25% drawdown
        assert result.kill_switch_triggered
        assert not result.approved

    def test_not_triggered_within_limit(self) -> None:
        risk = RiskConfig(max_drawdown_kill_switch=0.20)
        sup = RiskSupervisor(risk=risk)
        acct = Account.open(starting_cash=100_000.0)
        result = sup.check_kill_switch(acct, unrealized_pnl=-10_000.0)  # 10%
        assert not result.kill_switch_triggered
        assert result.approved


class TestApproveNewPosition:
    def _sup(self, **risk_kw) -> RiskSupervisor:
        return RiskSupervisor(risk=RiskConfig(**risk_kw))

    def test_approves_within_limits(self) -> None:
        sup = self._sup()
        acct = Account.open(starting_cash=100_000.0)
        pos = _position("P1")  # max loss = (5 - 1) * 100 = 400
        result = sup.approve_new_position(acct, pos, available_margin=100_000.0)
        assert result.approved

    def test_blocks_when_kill_switch_active(self) -> None:
        sup = self._sup(max_drawdown_kill_switch=0.20)
        acct = Account.open(starting_cash=100_000.0)
        pos = _position("P1")
        result = sup.approve_new_position(
            acct, pos, unrealized_pnl=-30_000.0, available_margin=100_000.0
        )
        assert not result.approved
        assert result.kill_switch_triggered

    def test_per_trade_cap(self) -> None:
        sup = self._sup(max_risk_fraction_per_trade=0.001, max_risk_fraction_per_day=0.5)
        acct = Account.open(starting_cash=100_000.0)
        pos = _position("P1", qty=10)  # max loss = 4000, > 0.1% of 100k = 100
        result = sup.approve_new_position(acct, pos, available_margin=100_000.0)
        assert not result.approved
        assert "per-trade" in result.reason

    def test_daily_budget(self) -> None:
        sup = self._sup(max_risk_fraction_per_trade=0.05, max_risk_fraction_per_day=0.05)
        acct = Account.open(starting_cash=100_000.0)
        pos = _position("P1", qty=10)  # 4000 risk, within per-trade cap (5000) ...
        result = sup.approve_new_position(
            acct,
            pos,
            risked_today=4_000.0,
            available_margin=100_000.0,  # ... but 8000 > 5000 daily
        )
        assert not result.approved
        assert "daily" in result.reason

    def test_margin(self) -> None:
        sup = self._sup()
        acct = Account.open(starting_cash=100_000.0)
        pos = _position("P1", qty=5)  # 2000 risk
        result = sup.approve_new_position(acct, pos, available_margin=500.0)
        assert not result.approved
        assert "margin" in result.reason

    def test_leverage_ceiling(self) -> None:
        # Leverage must bind before the per-trade/day caps: accumulate margin from existing
        # positions so that one more (cap-compliant) position tips total margin over equity.
        sup = self._sup(
            max_leverage=1.0, max_risk_fraction_per_trade=0.25, max_risk_fraction_per_day=0.5
        )
        acct = Account.open(starting_cash=10_000.0)
        # Five existing positions at 2000 risk each = 10000 margin already (= equity).
        for i in range(5):
            acct = acct.add_position(_position(f"E{i}", qty=5), premium_received=100.0)
        # One more 2000-risk position (20% of equity, within per-trade cap) pushes margin to
        # 12000 > 10000 equity => leverage 1.2x > 1.0 ceiling.
        pos = _position("P1", qty=5)
        result = sup.approve_new_position(acct, pos, available_margin=100_000.0)
        assert not result.approved
        assert "leverage" in result.reason

    def test_concentration(self) -> None:
        sup = RiskSupervisor(risk=RiskConfig(), config=RiskSupervisorConfig(max_open_positions=1))
        acct = Account.open(starting_cash=100_000.0).add_position(
            _position("P0"), premium_received=100.0
        )
        result = sup.approve_new_position(acct, _position("P1"), available_margin=100_000.0)
        assert not result.approved
        assert "max open positions" in result.reason

    def test_rejects_bad_config(self) -> None:
        with pytest.raises(ValidationError):
            RiskSupervisorConfig(max_open_positions=0)
