"""Tests for account and open-position state."""

from __future__ import annotations

import datetime as dt

import pytest

from options_engine.core.errors import ValidationError
from options_engine.pricing.instruments import IronCondor
from options_engine.strategy.account import Account, OpenPosition

UTC = dt.UTC


def _condor() -> IronCondor:
    return IronCondor(90.0, 95.0, 105.0, 110.0, 0.05, quantity=2)


def _position(pid: str = "P1") -> OpenPosition:
    return OpenPosition(
        position_id=pid,
        condor=_condor(),
        entry_credit=1.5,
        quantity=2,
        multiplier=100.0,
        entry_time=dt.datetime(2024, 1, 1, tzinfo=UTC),
        entry_spot=100.0,
    )


class TestOpenPosition:
    def test_economics(self) -> None:
        pos = _position()
        # Max profit = credit * qty * mult = 1.5 * 2 * 100.
        assert pos.max_profit == pytest.approx(300.0)
        # Max loss = (width 5 - credit 1.5) * 2 * 100.
        assert pos.max_loss == pytest.approx(700.0)
        assert pos.margin_requirement == pytest.approx(700.0)

    def test_rejects_nonpositive_credit(self) -> None:
        with pytest.raises(ValidationError):
            OpenPosition(
                position_id="P",
                condor=_condor(),
                entry_credit=0.0,
                quantity=1,
                multiplier=100.0,
                entry_time=dt.datetime(2024, 1, 1, tzinfo=UTC),
                entry_spot=100.0,
            )

    def test_rejects_naive_time(self) -> None:
        with pytest.raises(ValidationError):
            OpenPosition(
                position_id="P",
                condor=_condor(),
                entry_credit=1.0,
                quantity=1,
                multiplier=100.0,
                entry_time=dt.datetime(2024, 1, 1),
                entry_spot=100.0,
            )


class TestAccount:
    def test_open(self) -> None:
        acct = Account.open(starting_cash=100_000.0)
        assert acct.cash == 100_000.0
        assert acct.high_water_mark == 100_000.0
        assert acct.position_count() == 0

    def test_add_and_close_position(self) -> None:
        acct = Account.open(starting_cash=100_000.0)
        pos = _position()
        acct = acct.add_position(pos, premium_received=300.0)
        assert acct.position_count() == 1
        assert acct.cash == pytest.approx(100_300.0)
        assert acct.total_margin == pytest.approx(700.0)

        acct = acct.close_position("P1", realized=-100.0)
        assert acct.position_count() == 0
        assert acct.cash == pytest.approx(100_200.0)
        assert acct.realized_pnl == pytest.approx(-100.0)

    def test_close_unknown_position_raises(self) -> None:
        acct = Account.open(starting_cash=100_000.0)
        with pytest.raises(ValidationError):
            acct.close_position("missing", realized=0.0)

    def test_drawdown(self) -> None:
        acct = Account.open(starting_cash=100_000.0)
        # 10% mark-to-market loss => 10% drawdown.
        assert acct.drawdown(unrealized_pnl=-10_000.0) == pytest.approx(0.10)
        # Gains above HWM => zero drawdown.
        assert acct.drawdown(unrealized_pnl=5_000.0) == 0.0

    def test_high_water_mark_updates(self) -> None:
        acct = Account.open(starting_cash=100_000.0)
        acct2 = acct.with_high_water_mark_updated(unrealized_pnl=20_000.0)
        assert acct2.high_water_mark == pytest.approx(120_000.0)
        # After HWM rises, a pullback registers as drawdown from the peak.
        assert acct2.drawdown(unrealized_pnl=10_000.0) == pytest.approx(
            (120_000.0 - 110_000.0) / 120_000.0
        )

    def test_immutability_of_updates(self) -> None:
        acct = Account.open(starting_cash=100_000.0)
        acct.add_position(_position(), premium_received=300.0)
        # Original is unchanged (functional updates).
        assert acct.position_count() == 0
