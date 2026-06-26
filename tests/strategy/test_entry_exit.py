"""Tests for the multi-filter entry evaluator and multi-reason exit logic."""

from __future__ import annotations

import datetime as dt

import pytest

from options_engine.core.config import RiskConfig
from options_engine.core.errors import ValidationError
from options_engine.news import (
    EventSeverity,
    NewsItem,
    ReplayEventProvider,
    ReplayNewsProvider,
    ScheduledEvent,
)
from options_engine.news.gate import NewsGate
from options_engine.pricing.instruments import IronCondor
from options_engine.strategy.account import Account, OpenPosition
from options_engine.strategy.entry import EntryConfig, EntryEvaluator
from options_engine.strategy.exit import ExitConfig, evaluate_exit
from options_engine.strategy.risk_supervisor import RiskSupervisor

from .conftest import calm_regime, stressed_regime

UTC = dt.UTC
NOW = dt.datetime(2024, 3, 1, 15, 0, tzinfo=UTC)
UNIVERSE = frozenset({"SPX"})


def _news_gate(news=None, events=None) -> NewsGate:
    return NewsGate(
        news_provider=ReplayNewsProvider(news or []),
        event_provider=ReplayEventProvider(events or []),
        universe=UNIVERSE,
    )


def _evaluator(news_gate: NewsGate | None = None, **entry_kw) -> EntryEvaluator:
    return EntryEvaluator(
        risk=RiskConfig(),
        risk_supervisor=RiskSupervisor(risk=RiskConfig()),
        news_gate=news_gate or _news_gate(),
        config=EntryConfig(**entry_kw) if entry_kw else None,
    )


class TestEntryEvaluator:
    def _evaluate(self, ev: EntryEvaluator, market_setup, regime):
        return ev.evaluate(
            now=NOW,
            symbol="SPX",
            distribution=market_setup["dist"],
            chain=market_setup["chain"],
            regime=regime,
            account=Account.open(starting_cash=100_000.0),
            position_id="P1",
            multiplier=100.0,
            terminal_sample=market_setup["terminal_sample"],
            available_margin=50_000.0,
        )

    def test_blocked_by_regime(self, market_setup) -> None:
        decision = self._evaluate(_evaluator(), market_setup, stressed_regime())
        assert decision.rejected
        assert "regime gate" in decision.reason

    def test_blocked_by_news_cooloff(self, market_setup) -> None:
        shock = NewsItem(
            timestamp=dt.datetime(2024, 2, 29, 14, 0, tzinfo=UTC),
            headline="emergency rate hike shocks markets",
            source="wire",
        )
        ev = _evaluator(_news_gate(news=[shock]))
        decision = self._evaluate(ev, market_setup, calm_regime())
        assert decision.rejected
        assert "news gate" in decision.reason

    def test_blocked_by_scheduled_event(self, market_setup) -> None:
        fomc = ScheduledEvent(
            timestamp=dt.datetime(2024, 3, 1, 18, 0, tzinfo=UTC),
            name="FOMC",
            severity=EventSeverity.CRITICAL,
        )
        ev = _evaluator(_news_gate(events=[fomc]))
        decision = self._evaluate(ev, market_setup, calm_regime())
        assert decision.rejected
        assert "news gate" in decision.reason

    def test_full_pass_or_sizing_reject_is_clean(self, market_setup) -> None:
        # With a calm regime and clear gates, the decision is either a valid sized entry or a
        # clean sizing/edge rejection -- never an exception, and always with a reason.
        decision = self._evaluate(_evaluator(), market_setup, calm_regime())
        assert isinstance(decision.reason, str) and decision.reason
        if decision.enter:
            assert decision.position is not None
            assert decision.position.quantity >= 1
            # The stored condor's quantity matches the position quantity.
            assert decision.position.condor.quantity == decision.position.quantity

    def test_naive_now_rejected(self, market_setup) -> None:
        ev = _evaluator()
        with pytest.raises(ValidationError):
            ev.evaluate(
                now=dt.datetime(2024, 3, 1, 15, 0),
                symbol="SPX",
                distribution=market_setup["dist"],
                chain=market_setup["chain"],
                regime=calm_regime(),
                account=Account.open(starting_cash=100_000.0),
                position_id="P1",
                available_margin=50_000.0,
            )


def _position() -> OpenPosition:
    condor = IronCondor(90.0, 95.0, 105.0, 110.0, 10.0 / 252, quantity=2)
    return OpenPosition(
        position_id="P1",
        condor=condor,
        entry_credit=1.5,
        quantity=2,
        multiplier=100.0,
        entry_time=NOW,
        entry_spot=100.0,
    )


class TestExit:
    def test_profit_target(self) -> None:
        pos = _position()  # max profit = 300
        decision = evaluate_exit(
            pos,
            now=NOW + dt.timedelta(days=3),
            unrealized_pnl=200.0,
            config=ExitConfig(profit_target_fraction=0.5),
        )
        assert decision.exit_position
        assert decision.trigger == "profit_target"

    def test_stop_loss(self) -> None:
        pos = _position()  # credit collected = 300
        decision = evaluate_exit(
            pos,
            now=NOW + dt.timedelta(days=3),
            unrealized_pnl=-700.0,
            config=ExitConfig(stop_loss_credit_multiple=2.0),
        )
        assert decision.exit_position
        assert decision.trigger == "stop_loss"

    def test_regime_breach(self) -> None:
        pos = _position()
        decision = evaluate_exit(
            pos,
            now=NOW + dt.timedelta(days=2),
            unrealized_pnl=10.0,
            regime=stressed_regime(),
            config=ExitConfig(min_regime_low_prob=0.4),
        )
        assert decision.exit_position
        assert decision.trigger == "regime"

    def test_time_stop(self) -> None:
        pos = _position()  # 10 trading-day expiry (10/252 yr)
        # Advance enough calendar time that <= 1 trading day remains to expiry. Expiry is
        # ~14.5 calendar days (10 trading days / 252 * 365.25); advance ~14 calendar days.
        decision = evaluate_exit(
            pos,
            now=NOW + dt.timedelta(days=14.0),
            unrealized_pnl=10.0,
            config=ExitConfig(time_stop_days=1.0),
        )
        assert decision.exit_position
        assert decision.trigger == "time_stop"

    def test_hold(self) -> None:
        pos = _position()
        decision = evaluate_exit(
            pos,
            now=NOW + dt.timedelta(days=2),
            unrealized_pnl=20.0,
            regime=calm_regime(),
            config=ExitConfig(profit_target_fraction=0.5, stop_loss_credit_multiple=2.0),
        )
        assert decision.hold
        assert decision.trigger == "hold"

    def test_stop_loss_takes_priority(self) -> None:
        # A big loss fires stop-loss even if the regime is also bad.
        pos = _position()
        decision = evaluate_exit(
            pos,
            now=NOW + dt.timedelta(days=2),
            unrealized_pnl=-700.0,
            regime=stressed_regime(),
        )
        assert decision.trigger == "stop_loss"

    def test_naive_now_rejected(self) -> None:
        with pytest.raises(ValidationError):
            evaluate_exit(_position(), now=dt.datetime(2024, 3, 1), unrealized_pnl=0.0)
