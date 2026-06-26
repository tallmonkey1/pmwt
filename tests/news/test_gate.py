"""Tests for the combined news/event trade gate."""

from __future__ import annotations

import datetime as dt

import pytest

from options_engine.core.errors import ValidationError
from options_engine.news.events import EventSeverity, NewsItem, ScheduledEvent
from options_engine.news.gate import NewsGate, NewsGateConfig
from options_engine.news.providers import ReplayEventProvider, ReplayNewsProvider

UTC = dt.UTC
UNIVERSE = frozenset({"SPX", "SPY"})


def _gate(news: list[NewsItem], events: list[ScheduledEvent], **cfg_kwargs) -> NewsGate:
    return NewsGate(
        news_provider=ReplayNewsProvider(news),
        event_provider=ReplayEventProvider(events),
        universe=UNIVERSE,
        config=NewsGateConfig(**cfg_kwargs) if cfg_kwargs else None,
    )


class TestCooloff:
    def _shock(self) -> NewsItem:
        # Friday 2024-03-01, a critical market-wide headline.
        return NewsItem(
            timestamp=dt.datetime(2024, 3, 1, 14, 0, tzinfo=UTC),
            headline="Fed announces emergency rate hike",
            source="wire",
        )

    def test_blocks_during_cooloff(self) -> None:
        gate = _gate([self._shock()], [], cooloff_trading_days=5)
        # Monday 2024-03-04 = 1 trading day later.
        decision = gate.evaluate(dt.datetime(2024, 3, 4, 14, 0, tzinfo=UTC), symbol="SPX")
        assert decision.blocked
        assert decision.cooloff_active
        assert decision.trading_days_since_news == 1

    def test_allows_after_cooloff(self) -> None:
        gate = _gate([self._shock()], [], cooloff_trading_days=5)
        # 2024-03-08 (Fri) = 5 trading days after the shock.
        decision = gate.evaluate(dt.datetime(2024, 3, 8, 14, 0, tzinfo=UTC), symbol="SPX")
        assert decision.allow_new_risk
        assert not decision.cooloff_active

    def test_weekend_does_not_shorten_cooloff(self) -> None:
        gate = _gate([self._shock()], [], cooloff_trading_days=5)
        # Monday after the shock is only 1 trading day, despite 3 calendar days passing.
        monday = gate.evaluate(dt.datetime(2024, 3, 4, 14, 0, tzinfo=UTC), symbol="SPX")
        assert monday.trading_days_since_news == 1

    def test_benign_news_does_not_trigger(self) -> None:
        benign = NewsItem(
            timestamp=dt.datetime(2024, 3, 1, 14, 0, tzinfo=UTC),
            headline="Company schedules summer picnic",
            source="wire",
        )
        gate = _gate([benign], [], cooloff_trading_days=5)
        decision = gate.evaluate(dt.datetime(2024, 3, 4, 14, 0, tzinfo=UTC), symbol="SPX")
        assert decision.allow_new_risk

    def test_symbol_specific_news_scoped(self) -> None:
        # A symbol-tagged shock only cools off that symbol.
        shock = NewsItem(
            timestamp=dt.datetime(2024, 3, 1, 14, 0, tzinfo=UTC),
            headline="rate hike fears hit SPX",
            source="wire",
            symbols=("SPX",),
        )
        gate = _gate([shock], [], cooloff_trading_days=5)
        now = dt.datetime(2024, 3, 4, 14, 0, tzinfo=UTC)
        assert gate.evaluate(now, symbol="SPX").blocked
        assert gate.evaluate(now, symbol="SPY").allow_new_risk

    def test_zero_cooloff_disables_layer(self) -> None:
        gate = _gate([self._shock()], [], cooloff_trading_days=0)
        decision = gate.evaluate(dt.datetime(2024, 3, 1, 15, 0, tzinfo=UTC), symbol="SPX")
        assert decision.allow_new_risk


class TestBlackoutLayer:
    def test_blackout_blocks(self) -> None:
        fomc = ScheduledEvent(
            timestamp=dt.datetime(2024, 3, 20, 18, 0, tzinfo=UTC),
            name="FOMC",
            severity=EventSeverity.CRITICAL,
        )
        gate = _gate([], [fomc])
        decision = gate.evaluate(dt.datetime(2024, 3, 20, 6, 0, tzinfo=UTC), symbol="SPX")
        assert decision.blocked
        assert decision.blackout_active

    def test_clear_when_no_events_or_news(self) -> None:
        gate = _gate([], [])
        decision = gate.evaluate(dt.datetime(2024, 3, 20, 6, 0, tzinfo=UTC), symbol="SPX")
        assert decision.allow_new_risk


class TestGateValidation:
    def test_rejects_empty_universe(self) -> None:
        with pytest.raises(ValidationError):
            NewsGate(
                news_provider=ReplayNewsProvider([]),
                event_provider=ReplayEventProvider([]),
                universe=frozenset(),
            )

    def test_rejects_naive_now(self) -> None:
        gate = _gate([], [])
        with pytest.raises(ValidationError):
            gate.evaluate(dt.datetime(2024, 3, 20, 6, 0), symbol="SPX")

    def test_config_rejects_negative_cooloff(self) -> None:
        with pytest.raises(ValidationError):
            NewsGateConfig(cooloff_trading_days=-1)

    def test_config_rejects_huge_cooloff(self) -> None:
        with pytest.raises(ValidationError):
            NewsGateConfig(cooloff_trading_days=10_000)
