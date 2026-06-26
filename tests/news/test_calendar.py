"""Tests for scheduled-event blackout windows."""

from __future__ import annotations

import datetime as dt

import pytest

from options_engine.core.errors import ValidationError
from options_engine.news.calendar import BlackoutConfig, is_within_blackout
from options_engine.news.events import EventSeverity, ScheduledEvent
from options_engine.news.providers import ReplayEventProvider

UTC = dt.UTC


def _provider(*events: ScheduledEvent) -> ReplayEventProvider:
    return ReplayEventProvider(list(events))


class TestIsWithinBlackout:
    def _fomc(self) -> ScheduledEvent:
        return ScheduledEvent(
            timestamp=dt.datetime(2024, 3, 20, 18, 0, tzinfo=UTC),
            name="FOMC",
            severity=EventSeverity.CRITICAL,
        )

    def test_inside_lead_window(self) -> None:
        provider = _provider(self._fomc())
        cfg = BlackoutConfig(lead_time=dt.timedelta(hours=24))
        now = dt.datetime(2024, 3, 20, 6, 0, tzinfo=UTC)  # 12h before
        result = is_within_blackout(now, symbol="SPX", provider=provider, config=cfg)
        assert result.in_blackout
        assert result.triggering_event is not None

    def test_outside_window(self) -> None:
        provider = _provider(self._fomc())
        cfg = BlackoutConfig(lead_time=dt.timedelta(hours=24))
        now = dt.datetime(2024, 3, 18, 6, 0, tzinfo=UTC)  # > 2 days before
        result = is_within_blackout(now, symbol="SPX", provider=provider, config=cfg)
        assert not result.in_blackout

    def test_cooldown_after(self) -> None:
        provider = _provider(self._fomc())
        cfg = BlackoutConfig(cooldown_after=dt.timedelta(hours=3))
        now = dt.datetime(2024, 3, 20, 20, 0, tzinfo=UTC)  # 2h after
        assert is_within_blackout(now, symbol="SPX", provider=provider, config=cfg).in_blackout

    def test_severity_threshold(self) -> None:
        low_event = ScheduledEvent(
            timestamp=dt.datetime(2024, 3, 20, 18, 0, tzinfo=UTC),
            name="minor",
            severity=EventSeverity.LOW,
        )
        provider = _provider(low_event)
        cfg = BlackoutConfig(min_severity=EventSeverity.HIGH)
        now = dt.datetime(2024, 3, 20, 12, 0, tzinfo=UTC)
        assert not is_within_blackout(now, symbol="SPX", provider=provider, config=cfg).in_blackout

    def test_symbol_specific_event(self) -> None:
        earnings = ScheduledEvent(
            timestamp=dt.datetime(2024, 3, 20, 18, 0, tzinfo=UTC),
            name="AAPL earnings",
            severity=EventSeverity.HIGH,
            symbols=("AAPL",),
        )
        provider = _provider(earnings)
        cfg = BlackoutConfig(lead_time=dt.timedelta(hours=24))
        now = dt.datetime(2024, 3, 20, 12, 0, tzinfo=UTC)
        assert is_within_blackout(now, symbol="AAPL", provider=provider, config=cfg).in_blackout
        assert not is_within_blackout(now, symbol="SPX", provider=provider, config=cfg).in_blackout

    def test_rejects_naive_now(self) -> None:
        provider = _provider(self._fomc())
        with pytest.raises(ValidationError):
            is_within_blackout(dt.datetime(2024, 3, 20, 12), symbol="SPX", provider=provider)


class TestBlackoutConfig:
    def test_rejects_negative_lead(self) -> None:
        with pytest.raises(ValidationError):
            BlackoutConfig(lead_time=dt.timedelta(hours=-1))

    def test_rejects_bad_severity(self) -> None:
        with pytest.raises(ValidationError):
            BlackoutConfig(min_severity=2)  # type: ignore[arg-type]
