"""Tests for news/event data structures."""

from __future__ import annotations

import datetime as dt

import pytest

from options_engine.core.errors import ValidationError
from options_engine.news.events import EventSeverity, NewsItem, ScheduledEvent

UTC = dt.UTC


class TestEventSeverity:
    def test_ordering(self) -> None:
        assert EventSeverity.NONE < EventSeverity.LOW < EventSeverity.CRITICAL
        assert EventSeverity.HIGH >= EventSeverity.MEDIUM


class TestNewsItem:
    def test_valid(self) -> None:
        item = NewsItem(
            timestamp=dt.datetime(2024, 1, 1, tzinfo=UTC),
            headline="Big news",
            source="wire",
            symbols=("spx", " spy "),
        )
        assert item.symbols == ("SPX", "SPY")  # normalized upper-case, trimmed
        assert "Big news" in item.text

    def test_rejects_naive_timestamp(self) -> None:
        with pytest.raises(ValidationError):
            NewsItem(timestamp=dt.datetime(2024, 1, 1), headline="x", source="wire")

    def test_rejects_empty_headline(self) -> None:
        with pytest.raises(ValidationError):
            NewsItem(timestamp=dt.datetime(2024, 1, 1, tzinfo=UTC), headline="  ", source="wire")

    def test_rejects_empty_source(self) -> None:
        with pytest.raises(ValidationError):
            NewsItem(timestamp=dt.datetime(2024, 1, 1, tzinfo=UTC), headline="x", source="")


class TestScheduledEvent:
    def test_market_wide(self) -> None:
        ev = ScheduledEvent(
            timestamp=dt.datetime(2024, 1, 1, tzinfo=UTC),
            name="FOMC",
            severity=EventSeverity.CRITICAL,
        )
        assert ev.is_market_wide
        assert ev.affects("ANYTHING")

    def test_symbol_specific(self) -> None:
        ev = ScheduledEvent(
            timestamp=dt.datetime(2024, 1, 1, tzinfo=UTC),
            name="AAPL earnings",
            severity=EventSeverity.MEDIUM,
            symbols=("AAPL",),
        )
        assert not ev.is_market_wide
        assert ev.affects("aapl")
        assert not ev.affects("MSFT")

    def test_rejects_naive_timestamp(self) -> None:
        with pytest.raises(ValidationError):
            ScheduledEvent(timestamp=dt.datetime(2024, 1, 1), name="x", severity=EventSeverity.LOW)

    def test_rejects_bad_severity(self) -> None:
        with pytest.raises(ValidationError):
            ScheduledEvent(
                timestamp=dt.datetime(2024, 1, 1, tzinfo=UTC), name="x", severity=3  # type: ignore[arg-type]
            )
