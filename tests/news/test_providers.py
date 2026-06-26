"""Tests for news/event providers."""

from __future__ import annotations

import datetime as dt

import pytest

from options_engine.core.config import SecretRef
from options_engine.core.errors import ConfigurationError, ValidationError
from options_engine.news.events import EventSeverity, NewsItem, ScheduledEvent
from options_engine.news.providers import (
    ReplayEventProvider,
    ReplayNewsProvider,
    RestNewsProvider,
)

UTC = dt.UTC


def _news(day: int) -> NewsItem:
    return NewsItem(
        timestamp=dt.datetime(2024, 1, day, tzinfo=UTC), headline=f"news {day}", source="wire"
    )


def _event(day: int) -> ScheduledEvent:
    return ScheduledEvent(
        timestamp=dt.datetime(2024, 1, day, tzinfo=UTC),
        name=f"event {day}",
        severity=EventSeverity.HIGH,
    )


class TestReplayNewsProvider:
    def test_returns_window_half_open(self) -> None:
        provider = ReplayNewsProvider([_news(1), _news(5), _news(10)])
        items = provider.get_news_between(
            dt.datetime(2024, 1, 5, tzinfo=UTC), dt.datetime(2024, 1, 10, tzinfo=UTC)
        )
        # [start, end): includes day 5, excludes day 10.
        assert len(items) == 1
        assert items[0].timestamp.day == 5

    def test_sorted_output(self) -> None:
        provider = ReplayNewsProvider([_news(10), _news(1), _news(5)])
        items = provider.get_news_between(
            dt.datetime(2024, 1, 1, tzinfo=UTC), dt.datetime(2024, 1, 31, tzinfo=UTC)
        )
        assert [i.timestamp.day for i in items] == [1, 5, 10]

    def test_rejects_naive_window(self) -> None:
        provider = ReplayNewsProvider([_news(1)])
        with pytest.raises(ValidationError):
            provider.get_news_between(dt.datetime(2024, 1, 1), dt.datetime(2024, 1, 2))

    def test_rejects_inverted_window(self) -> None:
        provider = ReplayNewsProvider([])
        with pytest.raises(ValidationError):
            provider.get_news_between(
                dt.datetime(2024, 1, 5, tzinfo=UTC), dt.datetime(2024, 1, 1, tzinfo=UTC)
            )


class TestReplayEventProvider:
    def test_window(self) -> None:
        provider = ReplayEventProvider([_event(1), _event(5), _event(10)])
        events = provider.get_events_between(
            dt.datetime(2024, 1, 4, tzinfo=UTC), dt.datetime(2024, 1, 11, tzinfo=UTC)
        )
        assert [e.timestamp.day for e in events] == [5, 10]


class TestRestNewsProvider:
    def test_requires_credentials_at_call(self, monkeypatch: pytest.MonkeyPatch) -> None:
        provider = RestNewsProvider(
            api_key=SecretRef(env_var="NEWS_API_KEY_TEST"), base_url="https://api.example.com/"
        )
        # No env var set -> resolving the secret fails fast.
        monkeypatch.delenv("NEWS_API_KEY_TEST", raising=False)
        with pytest.raises(ConfigurationError):
            provider.get_news_between(
                dt.datetime(2024, 1, 1, tzinfo=UTC), dt.datetime(2024, 1, 2, tzinfo=UTC)
            )

    def test_raises_not_implemented_with_credentials(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("NEWS_API_KEY_TEST", "secret")
        provider = RestNewsProvider(
            api_key=SecretRef(env_var="NEWS_API_KEY_TEST"), base_url="https://api.example.com"
        )
        # With credentials present, the unimplemented vendor call is the single seam.
        with pytest.raises(NotImplementedError):
            provider.get_news_between(
                dt.datetime(2024, 1, 1, tzinfo=UTC), dt.datetime(2024, 1, 2, tzinfo=UTC)
            )

    def test_rejects_bad_construction(self) -> None:
        with pytest.raises(ValidationError):
            RestNewsProvider(api_key="not-a-secretref", base_url="x")  # type: ignore[arg-type]
        with pytest.raises(ValidationError):
            RestNewsProvider(api_key=SecretRef(env_var="X"), base_url="  ")
