r"""News and economic-calendar provider interfaces (SPEC §2.7, §8).

External data sources sit behind typed interfaces with multiple implementations: a
replay/recorded provider for the history backtest, a simulated provider for synthetic
scenarios, and (declared but not wired without credentials) a real-API provider whose key is
injected via the environment. This is the dependency-injection seam that guarantees
backtest <-> live parity (SPEC §8): the gate consumes the same interface in every mode; only
the data source changes.

Providers are *causal*: :meth:`get_news_between` / :meth:`get_events_between` return only
items in the requested half-open window ``[start, end)``, so a backtest can never read a
headline before it was published.
"""

from __future__ import annotations

import datetime as _dt
from abc import ABC, abstractmethod
from collections.abc import Iterable

from ..core.config import SecretRef
from ..core.errors import ValidationError
from .events import NewsItem, ScheduledEvent

__all__ = [
    "EventProvider",
    "NewsProvider",
    "ReplayEventProvider",
    "ReplayNewsProvider",
    "RestNewsProvider",
]


def _validate_window(start: _dt.datetime, end: _dt.datetime) -> None:
    """Validate a causal time window (both aware, start <= end)."""
    for name, ts in (("start", start), ("end", end)):
        if ts.tzinfo is None or ts.tzinfo.utcoffset(ts) is None:
            raise ValidationError(f"{name} must be timezone-aware", context={"name": name})
    if end < start:
        raise ValidationError(
            "end must not precede start",
            context={"start": start.isoformat(), "end": end.isoformat()},
        )


class NewsProvider(ABC):
    """Abstract source of breaking-news items."""

    @abstractmethod
    def get_news_between(self, start: _dt.datetime, end: _dt.datetime) -> tuple[NewsItem, ...]:
        """Return news items published in the half-open window ``[start, end)``."""
        raise NotImplementedError


class EventProvider(ABC):
    """Abstract source of scheduled calendar events."""

    @abstractmethod
    def get_events_between(
        self, start: _dt.datetime, end: _dt.datetime
    ) -> tuple[ScheduledEvent, ...]:
        """Return scheduled events occurring in the half-open window ``[start, end)``."""
        raise NotImplementedError


class ReplayNewsProvider(NewsProvider):
    """In-memory news provider over a fixed, pre-sorted set of items (backtest/testing)."""

    def __init__(self, items: Iterable[NewsItem]) -> None:
        sorted_items = tuple(sorted(items, key=lambda n: n.timestamp))
        for item in sorted_items:
            if not isinstance(item, NewsItem):
                raise ValidationError(
                    "all items must be NewsItem", context={"type": type(item).__name__}
                )
        self._items = sorted_items

    def get_news_between(self, start: _dt.datetime, end: _dt.datetime) -> tuple[NewsItem, ...]:
        _validate_window(start, end)
        return tuple(item for item in self._items if start <= item.timestamp < end)


class ReplayEventProvider(EventProvider):
    """In-memory scheduled-event provider over a fixed event set (backtest/testing)."""

    def __init__(self, events: Iterable[ScheduledEvent]) -> None:
        sorted_events = tuple(sorted(events, key=lambda e: e.timestamp))
        for event in sorted_events:
            if not isinstance(event, ScheduledEvent):
                raise ValidationError(
                    "all events must be ScheduledEvent", context={"type": type(event).__name__}
                )
        self._events = sorted_events

    def get_events_between(
        self, start: _dt.datetime, end: _dt.datetime
    ) -> tuple[ScheduledEvent, ...]:
        _validate_window(start, end)
        return tuple(e for e in self._events if start <= e.timestamp < end)


class RestNewsProvider(NewsProvider):
    """Real news-API provider skeleton (credential-gated, SPEC §8).

    The provider holds a :class:`SecretRef` to the API key (resolved from the environment at
    call time, never stored in config or logs) and the base endpoint. The actual HTTP call is
    intentionally left as the single integration point that requires live credentials and a
    chosen vendor -- everything around it (windowing, parsing contract, error handling) is the
    same interface the rest of the system already depends on. Without credentials this
    provider raises a clear, actionable error rather than silently returning empty data.
    """

    def __init__(self, *, api_key: SecretRef, base_url: str) -> None:
        if not isinstance(api_key, SecretRef):
            raise ValidationError("api_key must be a SecretRef", context={})
        if not base_url.strip():
            raise ValidationError("base_url must be non-empty", context={})
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")

    def get_news_between(self, start: _dt.datetime, end: _dt.datetime) -> tuple[NewsItem, ...]:
        _validate_window(start, end)
        # Resolve the key (fails fast if the env var is unset), then perform the vendor call.
        # The HTTP/parse step is the one credential-and-vendor-specific integration point.
        _ = self._api_key.resolve()
        raise NotImplementedError(
            "RestNewsProvider requires a configured news vendor and credentials; "
            "wire the vendor HTTP call here. Use ReplayNewsProvider for backtests."
        )
