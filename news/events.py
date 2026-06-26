r"""News and scheduled-event data structures (SPEC §2.7).

The news/event gate consumes two kinds of signal:

* **Breaking news** -- unscheduled items (a headline) with a timestamp, source, and text,
  classified for relevance and severity by :mod:`options_engine.news.classifier`.
* **Scheduled events** -- known calendar items (FOMC, CPI, OPEX, constituent earnings) with
  a timestamp and severity, around which the strategy must avoid opening new risk.

Both are immutable and self-validating so a malformed feed item can never silently change a
trading decision. Timestamps are timezone-aware UTC by contract; naive datetimes are
rejected, because silently mixing naive/aware times is a classic, costly bug in
event-driven trading systems.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from enum import IntEnum

from ..core.errors import ValidationError

__all__ = ["EventSeverity", "NewsItem", "ScheduledEvent"]


class EventSeverity(IntEnum):
    """Ordered severity of a news item or scheduled event.

    ``IntEnum`` so severities compare and threshold naturally (``>=``). ``NONE`` means the
    item is not material to trading; ``CRITICAL`` is a market-moving shock.
    """

    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


def _require_aware_utc(timestamp: _dt.datetime, *, name: str) -> _dt.datetime:
    """Return the timestamp if timezone-aware, else raise (no silent naive datetimes)."""
    if not isinstance(timestamp, _dt.datetime):
        raise ValidationError(
            f"{name} must be a datetime", context={"type": type(timestamp).__name__}
        )
    if timestamp.tzinfo is None or timestamp.tzinfo.utcoffset(timestamp) is None:
        raise ValidationError(f"{name} must be timezone-aware", context={"name": name})
    return timestamp


@dataclass(frozen=True, slots=True)
class NewsItem:
    """A single breaking-news item from a provider.

    Parameters
    ----------
    timestamp:
        Timezone-aware UTC publication time.
    headline:
        The headline text (non-empty).
    body:
        Optional longer text; defaults to empty.
    source:
        Provider/source identifier (non-empty).
    symbols:
        Tickers the item is tagged with (may be empty if untagged).
    """

    timestamp: _dt.datetime
    headline: str
    source: str
    body: str = ""
    symbols: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        _require_aware_utc(self.timestamp, name="timestamp")
        if not self.headline.strip():
            raise ValidationError("headline must be non-empty", context={})
        if not self.source.strip():
            raise ValidationError("source must be non-empty", context={})
        # Normalize symbols to an upper-case tuple for consistent matching.
        normalized = tuple(s.strip().upper() for s in self.symbols if s.strip())
        object.__setattr__(self, "symbols", normalized)

    @property
    def text(self) -> str:
        """Concatenated headline and body for classification."""
        return f"{self.headline}. {self.body}".strip()


@dataclass(frozen=True, slots=True)
class ScheduledEvent:
    """A known, scheduled market event with a severity.

    Parameters
    ----------
    timestamp:
        Timezone-aware UTC time of the event.
    name:
        Human-readable event name (e.g. ``"FOMC rate decision"``).
    severity:
        How market-moving the event is expected to be.
    symbols:
        Affected tickers (empty = market-wide, e.g. FOMC/CPI).
    """

    timestamp: _dt.datetime
    name: str
    severity: EventSeverity
    symbols: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        _require_aware_utc(self.timestamp, name="timestamp")
        if not self.name.strip():
            raise ValidationError("name must be non-empty", context={})
        if not isinstance(self.severity, EventSeverity):
            raise ValidationError(
                "severity must be an EventSeverity", context={"type": type(self.severity).__name__}
            )
        normalized = tuple(s.strip().upper() for s in self.symbols if s.strip())
        object.__setattr__(self, "symbols", normalized)

    @property
    def is_market_wide(self) -> bool:
        """True if the event affects the whole market (no specific symbols)."""
        return len(self.symbols) == 0

    def affects(self, symbol: str) -> bool:
        """Return True if this event is relevant to the given symbol."""
        return self.is_market_wide or symbol.strip().upper() in self.symbols
