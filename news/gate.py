r"""The news/event trade gate (SPEC §2.7).

Combines the two layers of event risk into a single go/no-go for opening new positions:

* **Scheduled-event blackout** (layer a) -- no new risk inside the pre/post window of a
  material calendar event (FOMC, CPI, OPEX, earnings).
* **Breaking-news cool-off** (layer b) -- on a *material* breaking-news item the gate enters
  a strict cool-off (default **5 trading days**, configurable) during which no new positions
  are opened; existing positions move to defensive management only (the strategy layer
  enforces that downstream).

The cool-off is measured in **trading days**, not calendar days, so a Friday shock correctly
suspends trading through the following week rather than expiring over the weekend. The gate
is stateful only in the sense that it scans the provider for the most recent material item;
it stores no hidden mutable state, so it is reproducible and testable.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field

from ..core.errors import ValidationError
from ..core.timegrid import TRADING_DAYS_PER_YEAR
from .calendar import BlackoutConfig, is_within_blackout
from .classifier import KeywordNewsClassifier, NewsClassifier
from .events import EventSeverity
from .providers import EventProvider, NewsProvider

__all__ = ["NewsGate", "NewsGateConfig", "NewsGateDecision"]


def _count_trading_days(start: _dt.datetime, end: _dt.datetime) -> int:
    """Return the number of trading days (Mon-Fri) strictly elapsed from ``start`` to ``end``.

    Counts business days in the half-open interval ``(start_date, end_date]`` by date, which
    is the natural "how many trading sessions since the shock" measure. Weekends are skipped;
    exchange holidays are not modelled here (a conservative over-count of available sessions
    would *shorten* the cool-off, so if anything callers should supply a holiday-aware clock
    -- documented as a known limitation).
    """
    if end <= start:
        return 0
    elapsed = 0
    day = start.date()
    last = end.date()
    while day < last:
        day += _dt.timedelta(days=1)
        if day.weekday() < 5:  # Monday=0 .. Friday=4
            elapsed += 1
    return elapsed


@dataclass(frozen=True, slots=True)
class NewsGateConfig:
    """Configuration for the news/event gate."""

    #: Cool-off length after a material breaking-news item, in *trading* days.
    cooloff_trading_days: int = 5
    #: Minimum classifier severity for a breaking item to trigger the cool-off.
    min_breaking_severity: EventSeverity = EventSeverity.HIGH
    #: How far back to scan for breaking news (must exceed the cool-off span).
    lookback: _dt.timedelta = _dt.timedelta(days=14)
    #: Scheduled-event blackout configuration.
    blackout: BlackoutConfig = field(default_factory=BlackoutConfig)

    def __post_init__(self) -> None:
        if self.cooloff_trading_days < 0:
            raise ValidationError(
                "cooloff_trading_days must be non-negative",
                context={"cooloff_trading_days": self.cooloff_trading_days},
            )
        if self.cooloff_trading_days > TRADING_DAYS_PER_YEAR:
            raise ValidationError(
                "cooloff_trading_days is implausibly large",
                context={"cooloff_trading_days": self.cooloff_trading_days},
            )
        if not isinstance(self.min_breaking_severity, EventSeverity):
            raise ValidationError("min_breaking_severity must be an EventSeverity", context={})
        if self.lookback <= _dt.timedelta(0):
            raise ValidationError("lookback must be positive", context={})


@dataclass(frozen=True, slots=True)
class NewsGateDecision:
    """The news gate's go/no-go with an auditable reason."""

    allow_new_risk: bool
    reason: str
    cooloff_active: bool
    blackout_active: bool
    trading_days_since_news: int | None

    @property
    def blocked(self) -> bool:
        """True if new risk is blocked for any reason."""
        return not self.allow_new_risk


class NewsGate:
    """Evaluates breaking-news cool-off and scheduled-event blackout for new risk.

    Parameters
    ----------
    news_provider:
        Source of breaking-news items.
    event_provider:
        Source of scheduled calendar events.
    universe:
        The set of symbols the strategy trades (for relevance classification).
    classifier:
        Breaking-news classifier; defaults to :class:`KeywordNewsClassifier`.
    config:
        Gate configuration; defaults to :class:`NewsGateConfig`.
    """

    def __init__(
        self,
        *,
        news_provider: NewsProvider,
        event_provider: EventProvider,
        universe: frozenset[str],
        classifier: NewsClassifier | None = None,
        config: NewsGateConfig | None = None,
    ) -> None:
        if not isinstance(news_provider, NewsProvider):
            raise ValidationError("news_provider must be a NewsProvider", context={})
        if not isinstance(event_provider, EventProvider):
            raise ValidationError("event_provider must be an EventProvider", context={})
        normalized = frozenset(s.strip().upper() for s in universe if s.strip())
        if not normalized:
            raise ValidationError("universe must be non-empty", context={})
        self._news_provider = news_provider
        self._event_provider = event_provider
        self._universe = normalized
        self._classifier = classifier or KeywordNewsClassifier()
        self._config = config or NewsGateConfig()

    def evaluate(self, now: _dt.datetime, *, symbol: str) -> NewsGateDecision:
        """Return the go/no-go decision for opening new risk in ``symbol`` at time ``now``.

        Blocks if either a scheduled-event blackout is active or a material breaking-news
        cool-off is still in effect. The blackout is checked first because it is the cheaper,
        more certain signal.
        """
        if now.tzinfo is None or now.tzinfo.utcoffset(now) is None:
            raise ValidationError("now must be timezone-aware", context={})
        sym = symbol.strip().upper()
        if not sym:
            raise ValidationError("symbol must be non-empty", context={})

        # Layer (a): scheduled-event blackout.
        blackout = is_within_blackout(
            now, symbol=sym, provider=self._event_provider, config=self._config.blackout
        )
        if blackout.in_blackout:
            return NewsGateDecision(
                allow_new_risk=False,
                reason=blackout.reason,
                cooloff_active=False,
                blackout_active=True,
                trading_days_since_news=None,
            )

        # Layer (b): breaking-news cool-off.
        most_recent_days = self._trading_days_since_material_news(now, symbol=sym)
        if most_recent_days is not None and most_recent_days < self._config.cooloff_trading_days:
            return NewsGateDecision(
                allow_new_risk=False,
                reason=(
                    f"breaking-news cool-off active: {most_recent_days} of "
                    f"{self._config.cooloff_trading_days} trading days elapsed"
                ),
                cooloff_active=True,
                blackout_active=False,
                trading_days_since_news=most_recent_days,
            )

        return NewsGateDecision(
            allow_new_risk=True,
            reason="no active blackout or cool-off",
            cooloff_active=False,
            blackout_active=False,
            trading_days_since_news=most_recent_days,
        )

    def _trading_days_since_material_news(self, now: _dt.datetime, *, symbol: str) -> int | None:
        """Return trading days since the most recent material news item, or None if none."""
        start = now - self._config.lookback
        items = self._news_provider.get_news_between(start, now)
        most_recent: _dt.datetime | None = None
        for item in items:
            # Restrict relevance to the queried symbol plus market-wide items.
            assessment = self._classifier.classify(item, universe=self._universe)
            if not assessment.is_relevant:
                continue
            if assessment.severity < self._config.min_breaking_severity:
                continue
            if self._item_applies_to_symbol(item, symbol) and (
                most_recent is None or item.timestamp > most_recent
            ):
                most_recent = item.timestamp
        if most_recent is None:
            return None
        return _count_trading_days(most_recent, now)

    def _item_applies_to_symbol(self, item: object, symbol: str) -> bool:
        """Return True if an item is symbol-specific to ``symbol`` or market-wide.

        Symbol-tagged items only apply to their tags; untagged items are treated as
        market-wide and apply to every symbol (the conservative risk choice).
        """
        symbols = getattr(item, "symbols", ())
        if not symbols:
            return True
        return symbol in symbols
