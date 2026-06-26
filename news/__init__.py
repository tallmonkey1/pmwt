"""News and event risk gate (SPEC §2.7).

Two-layer event-risk control: a scheduled-event blackout (FOMC/CPI/OPEX/earnings) and a
breaking-news cool-off (default 5 trading days), behind provider interfaces so the same gate
runs in backtest and live.

Public surface:

* Events: :class:`NewsItem`, :class:`ScheduledEvent`, :class:`EventSeverity`.
* Classifier: :class:`NewsClassifier`, :class:`KeywordNewsClassifier`,
  :class:`NewsAssessment`.
* Providers: :class:`NewsProvider`, :class:`EventProvider`, :class:`ReplayNewsProvider`,
  :class:`ReplayEventProvider`, :class:`RestNewsProvider`.
* Calendar: :func:`is_within_blackout`, :class:`BlackoutConfig`, :class:`BlackoutResult`.
* Gate: :class:`NewsGate`, :class:`NewsGateConfig`, :class:`NewsGateDecision`.
"""

from __future__ import annotations

from .calendar import BlackoutConfig, BlackoutResult, is_within_blackout
from .classifier import (
    DEFAULT_SEVERITY_LEXICON,
    KeywordNewsClassifier,
    NewsAssessment,
    NewsClassifier,
)
from .events import EventSeverity, NewsItem, ScheduledEvent
from .gate import NewsGate, NewsGateConfig, NewsGateDecision
from .providers import (
    EventProvider,
    NewsProvider,
    ReplayEventProvider,
    ReplayNewsProvider,
    RestNewsProvider,
)

__all__ = [
    "DEFAULT_SEVERITY_LEXICON",
    "BlackoutConfig",
    "BlackoutResult",
    "EventProvider",
    "EventSeverity",
    "KeywordNewsClassifier",
    "NewsAssessment",
    "NewsClassifier",
    "NewsGate",
    "NewsGateConfig",
    "NewsGateDecision",
    "NewsItem",
    "NewsProvider",
    "ReplayEventProvider",
    "ReplayNewsProvider",
    "RestNewsProvider",
    "ScheduledEvent",
    "is_within_blackout",
]
