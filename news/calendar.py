r"""Economic-calendar utilities and scheduled-event blackout windows (SPEC §2.7).

Layer (a) of the news gate is the *scheduled-event calendar*: known, dated risk events
(FOMC, CPI, OPEX, constituent earnings) before which the strategy must not open new risk.
This module provides:

* :func:`is_within_blackout` -- whether a decision time falls inside the pre-event blackout
  window of any material scheduled event from a provider, and
* :class:`BlackoutConfig` -- the lead-time and severity threshold that define a blackout.

Earnings and exchange-specific calendars are supplied through an
:class:`~options_engine.news.providers.EventProvider`; this module contains only the
pure, testable windowing logic, keeping data acquisition behind the provider seam.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass

from ..core.errors import ValidationError
from .events import EventSeverity, ScheduledEvent
from .providers import EventProvider

__all__ = ["BlackoutConfig", "BlackoutResult", "is_within_blackout"]


@dataclass(frozen=True, slots=True)
class BlackoutConfig:
    """Defines the pre-event blackout window."""

    #: How long before a scheduled event new risk is suspended.
    lead_time: _dt.timedelta = _dt.timedelta(hours=24)
    #: How long after a scheduled event the blackout persists (event digestion).
    cooldown_after: _dt.timedelta = _dt.timedelta(hours=2)
    #: Minimum severity for an event to trigger a blackout.
    min_severity: EventSeverity = EventSeverity.HIGH

    def __post_init__(self) -> None:
        if self.lead_time < _dt.timedelta(0):
            raise ValidationError("lead_time must be non-negative", context={})
        if self.cooldown_after < _dt.timedelta(0):
            raise ValidationError("cooldown_after must be non-negative", context={})
        if not isinstance(self.min_severity, EventSeverity):
            raise ValidationError("min_severity must be an EventSeverity", context={})


@dataclass(frozen=True, slots=True)
class BlackoutResult:
    """Outcome of a blackout check, naming the triggering event for auditability."""

    in_blackout: bool
    triggering_event: ScheduledEvent | None
    reason: str


def is_within_blackout(
    now: _dt.datetime,
    *,
    symbol: str,
    provider: EventProvider,
    config: BlackoutConfig | None = None,
) -> BlackoutResult:
    """Return whether ``now`` is inside the blackout window of a material scheduled event.

    Parameters
    ----------
    now:
        Timezone-aware decision time.
    symbol:
        The symbol under consideration (an event is relevant if market-wide or tagged with
        this symbol).
    provider:
        Source of scheduled events.
    config:
        Blackout window definition; defaults to :class:`BlackoutConfig`.

    Returns
    -------
    BlackoutResult
        Whether ``now`` is in blackout and, if so, which event triggered it.
    """
    if now.tzinfo is None or now.tzinfo.utcoffset(now) is None:
        raise ValidationError("now must be timezone-aware", context={})
    cfg = config or BlackoutConfig()

    # Query the window that could possibly cover 'now': events whose blackout could include
    # it span from (now - cooldown_after) to (now + lead_time).
    window_start = now - cfg.cooldown_after
    window_end = now + cfg.lead_time + _dt.timedelta(microseconds=1)
    events = provider.get_events_between(window_start, window_end)

    for event in events:
        if event.severity < cfg.min_severity:
            continue
        if not event.affects(symbol):
            continue
        blackout_start = event.timestamp - cfg.lead_time
        blackout_end = event.timestamp + cfg.cooldown_after
        if blackout_start <= now <= blackout_end:
            return BlackoutResult(
                in_blackout=True,
                triggering_event=event,
                reason=(
                    f"within blackout for '{event.name}' "
                    f"(severity={event.severity.name}) at {event.timestamp.isoformat()}"
                ),
            )

    return BlackoutResult(
        in_blackout=False, triggering_event=None, reason="no material scheduled event nearby"
    )
