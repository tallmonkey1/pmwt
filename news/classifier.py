r"""Breaking-news relevance and severity classification (SPEC §2.7).

The spec allows "keyword recognition or AI" for breaking-news detection. We implement a
**deterministic, auditable keyword-and-lexicon classifier** as the production default, behind
a small abstract interface so a transformer-based relevance/sentiment model can be dropped
in later without touching the gate. The keyword approach is the right *default* for an
institutional risk control: it is transparent, has no inference dependency or latency, never
produces a non-deterministic veto, and its decisions can be explained to a risk committee
("blocked because the headline matched CRITICAL term 'rate hike' and ticker SPX").

The classifier scores a :class:`NewsItem` on two axes:

* **Relevance** -- is the item about a symbol we trade or the broad market?
* **Severity** -- how market-moving is it, from a weighted lexicon of risk terms?

It returns a :class:`NewsAssessment` combining both, which the gate thresholds.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from ..core.errors import ValidationError
from .events import EventSeverity, NewsItem

__all__ = [
    "DEFAULT_SEVERITY_LEXICON",
    "KeywordNewsClassifier",
    "NewsAssessment",
    "NewsClassifier",
]

# Default lexicon mapping lower-case risk terms to a severity. Curated for index-options
# tail risk: monetary-policy surprises, macro shocks, and systemic events dominate.
DEFAULT_SEVERITY_LEXICON: dict[str, EventSeverity] = {
    # Critical, market-moving shocks.
    "rate hike": EventSeverity.CRITICAL,
    "rate cut": EventSeverity.CRITICAL,
    "emergency meeting": EventSeverity.CRITICAL,
    "default": EventSeverity.CRITICAL,
    "crash": EventSeverity.CRITICAL,
    "circuit breaker": EventSeverity.CRITICAL,
    "war": EventSeverity.CRITICAL,
    "invasion": EventSeverity.CRITICAL,
    "downgrade": EventSeverity.CRITICAL,
    "bankruptcy": EventSeverity.CRITICAL,
    "bank run": EventSeverity.CRITICAL,
    "contagion": EventSeverity.CRITICAL,
    # High-severity macro / policy.
    "inflation": EventSeverity.HIGH,
    "cpi": EventSeverity.HIGH,
    "fomc": EventSeverity.HIGH,
    "federal reserve": EventSeverity.HIGH,
    "jobs report": EventSeverity.HIGH,
    "recession": EventSeverity.HIGH,
    "tariff": EventSeverity.HIGH,
    "sanctions": EventSeverity.HIGH,
    "guidance cut": EventSeverity.HIGH,
    "profit warning": EventSeverity.HIGH,
    # Medium-severity items.
    "earnings": EventSeverity.MEDIUM,
    "merger": EventSeverity.MEDIUM,
    "acquisition": EventSeverity.MEDIUM,
    "lawsuit": EventSeverity.MEDIUM,
    "investigation": EventSeverity.MEDIUM,
    "volatility": EventSeverity.MEDIUM,
    "selloff": EventSeverity.MEDIUM,
}


@dataclass(frozen=True, slots=True)
class NewsAssessment:
    """The classifier's verdict on a news item."""

    severity: EventSeverity
    is_relevant: bool
    matched_terms: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_material(self) -> bool:
        """True if the item is relevant and at least MEDIUM severity."""
        return self.is_relevant and self.severity >= EventSeverity.MEDIUM


class NewsClassifier(ABC):
    """Abstract interface for breaking-news classifiers.

    A concrete classifier maps a :class:`NewsItem` (plus the universe of traded symbols) to a
    :class:`NewsAssessment`. The abstraction lets the gate stay agnostic to whether the
    classifier is keyword-based or model-based.
    """

    @abstractmethod
    def classify(self, item: NewsItem, *, universe: frozenset[str]) -> NewsAssessment:
        """Return the assessment of a news item for a given traded-symbol universe."""
        raise NotImplementedError


class KeywordNewsClassifier(NewsClassifier):
    """Deterministic keyword-and-lexicon news classifier.

    Parameters
    ----------
    lexicon:
        Mapping from risk terms (lower-case) to severities. Defaults to
        :data:`DEFAULT_SEVERITY_LEXICON`.
    treat_untagged_as_market_wide:
        If True, an item with no symbol tags is considered relevant to the whole market
        (the conservative choice for a risk control). If False, only symbol-tagged or
        symbol-mentioning items are relevant.
    """

    def __init__(
        self,
        *,
        lexicon: dict[str, EventSeverity] | None = None,
        treat_untagged_as_market_wide: bool = True,
    ) -> None:
        source = lexicon if lexicon is not None else DEFAULT_SEVERITY_LEXICON
        if not source:
            raise ValidationError("lexicon must be non-empty", context={})
        # Pre-compile word-boundary patterns for each term for robust, case-insensitive
        # matching (so "default" does not match "defaulted" spuriously, but "rate hike"
        # matches as a phrase).
        self._terms: tuple[tuple[str, EventSeverity, re.Pattern[str]], ...] = tuple(
            (term, severity, re.compile(rf"(?<!\w){re.escape(term)}(?!\w)", re.IGNORECASE))
            for term, severity in source.items()
        )
        self._treat_untagged_as_market_wide = treat_untagged_as_market_wide

    def classify(self, item: NewsItem, *, universe: frozenset[str]) -> NewsAssessment:
        """Classify a news item against the traded-symbol universe."""
        if not isinstance(item, NewsItem):
            raise ValidationError("item must be a NewsItem", context={"type": type(item).__name__})
        normalized_universe = frozenset(s.strip().upper() for s in universe if s.strip())

        relevant = self._is_relevant(item, normalized_universe)

        text = item.text
        matched: list[str] = []
        max_severity = EventSeverity.NONE
        for term, severity, pattern in self._terms:
            if pattern.search(text):
                matched.append(term)
                max_severity = max(max_severity, severity)

        return NewsAssessment(
            severity=max_severity,
            is_relevant=relevant,
            matched_terms=tuple(matched),
        )

    def _is_relevant(self, item: NewsItem, universe: frozenset[str]) -> bool:
        """Decide whether an item is relevant to the traded universe."""
        if item.symbols:
            # Tagged: relevant iff any tag is in the universe.
            return any(sym in universe for sym in item.symbols)
        # Untagged: check for any universe ticker mentioned as a whole word, else fall back
        # to the market-wide policy.
        text = item.text
        for sym in universe:
            if re.search(rf"(?<!\w){re.escape(sym)}(?!\w)", text):
                return True
        return self._treat_untagged_as_market_wide
