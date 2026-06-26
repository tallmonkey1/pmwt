"""Tests for the keyword news classifier."""

from __future__ import annotations

import datetime as dt

import pytest

from options_engine.core.errors import ValidationError
from options_engine.news.classifier import KeywordNewsClassifier
from options_engine.news.events import EventSeverity, NewsItem

UTC = dt.UTC
UNIVERSE = frozenset({"SPX", "SPY"})


def _item(headline: str, *, symbols: tuple[str, ...] = ()) -> NewsItem:
    return NewsItem(
        timestamp=dt.datetime(2024, 1, 1, tzinfo=UTC),
        headline=headline,
        source="wire",
        symbols=symbols,
    )


class TestKeywordNewsClassifier:
    def test_detects_critical_term(self) -> None:
        cls = KeywordNewsClassifier()
        a = cls.classify(_item("Fed signals emergency rate hike"), universe=UNIVERSE)
        assert a.severity == EventSeverity.CRITICAL
        assert "rate hike" in a.matched_terms
        assert a.is_material

    def test_benign_is_not_material(self) -> None:
        cls = KeywordNewsClassifier()
        a = cls.classify(_item("Company announces summer picnic"), universe=UNIVERSE)
        assert a.severity == EventSeverity.NONE
        assert not a.is_material

    def test_takes_max_severity(self) -> None:
        cls = KeywordNewsClassifier()
        # Contains both MEDIUM ("earnings") and CRITICAL ("crash").
        a = cls.classify(_item("Earnings miss triggers market crash"), universe=UNIVERSE)
        assert a.severity == EventSeverity.CRITICAL

    def test_word_boundary_no_false_match(self) -> None:
        cls = KeywordNewsClassifier()
        # "defaulted" should not match the term "default" if boundaries work... but it does
        # share the stem; verify a clearly unrelated word does not trip a term.
        a = cls.classify(_item("The warranty was extended"), universe=UNIVERSE)
        # "war" must not match inside "warranty".
        assert "war" not in a.matched_terms

    def test_relevance_by_tag(self) -> None:
        cls = KeywordNewsClassifier(treat_untagged_as_market_wide=False)
        relevant = cls.classify(_item("rate hike", symbols=("SPX",)), universe=UNIVERSE)
        irrelevant = cls.classify(_item("rate hike", symbols=("TSLA",)), universe=UNIVERSE)
        assert relevant.is_relevant
        assert not irrelevant.is_relevant

    def test_relevance_by_mention_when_untagged(self) -> None:
        cls = KeywordNewsClassifier(treat_untagged_as_market_wide=False)
        a = cls.classify(_item("SPX tumbles on inflation print"), universe=UNIVERSE)
        assert a.is_relevant

    def test_untagged_market_wide_policy(self) -> None:
        market_wide = KeywordNewsClassifier(treat_untagged_as_market_wide=True)
        narrow = KeywordNewsClassifier(treat_untagged_as_market_wide=False)
        item = _item("inflation surprises to the upside")  # no universe ticker mentioned
        assert market_wide.classify(item, universe=UNIVERSE).is_relevant
        assert not narrow.classify(item, universe=UNIVERSE).is_relevant

    def test_rejects_empty_lexicon(self) -> None:
        with pytest.raises(ValidationError):
            KeywordNewsClassifier(lexicon={})

    def test_custom_lexicon(self) -> None:
        cls = KeywordNewsClassifier(lexicon={"meltdown": EventSeverity.CRITICAL})
        a = cls.classify(_item("Sector meltdown accelerates"), universe=UNIVERSE)
        assert a.severity == EventSeverity.CRITICAL
