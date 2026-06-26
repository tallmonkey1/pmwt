"""Tests for quote data structures."""

from __future__ import annotations

import pytest

from options_engine.core.enums import OptionRight
from options_engine.core.errors import ValidationError
from options_engine.market.quotes import Quote, QuotedOption
from options_engine.pricing.instruments import EuropeanOption


class TestQuote:
    def test_valid(self) -> None:
        q = Quote(bid=1.0, ask=1.2, bid_size=10, ask_size=10)
        assert q.mid == pytest.approx(1.1)
        assert q.spread == pytest.approx(0.2)
        assert q.half_spread == pytest.approx(0.1)
        assert q.relative_spread == pytest.approx(0.2 / 1.1)

    def test_rejects_crossed_market(self) -> None:
        with pytest.raises(ValidationError):
            Quote(bid=1.2, ask=1.0, bid_size=10, ask_size=10)

    def test_rejects_locked_market(self) -> None:
        with pytest.raises(ValidationError):
            Quote(bid=1.0, ask=1.0, bid_size=10, ask_size=10)

    def test_rejects_negative_bid(self) -> None:
        with pytest.raises(ValidationError):
            Quote(bid=-0.1, ask=1.0, bid_size=10, ask_size=10)

    def test_rejects_nonpositive_size(self) -> None:
        with pytest.raises(ValidationError):
            Quote(bid=1.0, ask=1.2, bid_size=0, ask_size=10)

    def test_rejects_bool_size(self) -> None:
        with pytest.raises(ValidationError):
            Quote(bid=1.0, ask=1.2, bid_size=True, ask_size=10)  # type: ignore[arg-type]

    def test_zero_bid_allowed(self) -> None:
        q = Quote(bid=0.0, ask=0.05, bid_size=1, ask_size=1)
        assert q.bid == 0.0


class TestQuotedOption:
    def _option(self) -> EuropeanOption:
        return EuropeanOption(strike=100.0, expiry=0.1, right=OptionRight.CALL)

    def test_edge_to_mid(self) -> None:
        q = Quote(bid=1.0, ask=1.2, bid_size=10, ask_size=10)
        qo = QuotedOption(option=self._option(), quote=q, theoretical_value=1.0)
        assert qo.edge_to_mid == pytest.approx(0.1)

    def test_rejects_bad_types(self) -> None:
        q = Quote(bid=1.0, ask=1.2, bid_size=10, ask_size=10)
        with pytest.raises(ValidationError):
            QuotedOption(option="x", quote=q, theoretical_value=1.0)  # type: ignore[arg-type]

    def test_rejects_negative_theo(self) -> None:
        q = Quote(bid=1.0, ask=1.2, bid_size=10, ask_size=10)
        with pytest.raises(ValidationError):
            QuotedOption(option=self._option(), quote=q, theoretical_value=-1.0)
