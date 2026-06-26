"""Tests for the alpha-driven Avellaneda-Stoikov quote noise."""

from __future__ import annotations

import numpy as np
import pytest

from options_engine.core.errors import ValidationError
from options_engine.core.market_alpha import MarketAlpha
from options_engine.market.alpha_noise import alpha_noise_intensity, apply_alpha_noise
from options_engine.market.quotes import Quote


def _quote(*, bid: float = 1.0, ask: float = 1.2, size: int = 10) -> Quote:
    return Quote(bid=bid, ask=ask, bid_size=size, ask_size=size)


class TestAlphaNoiseIntensity:
    def test_intensity_zero_at_full_suppression(self) -> None:
        alpha = MarketAlpha.from_components(stoikov_noise_suppression=1.0)
        assert alpha_noise_intensity(alpha) == 0.0

    def test_intensity_grows_as_suppression_falls(self) -> None:
        hi = alpha_noise_intensity(
            MarketAlpha.from_components(stoikov_noise_suppression=0.9)
        )
        lo = alpha_noise_intensity(
            MarketAlpha.from_components(stoikov_noise_suppression=0.1)
        )
        assert lo > hi


class TestApplyAlphaNoise:
    def test_ones_alpha_returns_unchanged_quote(self) -> None:
        q = _quote()
        alpha = MarketAlpha.ones()
        rng = np.random.default_rng(0)
        out = apply_alpha_noise(q, alpha=alpha, rng=rng)
        assert out.bid == q.bid
        assert out.ask == q.ask
        assert out.bid_size == q.bid_size
        assert out.ask_size == q.ask_size

    def test_zeros_alpha_widens_spread(self) -> None:
        q = _quote(bid=1.0, ask=1.2)
        alpha = MarketAlpha.zeros()
        rng = np.random.default_rng(0)
        out = apply_alpha_noise(q, alpha=alpha, rng=rng)
        # Bid should have moved DOWN, ask should have moved UP, so the spread widened.
        assert out.bid <= q.bid
        assert out.ask >= q.ask
        # The result must still be a valid non-crossed quote.
        assert out.ask > out.bid

    def test_reproducible_with_same_seed(self) -> None:
        q = _quote()
        alpha = MarketAlpha.from_components(stoikov_noise_suppression=0.5)
        a = apply_alpha_noise(q, alpha=alpha, rng=np.random.default_rng(42))
        b = apply_alpha_noise(q, alpha=alpha, rng=np.random.default_rng(42))
        assert a.bid == pytest.approx(b.bid)
        assert a.ask == pytest.approx(b.ask)

    def test_rejects_non_quote(self) -> None:
        alpha = MarketAlpha.zeros()
        with pytest.raises(ValidationError):
            apply_alpha_noise("not a quote", alpha=alpha, rng=np.random.default_rng(0))

    def test_rejects_non_alpha(self) -> None:
        q = _quote()
        with pytest.raises(ValidationError):
            apply_alpha_noise(q, alpha="not an alpha", rng=np.random.default_rng(0))

    def test_bounded_no_negative_price(self) -> None:
        # Even at maximum noise, the perturbed bid cannot go below zero.
        q = _quote(bid=0.01, ask=0.05)
        alpha = MarketAlpha.zeros()
        rng = np.random.default_rng(7)
        out = apply_alpha_noise(q, alpha=alpha, rng=rng)
        assert out.bid >= 0.0
        assert out.ask > out.bid
