"""Tests for the fill / slippage simulator."""

from __future__ import annotations

import pytest

from options_engine.core.enums import OrderSide
from options_engine.core.errors import ValidationError
from options_engine.market.execution_sim import (
    FillModelConfig,
    simulate_fill,
)
from options_engine.market.quotes import Quote


def _quote() -> Quote:
    return Quote(bid=1.0, ask=1.2, bid_size=10, ask_size=10)


class TestSimulateFill:
    def test_buy_fills_at_ask_within_touch(self) -> None:
        fill = simulate_fill(_quote(), side=OrderSide.BUY, quantity=5)
        assert fill.filled_quantity == 5
        assert fill.average_price == pytest.approx(1.2)
        assert fill.is_complete

    def test_sell_fills_at_bid_within_touch(self) -> None:
        fill = simulate_fill(_quote(), side=OrderSide.SELL, quantity=5)
        assert fill.average_price == pytest.approx(1.0)

    def test_buy_slippage_is_half_spread_at_touch(self) -> None:
        fill = simulate_fill(_quote(), side=OrderSide.BUY, quantity=1)
        # Buying at the ask vs mid = half the spread.
        assert fill.slippage_per_contract == pytest.approx(0.1)

    def test_large_order_walks_the_book(self) -> None:
        # Order exceeding the touch size pays progressively worse prices.
        fill = simulate_fill(_quote(), side=OrderSide.BUY, quantity=20)
        assert fill.filled_quantity == 20
        assert fill.average_price > 1.2  # worse than the touch

    def test_partial_fill_beyond_depth(self) -> None:
        cfg = FillModelConfig(max_fill_multiple=2.0)
        fill = simulate_fill(_quote(), side=OrderSide.BUY, quantity=100, config=cfg)
        # Depth capped at 2 * 10 = 20 contracts.
        assert fill.filled_quantity == 20
        assert not fill.is_complete

    def test_slippage_cost_nonnegative(self) -> None:
        for side in (OrderSide.BUY, OrderSide.SELL):
            fill = simulate_fill(_quote(), side=side, quantity=30)
            assert fill.slippage_per_contract >= 0.0
            assert fill.total_slippage_cost >= 0.0

    def test_zero_impact_config(self) -> None:
        cfg = FillModelConfig(impact_per_level=0.0)
        fill = simulate_fill(_quote(), side=OrderSide.BUY, quantity=30, config=cfg)
        # No impact => everything at the touch.
        assert fill.average_price == pytest.approx(1.2)

    def test_rejects_bad_quantity(self) -> None:
        with pytest.raises(ValidationError):
            simulate_fill(_quote(), side=OrderSide.BUY, quantity=0)

    def test_rejects_bad_side(self) -> None:
        with pytest.raises(ValidationError):
            simulate_fill(_quote(), side="BUY", quantity=1)  # type: ignore[arg-type]
