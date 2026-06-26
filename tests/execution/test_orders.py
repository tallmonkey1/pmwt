"""Tests for order/fill/state data structures."""

from __future__ import annotations

import datetime as dt

import pytest

from options_engine.core.enums import OrderSide, OrderStatus
from options_engine.core.errors import ValidationError
from options_engine.execution.orders import Fill, Order, OrderState
from options_engine.pricing.instruments import IronCondor

UTC = dt.UTC
NOW = dt.datetime(2024, 1, 1, tzinfo=UTC)


def _condor() -> IronCondor:
    return IronCondor(90.0, 95.0, 105.0, 110.0, 0.05, quantity=1)


def _order(**kw) -> Order:
    defaults = {
        "client_order_id": "A1",
        "condor": _condor(),
        "side": OrderSide.SELL,
        "quantity": 2,
        "limit_credit": 1.5,
        "multiplier": 100.0,
        "created_at": NOW,
    }
    defaults.update(kw)
    return Order(**defaults)  # type: ignore[arg-type]


class TestOrder:
    def test_valid(self) -> None:
        order = _order()
        assert order.max_loss == pytest.approx((5.0 - 1.5) * 2 * 100.0)
        assert order.notional_credit == pytest.approx(1.5 * 2 * 100.0)

    def test_rejects_empty_id(self) -> None:
        with pytest.raises(ValidationError):
            _order(client_order_id="  ")

    def test_rejects_zero_quantity(self) -> None:
        with pytest.raises(ValidationError):
            _order(quantity=0)

    def test_rejects_negative_credit(self) -> None:
        with pytest.raises(ValidationError):
            _order(limit_credit=-1.0)

    def test_rejects_naive_time(self) -> None:
        with pytest.raises(ValidationError):
            _order(created_at=dt.datetime(2024, 1, 1))


class TestFill:
    def test_valid(self) -> None:
        fill = Fill(
            client_order_id="A1",
            filled_quantity=2,
            fill_credit=1.4,
            multiplier=100.0,
            timestamp=NOW,
        )
        assert fill.filled_quantity == 2

    def test_zero_quantity_allowed(self) -> None:
        # A zero-quantity fill represents "no fill" and is valid.
        fill = Fill(
            client_order_id="A1",
            filled_quantity=0,
            fill_credit=0.0,
            multiplier=100.0,
            timestamp=NOW,
        )
        assert fill.filled_quantity == 0

    def test_rejects_negative_quantity(self) -> None:
        with pytest.raises(ValidationError):
            Fill(
                client_order_id="A1",
                filled_quantity=-1,
                fill_credit=0.0,
                multiplier=100.0,
                timestamp=NOW,
            )


class TestOrderState:
    def test_remaining_and_terminal(self) -> None:
        order = _order(quantity=3)
        state = OrderState(order=order, status=OrderStatus.PARTIALLY_FILLED, filled_quantity=1)
        assert state.remaining_quantity == 2
        assert not state.is_terminal

    def test_filled_is_terminal(self) -> None:
        order = _order(quantity=2)
        state = OrderState(order=order, status=OrderStatus.FILLED, filled_quantity=2)
        assert state.is_terminal

    def test_rejects_overfill(self) -> None:
        order = _order(quantity=2)
        with pytest.raises(ValidationError):
            OrderState(order=order, status=OrderStatus.FILLED, filled_quantity=3)
