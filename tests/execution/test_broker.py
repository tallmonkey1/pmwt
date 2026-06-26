"""Tests for the simulated broker."""

from __future__ import annotations

import datetime as dt

import pytest

from options_engine.core.enums import OrderSide
from options_engine.core.errors import ValidationError
from options_engine.core.random import RandomFactory
from options_engine.execution.broker import SimulatedBroker
from options_engine.execution.orders import Order
from options_engine.pricing.instruments import IronCondor

UTC = dt.UTC
NOW = dt.datetime(2024, 1, 1, tzinfo=UTC)


def _order(cid: str = "A1", qty: int = 5, side: OrderSide = OrderSide.SELL) -> Order:
    return Order(
        client_order_id=cid,
        condor=IronCondor(90.0, 95.0, 105.0, 110.0, 0.05, quantity=qty),
        side=side,
        quantity=qty,
        limit_credit=1.0,
        multiplier=100.0,
        created_at=NOW,
    )


class TestSimulatedBroker:
    def test_is_never_live(self) -> None:
        assert SimulatedBroker().is_live is False

    def test_full_fill(self) -> None:
        broker = SimulatedBroker(fill_probability=1.0)
        fill = broker.submit(_order(qty=5))
        assert fill.filled_quantity == 5

    def test_no_fill_on_zero_probability(self) -> None:
        broker = SimulatedBroker(fill_probability=0.0)
        fill = broker.submit(_order())
        assert fill.filled_quantity == 0

    def test_partial_fill(self) -> None:
        broker = SimulatedBroker(
            rng_factory=RandomFactory(1), fill_probability=1.0, partial_fill_probability=1.0
        )
        fill = broker.submit(_order(qty=10))
        assert 1 <= fill.filled_quantity < 10

    def test_credit_slippage(self) -> None:
        broker = SimulatedBroker(fill_probability=1.0, credit_slippage=0.2)
        fill = broker.submit(_order())
        assert fill.fill_credit == pytest.approx(0.8)  # 1.0 * (1 - 0.2)

    def test_positions_tracked(self) -> None:
        broker = SimulatedBroker(fill_probability=1.0)
        broker.submit(_order(cid="A1", qty=3, side=OrderSide.SELL))
        positions = broker.positions()
        assert any(p.client_order_id == "A1" and p.quantity == 3 for p in positions)

    def test_reproducible(self) -> None:
        a = SimulatedBroker(rng_factory=RandomFactory(7), fill_probability=0.5)
        b = SimulatedBroker(rng_factory=RandomFactory(7), fill_probability=0.5)
        fa = [a.submit(_order(cid=f"O{i}")).filled_quantity for i in range(20)]
        fb = [b.submit(_order(cid=f"O{i}")).filled_quantity for i in range(20)]
        assert fa == fb

    def test_rejects_non_order(self) -> None:
        with pytest.raises(ValidationError):
            SimulatedBroker().submit("not an order")  # type: ignore[arg-type]
