"""Tests for the order-management system: idempotency, risk gating, reconciliation, flatten."""

from __future__ import annotations

import datetime as dt

import pytest

from options_engine.core.config import RiskConfig
from options_engine.core.enums import OrderSide, OrderStatus
from options_engine.core.errors import ExecutionError, RiskLimitError, ValidationError
from options_engine.execution.broker import Broker, BrokerPosition, SimulatedBroker
from options_engine.execution.oms import OrderManagementSystem
from options_engine.execution.orders import Fill, Order
from options_engine.pricing.instruments import IronCondor
from options_engine.strategy.account import Account, OpenPosition
from options_engine.strategy.risk_supervisor import RiskSupervisor

UTC = dt.UTC
NOW = dt.datetime(2024, 1, 1, tzinfo=UTC)


def _condor() -> IronCondor:
    return IronCondor(90.0, 95.0, 105.0, 110.0, 0.05, quantity=1)


def _sell(cid: str = "A1", qty: int = 1) -> Order:
    return Order(
        client_order_id=cid,
        condor=_condor(),
        side=OrderSide.SELL,
        quantity=qty,
        limit_credit=1.0,
        multiplier=100.0,
        created_at=NOW,
    )


def _buy(cid: str, qty: int = 1) -> Order:
    return Order(
        client_order_id=cid,
        condor=_condor(),
        side=OrderSide.BUY,
        quantity=qty,
        limit_credit=0.5,
        multiplier=100.0,
        created_at=NOW,
    )


def _position(cid: str = "A1", qty: int = 1) -> OpenPosition:
    return OpenPosition(
        position_id=cid,
        condor=_condor(),
        entry_credit=1.0,
        quantity=qty,
        multiplier=100.0,
        entry_time=NOW,
        entry_spot=100.0,
    )


def _oms(broker: Broker | None = None, risk: RiskConfig | None = None) -> OrderManagementSystem:
    return OrderManagementSystem(
        broker=broker or SimulatedBroker(fill_probability=1.0),
        risk_supervisor=RiskSupervisor(risk=risk or RiskConfig()),
    )


class TestSubmitOpen:
    def _submit(self, oms: OrderManagementSystem, order: Order, **kw):
        defaults = {
            "account": Account.open(starting_cash=100_000.0),
            "candidate_position": _position(order.client_order_id, order.quantity),
            "available_margin": 50_000.0,
        }
        defaults.update(kw)
        return oms.submit_open(order, **defaults)

    def test_fills_and_records(self) -> None:
        oms = _oms()
        state = self._submit(oms, _sell())
        assert state.status is OrderStatus.FILLED
        assert state.filled_quantity == 1

    def test_idempotent_no_duplicate(self) -> None:
        oms = _oms()
        s1 = self._submit(oms, _sell("DUP"))
        s2 = self._submit(oms, _sell("DUP"))
        # Same id => same state, no second broker submission.
        assert s1.filled_quantity == s2.filled_quantity
        assert len(oms._orders) == 1

    def test_risk_veto_blocks_order(self) -> None:
        # Kill-switch tripped by a large drawdown => order rejected, RiskLimitError raised.
        oms = _oms(risk=RiskConfig(max_drawdown_kill_switch=0.05))
        with pytest.raises(RiskLimitError):
            self._submit(oms, _sell("V1"), unrealized_pnl=-20_000.0)
        assert oms.order_state("V1").status is OrderStatus.REJECTED

    def test_per_trade_cap_veto(self) -> None:
        oms = _oms(
            risk=RiskConfig(max_risk_fraction_per_trade=0.001, max_risk_fraction_per_day=0.5)
        )
        big = _sell("BIG", qty=10)
        with pytest.raises(RiskLimitError):
            self._submit(oms, big, candidate_position=_position("BIG", 10))

    def test_rejects_buy_as_open(self) -> None:
        oms = _oms()
        with pytest.raises(ValidationError):
            self._submit(oms, _buy("B1"))

    def test_no_fill_marks_rejected(self) -> None:
        oms = _oms(broker=SimulatedBroker(fill_probability=0.0))
        state = self._submit(oms, _sell("NF"))
        assert state.status is OrderStatus.REJECTED


class TestSubmitClose:
    def test_close_not_risk_gated(self) -> None:
        # Closing is allowed even when the kill-switch would block opening.
        oms = _oms(risk=RiskConfig(max_drawdown_kill_switch=0.05))
        state = oms.submit_close(_buy("C1"))
        assert state.status in {
            OrderStatus.FILLED,
            OrderStatus.PARTIALLY_FILLED,
            OrderStatus.REJECTED,
        }

    def test_rejects_sell_as_close(self) -> None:
        oms = _oms()
        with pytest.raises(ValidationError):
            oms.submit_close(_sell("S1"))

    def test_close_idempotent(self) -> None:
        oms = _oms()
        a = oms.submit_close(_buy("C2"))
        b = oms.submit_close(_buy("C2"))
        assert a.filled_quantity == b.filled_quantity


class _StubBroker(Broker):
    """A broker returning configurable positions for reconciliation tests."""

    def __init__(self, positions: tuple[BrokerPosition, ...]) -> None:
        self._positions = positions

    def submit(self, order: Order) -> Fill:
        return Fill(
            client_order_id=order.client_order_id,
            filled_quantity=order.quantity,
            fill_credit=order.limit_credit,
            multiplier=order.multiplier,
            timestamp=order.created_at,
        )

    def cancel(self, client_order_id: str) -> bool:
        return False

    def positions(self) -> tuple[BrokerPosition, ...]:
        return self._positions

    @property
    def is_live(self) -> bool:
        return False


class TestReconciliation:
    def _open(self, oms: OrderManagementSystem, cid: str, qty: int) -> None:
        oms.submit_open(
            _sell(cid, qty),
            account=Account.open(starting_cash=100_000.0),
            candidate_position=_position(cid, qty),
            available_margin=50_000.0,
        )

    def test_consistent(self) -> None:
        oms = OrderManagementSystem(
            broker=_StubBroker((BrokerPosition(client_order_id="A1", quantity=1),)),
            risk_supervisor=RiskSupervisor(risk=RiskConfig()),
        )
        self._open(oms, "A1", 1)
        result = oms.reconcile()
        assert result.consistent
        assert result.divergences == 0

    def test_broker_only_divergence(self) -> None:
        oms = OrderManagementSystem(
            broker=_StubBroker((BrokerPosition(client_order_id="GHOST", quantity=1),)),
            risk_supervisor=RiskSupervisor(risk=RiskConfig()),
        )
        result = oms.reconcile()
        assert not result.consistent
        assert "GHOST" in result.broker_only

    def test_quantity_mismatch(self) -> None:
        oms = OrderManagementSystem(
            broker=_StubBroker((BrokerPosition(client_order_id="A1", quantity=5),)),
            risk_supervisor=RiskSupervisor(risk=RiskConfig()),
        )
        self._open(oms, "A1", 1)  # local says 1, broker says 5
        result = oms.reconcile()
        assert not result.consistent
        assert "A1" in result.quantity_mismatches


class TestFlattenAll:
    def test_closes_open_positions(self) -> None:
        oms = _oms()
        oms.submit_open(
            _sell("A1", 1),
            account=Account.open(starting_cash=100_000.0),
            candidate_position=_position("A1", 1),
            available_margin=50_000.0,
        )
        closing = {"A1": _buy("A1:close", 1)}
        results = oms.flatten_all(closing_orders=closing)
        assert len(results) == 1

    def test_missing_closing_order_raises(self) -> None:
        oms = _oms()
        oms.submit_open(
            _sell("A1", 1),
            account=Account.open(starting_cash=100_000.0),
            candidate_position=_position("A1", 1),
            available_margin=50_000.0,
        )
        with pytest.raises(ExecutionError):
            oms.flatten_all(closing_orders={})
