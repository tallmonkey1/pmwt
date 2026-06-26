r"""Broker interface and the simulated broker (SPEC §7, §8).

The OMS depends only on the abstract :class:`Broker` interface, so the *same* execution code
runs against a simulated broker (history/paper, and all tests) and a real broker (live),
guaranteeing backtest <-> live parity. The simulated broker is the **default everywhere**: it
is impossible to accidentally trade real money through it.

A broker exposes three operations the OMS needs:

* :meth:`submit` -- place an order, returning an immediate :class:`Fill` (the simulated
  broker fills synchronously; a real async broker would report fills via callbacks, adapted
  to this contract).
* :meth:`cancel` -- cancel a working order by client id.
* :meth:`positions` -- report the broker's view of open positions, for reconciliation.

The simulated broker models partial fills and a configurable fill probability so tests can
exercise the OMS's partial-fill and rejection paths deterministically.
"""

from __future__ import annotations

import datetime as _dt
from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..core.errors import ValidationError
from ..core.random import RandomFactory
from ..core.validation import check_unit_interval
from .orders import Fill, Order

__all__ = ["Broker", "BrokerPosition", "SimulatedBroker"]


@dataclass(frozen=True, slots=True)
class BrokerPosition:
    """The broker's reported view of an open position (for reconciliation)."""

    client_order_id: str
    quantity: int


class Broker(ABC):
    """Abstract broker the OMS trades through."""

    @abstractmethod
    def submit(self, order: Order) -> Fill:
        """Submit an order and return a fill report (possibly partial or zero-quantity)."""
        raise NotImplementedError

    @abstractmethod
    def cancel(self, client_order_id: str) -> bool:
        """Cancel a working order; return True if a cancel was actioned."""
        raise NotImplementedError

    @abstractmethod
    def positions(self) -> tuple[BrokerPosition, ...]:
        """Return the broker's current open positions."""
        raise NotImplementedError

    @property
    @abstractmethod
    def is_live(self) -> bool:
        """True only for a broker that can move real capital."""
        raise NotImplementedError


class SimulatedBroker(Broker):
    """A deterministic, in-memory broker for backtest, paper, and tests.

    Parameters
    ----------
    rng_factory:
        Reproducible randomness for fill outcomes.
    fill_probability:
        Probability that a submitted order fills at all (models liquidity gaps). The rest are
        reported as zero-quantity fills (no fill).
    partial_fill_probability:
        Conditional on filling, the probability the fill is partial (a random fraction of the
        requested quantity), exercising the OMS partial-fill path.
    credit_slippage:
        Fraction by which the realized credit is worse than the limit (models paying the
        spread); ``0`` fills exactly at the limit.
    """

    def __init__(
        self,
        *,
        rng_factory: RandomFactory | None = None,
        fill_probability: float = 1.0,
        partial_fill_probability: float = 0.0,
        credit_slippage: float = 0.0,
    ) -> None:
        check_unit_interval(fill_probability, name="fill_probability")
        check_unit_interval(partial_fill_probability, name="partial_fill_probability")
        check_unit_interval(credit_slippage, name="credit_slippage")
        self._rng = (rng_factory or RandomFactory(0)).generator("broker.sim")
        self._fill_probability = fill_probability
        self._partial_fill_probability = partial_fill_probability
        self._credit_slippage = credit_slippage
        self._positions: dict[str, int] = {}

    @property
    def is_live(self) -> bool:
        """Always False: the simulated broker can never move real capital."""
        return False

    def submit(self, order: Order) -> Fill:
        """Simulate an order submission and return a fill report."""
        if not isinstance(order, Order):
            raise ValidationError("order must be an Order", context={})

        if self._rng.random() > self._fill_probability:
            # No fill (liquidity gap).
            return Fill(
                client_order_id=order.client_order_id,
                filled_quantity=0,
                fill_credit=0.0,
                multiplier=order.multiplier,
                timestamp=order.created_at,
            )

        filled_qty = order.quantity
        if self._rng.random() < self._partial_fill_probability and order.quantity > 1:
            filled_qty = int(self._rng.integers(1, order.quantity))

        fill_credit = order.limit_credit * (1.0 - self._credit_slippage)
        # Track net position (SELL opens positive condor inventory, BUY reduces it).
        delta = filled_qty if order.side.value == "SELL" else -filled_qty
        self._positions[order.client_order_id] = (
            self._positions.get(order.client_order_id, 0) + delta
        )
        return Fill(
            client_order_id=order.client_order_id,
            filled_quantity=filled_qty,
            fill_credit=fill_credit,
            multiplier=order.multiplier,
            timestamp=order.created_at,
        )

    def cancel(self, client_order_id: str) -> bool:
        """A synchronous simulated broker has no resting orders to cancel."""
        if not client_order_id.strip():
            raise ValidationError("client_order_id must be non-empty", context={})
        return False

    def positions(self) -> tuple[BrokerPosition, ...]:
        """Return the simulated open positions."""
        return tuple(
            BrokerPosition(client_order_id=cid, quantity=qty)
            for cid, qty in self._positions.items()
            if qty != 0
        )

    def now(self) -> _dt.datetime:
        """Return a timezone-aware timestamp (utility for callers needing one)."""
        return _dt.datetime.now(_dt.UTC)
