r"""Order and fill data structures for the OMS (SPEC §7).

These immutable, self-validating records are the vocabulary of the order-management system.
The single most important field is :attr:`Order.client_order_id`: a caller-supplied unique id
that makes submission **idempotent** (re-submitting the same id is a no-op, not a duplicate
order). Accidental order multiplication -- from a retry, a reconnect, or a double call -- is
the most expensive execution bug, and idempotency is its structural prevention.

An :class:`Order` here represents a full iron-condor order (four legs traded as one ticket),
because the engine never trades naked legs. Its defined-risk economics are carried so the OMS
and risk supervisor can reason about it without re-pricing.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field

from ..core.enums import OrderSide, OrderStatus
from ..core.errors import ValidationError
from ..core.validation import check_non_negative, check_positive
from ..pricing.instruments import IronCondor

__all__ = ["Fill", "Order", "OrderState"]


@dataclass(frozen=True, slots=True)
class Order:
    """An iron-condor order ticket (four legs, traded as one).

    Parameters
    ----------
    client_order_id:
        Caller-supplied unique identifier. The OMS deduplicates on this, guaranteeing
        idempotent submission. Must be non-empty.
    condor:
        The iron-condor structure to trade.
    side:
        ``SELL`` to open the short (credit) condor, ``BUY`` to close it.
    quantity:
        Number of condors (positive).
    limit_credit:
        The limit price as a net credit per condor (for a SELL) or net debit (for a BUY).
        Must be non-negative; the broker may improve but not worsen beyond it.
    multiplier:
        Contract multiplier (e.g. 100).
    created_at:
        Timezone-aware creation time.
    """

    client_order_id: str
    condor: IronCondor
    side: OrderSide
    quantity: int
    limit_credit: float
    multiplier: float
    created_at: _dt.datetime
    symbol: str = "SPX"

    def __post_init__(self) -> None:
        if not self.client_order_id.strip():
            raise ValidationError("client_order_id must be non-empty", context={})
        if not self.symbol.strip():
            raise ValidationError("symbol must be non-empty", context={})
        if not isinstance(self.condor, IronCondor):
            raise ValidationError("condor must be an IronCondor", context={})
        if not isinstance(self.side, OrderSide):
            raise ValidationError("side must be an OrderSide", context={})
        if self.quantity < 1:
            raise ValidationError("quantity must be >= 1", context={"quantity": self.quantity})
        check_non_negative(self.limit_credit, name="limit_credit")
        check_positive(self.multiplier, name="multiplier")
        if (
            self.created_at.tzinfo is None
            or self.created_at.tzinfo.utcoffset(self.created_at) is None
        ):
            raise ValidationError("created_at must be timezone-aware", context={})

    @property
    def max_loss(self) -> float:
        """Defined-risk maximum loss for this order (per the condor width minus credit)."""
        per_condor = max(0.0, self.condor.max_spread_width - self.limit_credit)
        return per_condor * self.quantity * self.multiplier

    @property
    def notional_credit(self) -> float:
        """Total credit sought across all condors (per the limit), for accounting."""
        return self.limit_credit * self.quantity * self.multiplier


@dataclass(frozen=True, slots=True)
class Fill:
    """A (possibly partial) execution report for an order."""

    client_order_id: str
    filled_quantity: int
    fill_credit: float  # realized net credit per condor at the fill
    multiplier: float
    timestamp: _dt.datetime

    def __post_init__(self) -> None:
        if not self.client_order_id.strip():
            raise ValidationError("client_order_id must be non-empty", context={})
        if self.filled_quantity < 0:
            raise ValidationError("filled_quantity must be >= 0", context={})
        check_non_negative(self.fill_credit, name="fill_credit")
        check_positive(self.multiplier, name="multiplier")
        if (
            self.timestamp.tzinfo is None
            or self.timestamp.tzinfo.utcoffset(self.timestamp) is None
        ):
            raise ValidationError("timestamp must be timezone-aware", context={})


@dataclass(frozen=True, slots=True)
class OrderState:
    """The OMS's tracked lifecycle state for an order."""

    order: Order
    status: OrderStatus
    filled_quantity: int = 0
    fills: tuple[Fill, ...] = field(default_factory=tuple)
    reason: str = ""

    def __post_init__(self) -> None:
        if self.filled_quantity < 0 or self.filled_quantity > self.order.quantity:
            raise ValidationError(
                "filled_quantity must be in [0, order.quantity]",
                context={"filled": self.filled_quantity, "order_qty": self.order.quantity},
            )

    @property
    def remaining_quantity(self) -> int:
        """Quantity still to be filled."""
        return self.order.quantity - self.filled_quantity

    @property
    def is_terminal(self) -> bool:
        """True if the order is in a terminal status (no further activity expected)."""
        return self.status in {
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
        }
