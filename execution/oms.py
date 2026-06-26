r"""Order-management system: idempotent, risk-gated order lifecycle (SPEC §7, §4.5).

The OMS is the single choke point through which every order flows, in every mode. It enforces
the institutional execution controls:

* **Idempotency** -- an order's ``client_order_id`` is recorded before submission; a duplicate
  id is rejected, so retries/reconnects/double-calls can never create duplicate positions.
* **Risk gate on every order** -- before *opening* risk, the deterministic risk supervisor
  must approve; a veto (including the drawdown kill-switch) blocks the order. Closing orders
  (flattening) are always allowed, since reducing risk is never unsafe.
* **State machine** -- each order transitions through validated lifecycle states
  (PENDING_NEW -> WORKING -> PARTIALLY_FILLED/FILLED/REJECTED), recorded in an auditable
  ledger.
* **Reconciliation** -- the local ledger can be checked against the broker's reported
  positions; any divergence is reported and should halt new trading.
* **Flatten-all** -- a kill-switch entry point that closes all open positions.

The OMS is broker-agnostic (it holds a :class:`Broker`); the same logic runs against the
simulated and live brokers, which is what makes live behaviour match the backtest.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..core.enums import OrderSide, OrderStatus
from ..core.errors import ExecutionError, RiskLimitError, ValidationError
from ..core.logging import bind_context, get_logger
from ..strategy.account import Account, OpenPosition
from ..strategy.risk_supervisor import RiskSupervisor
from .broker import Broker
from .orders import Fill, Order, OrderState

__all__ = ["OrderManagementSystem", "ReconciliationResult"]

_logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ReconciliationResult:
    """The outcome of reconciling the local ledger against the broker."""

    consistent: bool
    local_only: tuple[str, ...]  # client ids the OMS has but the broker does not
    broker_only: tuple[str, ...]  # client ids the broker has but the OMS does not
    quantity_mismatches: tuple[str, ...]  # client ids whose quantities differ

    @property
    def divergences(self) -> int:
        """Total number of divergent order ids."""
        return len(self.local_only) + len(self.broker_only) + len(self.quantity_mismatches)


class OrderManagementSystem:
    """Idempotent, risk-gated order lifecycle manager over a broker.

    Parameters
    ----------
    broker:
        The broker to trade through (simulated or live).
    risk_supervisor:
        The deterministic risk overlay consulted before every opening order.
    """

    def __init__(self, *, broker: Broker, risk_supervisor: RiskSupervisor) -> None:
        if not isinstance(broker, Broker):
            raise ValidationError("broker must be a Broker", context={})
        if not isinstance(risk_supervisor, RiskSupervisor):
            raise ValidationError("risk_supervisor must be a RiskSupervisor", context={})
        self._broker = broker
        self._supervisor = risk_supervisor
        self._orders: dict[str, OrderState] = {}

    @property
    def broker_is_live(self) -> bool:
        """True if the underlying broker can move real capital."""
        return self._broker.is_live

    def order_state(self, client_order_id: str) -> OrderState:
        """Return the tracked state of an order, or raise if unknown."""
        if client_order_id not in self._orders:
            raise ExecutionError(
                "unknown client_order_id", context={"client_order_id": client_order_id}
            )
        return self._orders[client_order_id]

    def open_orders(self) -> tuple[OrderState, ...]:
        """Return all non-terminal tracked orders."""
        return tuple(s for s in self._orders.values() if not s.is_terminal)

    # -- submission ------------------------------------------------------------------

    def submit_open(
        self,
        order: Order,
        *,
        account: Account,
        candidate_position: OpenPosition,
        unrealized_pnl: float = 0.0,
        risked_today: float = 0.0,
        available_margin: float,
    ) -> OrderState:
        """Submit an *opening* (risk-increasing) order, gated by the risk supervisor.

        Idempotent on ``client_order_id``: a re-submission returns the existing state without
        contacting the broker. A risk-supervisor veto raises :class:`RiskLimitError` and no
        order is sent.
        """
        with bind_context(client_order_id=order.client_order_id):
            existing = self._idempotency_check(order)
            if existing is not None:
                return existing

            if order.side is not OrderSide.SELL:
                raise ValidationError(
                    "submit_open expects a SELL (credit) order", context={"side": order.side.value}
                )

            approval = self._supervisor.approve_new_position(
                account,
                candidate_position,
                unrealized_pnl=unrealized_pnl,
                risked_today=risked_today,
                available_margin=available_margin,
            )
            if not approval.approved:
                # Record the rejection in the ledger for the audit trail, then refuse.
                self._orders[order.client_order_id] = OrderState(
                    order=order, status=OrderStatus.REJECTED, reason=approval.reason
                )
                _logger.warning(
                    "order_rejected_by_risk_supervisor",
                    extra={
                        "reason": approval.reason,
                        "kill_switch": approval.kill_switch_triggered,
                    },
                )
                raise RiskLimitError(
                    "order rejected by risk supervisor", context={"reason": approval.reason}
                )

            return self._submit_to_broker(order)

    def submit_close(self, order: Order) -> OrderState:
        """Submit a *closing* (risk-reducing) order; not gated (reducing risk is always safe).

        Still idempotent on ``client_order_id``.
        """
        with bind_context(client_order_id=order.client_order_id):
            existing = self._idempotency_check(order)
            if existing is not None:
                return existing
            if order.side is not OrderSide.BUY:
                raise ValidationError(
                    "submit_close expects a BUY (debit) order to close the short condor",
                    context={"side": order.side.value},
                )
            return self._submit_to_broker(order)

    def _idempotency_check(self, order: Order) -> OrderState | None:
        """Return the existing state if this id was already submitted, else None."""
        existing = self._orders.get(order.client_order_id)
        if existing is not None:
            _logger.info(
                "duplicate_order_suppressed",
                extra={"client_order_id": order.client_order_id, "status": existing.status.value},
            )
            return existing
        return None

    def _submit_to_broker(self, order: Order) -> OrderState:
        """Record PENDING_NEW, submit to the broker, and fold the fill into the state."""
        # Record intent *before* submission so a crash mid-submit leaves an auditable trace
        # and a retry is recognised as a duplicate.
        self._orders[order.client_order_id] = OrderState(
            order=order, status=OrderStatus.PENDING_NEW
        )
        try:
            fill = self._broker.submit(order)
        except Exception as exc:
            self._orders[order.client_order_id] = OrderState(
                order=order, status=OrderStatus.REJECTED, reason=f"broker error: {exc}"
            )
            raise ExecutionError("broker submission failed", context={"error": str(exc)}) from exc

        state = self._apply_fill(order, fill)
        self._orders[order.client_order_id] = state
        _logger.info(
            "order_submitted",
            extra={
                "client_order_id": order.client_order_id,
                "status": state.status.value,
                "filled_quantity": state.filled_quantity,
                "live": self._broker.is_live,
            },
        )
        return state

    @staticmethod
    def _apply_fill(order: Order, fill: Fill) -> OrderState:
        """Fold a fill report into a validated terminal/partial order state."""
        if fill.client_order_id != order.client_order_id:
            raise ExecutionError(
                "fill client_order_id mismatch",
                context={"order": order.client_order_id, "fill": fill.client_order_id},
            )
        if fill.filled_quantity <= 0:
            return OrderState(
                order=order, status=OrderStatus.REJECTED, reason="no fill (no liquidity)"
            )
        if fill.filled_quantity >= order.quantity:
            status = OrderStatus.FILLED
        else:
            status = OrderStatus.PARTIALLY_FILLED
        return OrderState(
            order=order,
            status=status,
            filled_quantity=min(fill.filled_quantity, order.quantity),
            fills=(fill,),
        )

    # -- reconciliation & flatten ----------------------------------------------------

    def reconcile(self) -> ReconciliationResult:
        """Compare the local filled ledger against the broker's reported positions.

        A divergence indicates a desync (a fill we missed, or a position the broker doesn't
        recognise) and should halt new trading until investigated.
        """
        broker_positions = {p.client_order_id: p.quantity for p in self._broker.positions()}
        local_positions = {
            cid: s.filled_quantity
            for cid, s in self._orders.items()
            if s.filled_quantity > 0 and s.order.side is OrderSide.SELL
        }

        local_only = tuple(sorted(set(local_positions) - set(broker_positions)))
        broker_only = tuple(sorted(set(broker_positions) - set(local_positions)))
        mismatches = tuple(
            sorted(
                cid
                for cid in set(local_positions) & set(broker_positions)
                if local_positions[cid] != broker_positions[cid]
            )
        )
        result = ReconciliationResult(
            consistent=not (local_only or broker_only or mismatches),
            local_only=local_only,
            broker_only=broker_only,
            quantity_mismatches=mismatches,
        )
        if not result.consistent:
            _logger.error(
                "reconciliation_divergence",
                extra={
                    "local_only": list(local_only),
                    "broker_only": list(broker_only),
                    "quantity_mismatches": list(mismatches),
                },
            )
        return result

    def flatten_all(self, *, closing_orders: dict[str, Order]) -> tuple[OrderState, ...]:
        """Close all open (filled, short) positions using caller-supplied closing orders.

        The kill-switch entry point. Closing orders are *not* risk-gated (reducing risk is
        always permitted). ``closing_orders`` maps each open position's client id to a BUY
        order that closes it; any open position without a supplied closing order is reported as
        an error so nothing is silently left open.
        """
        results: list[OrderState] = []
        open_short_ids = [
            cid
            for cid, s in self._orders.items()
            if s.filled_quantity > 0
            and s.order.side is OrderSide.SELL
            and not _is_closed(self._orders, cid)
        ]
        missing = [cid for cid in open_short_ids if cid not in closing_orders]
        if missing:
            raise ExecutionError(
                "flatten_all is missing closing orders for open positions",
                context={"missing": missing},
            )
        for cid in open_short_ids:
            results.append(self.submit_close(closing_orders[cid]))
        _logger.warning("flatten_all_executed", extra={"closed": len(results)})
        return tuple(results)


def _is_closed(orders: dict[str, OrderState], open_client_id: str) -> bool:
    """Return True if a closing BUY order referencing the open id has already filled.

    Closing orders use the convention ``"<open_client_id>:close"`` for their own client id, so
    a flattened position is not flattened twice (idempotency across the open/close pair).
    """
    close_id = f"{open_client_id}:close"
    closing = orders.get(close_id)
    return closing is not None and closing.filled_quantity > 0
