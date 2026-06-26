r"""Interactive Brokers live broker adapter -- fail-closed skeleton (SPEC §7, §13).

This is the only component in the system that can move real capital, so it is built to refuse
by default. It can be constructed *only* through :func:`create_ibkr_broker`, which enforces,
in order:

1. the operational mode is ``ACCOUNT_TRADING``,
2. live trading is **armed** (the typed-confirmation lock, :class:`LiveTradingArming`),
3. credentials resolve from the environment (via :class:`SecretRef`).

If any check fails, construction raises -- it never silently degrades to a no-op that looks
like it worked. The actual IBKR SDK calls are a single, clearly-marked integration seam
(:meth:`IBKRBroker._place_order_impl`); until that seam is wired to a real account it raises
``NotImplementedError`` with an actionable message. This means **the code cannot place a real
order until an engineer deliberately implements that one method against the operator's own
credentials and signs off** -- exactly the control the stakes demand.
"""

from __future__ import annotations

import datetime as _dt

from ..core.config import SecretRef
from ..core.enums import OperationalMode
from ..core.errors import ConfigurationError, ExecutionError
from ..core.logging import get_logger
from .broker import Broker, BrokerPosition
from .live_guard import LiveTradingArming
from .orders import Fill, Order

__all__ = ["IBKRBroker", "IBKRConfig", "create_ibkr_broker"]

_logger = get_logger(__name__)


class IBKRConfig:
    """Connection configuration for the IBKR adapter (credentials via env, never inline)."""

    __slots__ = ("account_secret", "client_id", "host", "port")

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 7496,
        client_id: int = 1,
        account_secret: SecretRef,
    ) -> None:
        if not host.strip():
            raise ConfigurationError("host must be non-empty", context={})
        if not 1 <= port <= 65535:
            raise ConfigurationError("port must be a valid TCP port", context={"port": port})
        if not isinstance(account_secret, SecretRef):
            raise ConfigurationError("account_secret must be a SecretRef", context={})
        self.host = host
        self.port = port
        self.client_id = client_id
        self.account_secret = account_secret


class IBKRBroker(Broker):
    """Live IBKR broker. Construct via :func:`create_ibkr_broker`, never directly in code paths.

    The constructor itself re-checks arming and credentials as a defense-in-depth measure, so
    even a direct instantiation cannot bypass the locks.
    """

    def __init__(self, *, config: IBKRConfig, arming: LiveTradingArming) -> None:
        # Defense in depth: re-enforce the locks even on direct construction.
        # Order matters: arming is checked first (free), then credentials (free), then the
        # optional SDK is imported. This guarantees the credential-lock test fails on the
        # missing-env-var path even when ib_insync is not installed.
        arming.require_armed()
        self._account = config.account_secret.resolve()  # fails fast if env var unset
        self._config = config
        self._arming = arming
        self._connected = False
        # Only now pull in the optional SDK; a missing install surfaces as ImportError,
        # which is a clear, actionable failure for the operator rather than a silent
        # no-op.
        from ib_insync import IB

        self._ib = IB()
        _logger.warning(
            "ibkr_broker_constructed_live",
            extra={"host": config.host, "port": config.port, "client_id": config.client_id},
        )

    def _connect(self) -> None:
        if not self._connected:
            self._ib.connect(self._config.host, self._config.port, clientId=self._config.client_id)
            self._connected = True

    @property
    def is_live(self) -> bool:
        """Always True: this broker moves real capital."""
        return True

    def submit(self, order: Order) -> Fill:
        """Submit a live order via the IBKR SDK (the single integration seam).

        Defers directly to :meth:`_place_order_impl` without first connecting to TWS: the
        integration seam is the only place that talks to a live broker, and it raises
        :class:`NotImplementedError` until an operator deliberately wires it. That makes
        the safety contract testable without a running IBKR TWS and matches ``SAFETY.md``:
        "the final placement seam is deliberately unimplemented until an operator wires
        and signs off on it".
        """
        if not isinstance(order, Order):
            raise ExecutionError("order must be an Order", context={})
        return self._place_order_impl(order)

    def cancel(self, client_order_id: str) -> bool:
        """Cancel a live working order (integration seam).

        Defers to the placement seam for the same reason as :meth:`submit`: the cancel
        path is intentionally unimplemented until the broker wiring is finalised.
        """
        return _raise_unimplemented_cancel(client_order_id)

    def positions(self) -> tuple[BrokerPosition, ...]:
        """Return live positions for reconciliation (integration seam).

        Defers to the placement seam until the broker wiring is finalised.
        """
        return _raise_unimplemented_positions()

    def _place_order_impl(self, order: Order) -> Fill:
        """The single, deliberately unimplemented seam where a real IBKR order is sent.

        Per ``SAFETY.md`` this method is the only place the engine talks to a live broker
        and it MUST raise :class:`NotImplementedError` until an operator deliberately wires
        it against their own credentials and signs off. The body below documents the
        intended shape of the implementation (build a ``Bag`` combo contract from the four
        condor legs, submit a single combo limit order) without executing any of it, so a
        stray config flag or typo cannot route a real order. When the operator is ready,
        they replace the ``raise`` with the documented block and remove the explicit gate.

        Reference implementation (deliberately inert until wired):
        --------------------------------------------------------------------
        from ib_insync import Bag, ComboLeg, Option as IBOption, Order as IBOrder

        combo_legs: list[ComboLeg] = []
        for leg in order.condor.legs():
            right = "C" if leg.option.right.value == "CALL" else "P"
            days = max(1, int(round(leg.option.expiry * 365.25)))
            expiry_date = order.created_at + _dt.timedelta(days=days)
            expiry_str = expiry_date.strftime("%Y%m%d")
            contract = IBOption(
                symbol=order.symbol,
                lastTradeDateOrContractMonth=expiry_str,
                strike=leg.option.strike,
                right=right,
                exchange="SMART",
                currency="USD",
            )
            self._ib.qualifyContracts(contract)
            combo_legs.append(ComboLeg(conId=contract.conId, ratio=abs(leg.quantity),
                                       action="BUY" if leg.quantity > 0 else "SELL",
                                       exchange="SMART"))
        bag = Bag()
        for cl in combo_legs:
            bag.append(cl)
        self._ib.qualifyContracts(bag)
        ib_order = IBOrder(
            action="BUY" if order.side.value == "BUY" else "SELL",
            totalQuantity=order.quantity,
            orderType="LMT",
            lmtPrice=order.limit_credit,
        )
        trade = self._ib.placeOrder(bag, ib_order)
        self._ib.sleep(1)
        return Fill(
            client_order_id=order.client_order_id,
            filled_quantity=int(trade.filled()),
            fill_credit=float(trade.avgFillPrice()),
            multiplier=order.multiplier,
            timestamp=trade.log[-1].time if trade.log else order.created_at,
        )
        --------------------------------------------------------------------
        """
        raise NotImplementedError(
            "IBKRBroker._place_order_impl is the deliberately unimplemented live-order seam; "
            "wire it against your operator credentials and sign off before any real order can "
            "flow. See src/options_engine/execution/SAFETY.md."
        )


def create_ibkr_broker(
    *,
    operational_mode: OperationalMode,
    config: IBKRConfig,
    arming: LiveTradingArming,
) -> IBKRBroker:
    """Create a live IBKR broker, enforcing all real-order locks (the only sanctioned entry).

    Raises :class:`ConfigurationError` unless the operational mode is ``ACCOUNT_TRADING`` and
    live trading is armed; raises if credentials are unset. There is no path to a live broker
    that bypasses these checks.
    """
    if operational_mode is not OperationalMode.ACCOUNT_TRADING:
        raise ConfigurationError(
            "a live broker may only be created in ACCOUNT_TRADING mode",
            context={"operational_mode": operational_mode.value},
        )
    arming.require_armed()
    return IBKRBroker(config=config, arming=arming)


def _raise_unimplemented_cancel(client_order_id: str) -> bool:
    """Stand-in for ``IBKRBroker.cancel`` until the live-broker wiring is finalised.

    Mirrors the deliberate failure mode of the placement seam: a single, clearly-marked
    integration point that must be implemented before any real cancel can be sent.
    """
    raise NotImplementedError(
        "IBKRBroker.cancel requires the live-broker wiring to be finalised; "
        "see src/options_engine/execution/SAFETY.md."
    )


def _raise_unimplemented_positions() -> tuple[BrokerPosition, ...]:
    """Stand-in for ``IBKRBroker.positions`` until the live-broker wiring is finalised."""
    raise NotImplementedError(
        "IBKRBroker.positions requires the live-broker wiring to be finalised; "
        "see src/options_engine/execution/SAFETY.md."
    )
