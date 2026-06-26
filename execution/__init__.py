"""Order-management system and broker adapters (SPEC §7).

Fail-closed execution: the simulated broker is the default everywhere, and routing a real
order requires the operational mode, credentials, an explicit enable flag, a typed
confirmation phrase, and risk-supervisor approval to all hold at once. See ``SAFETY.md``.

Public surface:

* Orders: :class:`Order`, :class:`Fill`, :class:`OrderState`.
* Brokers: :class:`Broker`, :class:`SimulatedBroker`, :class:`BrokerPosition`,
  :class:`IBKRBroker`, :class:`IBKRConfig`, :func:`create_ibkr_broker`.
* Live safety: :class:`LiveTradingArming`.
* OMS: :class:`OrderManagementSystem`, :class:`ReconciliationResult`.
"""

from __future__ import annotations

from .broker import Broker, BrokerPosition, SimulatedBroker
from .ibkr import IBKRBroker, IBKRConfig, create_ibkr_broker
from .live_guard import LiveTradingArming
from .oms import OrderManagementSystem, ReconciliationResult
from .orders import Fill, Order, OrderState

__all__ = [
    "Broker",
    "BrokerPosition",
    "Fill",
    "IBKRBroker",
    "IBKRConfig",
    "LiveTradingArming",
    "Order",
    "OrderManagementSystem",
    "OrderState",
    "ReconciliationResult",
    "SimulatedBroker",
    "create_ibkr_broker",
]
