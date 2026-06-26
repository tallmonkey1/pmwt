"""Domain enumerations shared across the engine.

Centralizing these prevents the "stringly-typed" anti-pattern (SPEC §13: correctness,
consistency). Every enum is a ``str``-backed :class:`enum.Enum` so values serialize
cleanly to JSON/config while remaining type-safe in code.
"""

from __future__ import annotations

from enum import StrEnum

__all__ = [
    "OperationalMode",
    "OptionRight",
    "OptionStyle",
    "OrderSide",
    "OrderStatus",
    "StrategicAction",
    "TradingMode",
    "VolRegime",
]


class TradingMode(StrEnum):
    """Decision/holding cadence (SPEC §1.1)."""

    NORMAL = "NORMAL"  # 1 hour to 1 day
    MFD = "MFD"  # 1 minute to 1 hour (medium-frequency decisioning)


class OperationalMode(StrEnum):
    """How the engine is being run (SPEC §1.2)."""

    HISTORY_BACKTEST = "HISTORY_BACKTEST"
    LIVE_BACKTEST = "LIVE_BACKTEST"  # paper trading on live data
    ACCOUNT_TRADING = "ACCOUNT_TRADING"  # real capital


class VolRegime(StrEnum):
    """Volatility regime labels produced by the regime layer (SPEC §2.6)."""

    LOW = "LOW"
    MID = "MID"
    HIGH = "HIGH"


class OptionRight(StrEnum):
    """Option right."""

    CALL = "CALL"
    PUT = "PUT"


class OptionStyle(StrEnum):
    """Exercise style. Index options targeted here are typically European/cash-settled."""

    EUROPEAN = "EUROPEAN"
    AMERICAN = "AMERICAN"


class StrategicAction(StrEnum):
    """Top-level decision of the RL strategic head (SPEC §4.1).

    Three strategic choices, exactly as specified:

    * ``HARVEST_THETA`` -- sell an iron condor (theta capture).
    * ``HARVEST_GAMMA`` -- dynamically hedge / use the directional trade book.
    * ``FLAT`` -- no new risk / stay out (free-zero baseline).
    """

    HARVEST_THETA = "HARVEST_THETA"
    HARVEST_GAMMA = "HARVEST_GAMMA"
    FLAT = "FLAT"  # no new risk / stay out


class OrderSide(StrEnum):
    """Order side."""

    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(StrEnum):
    """Lifecycle states for the order-management state machine (SPEC §7 execution/oms)."""

    PENDING_NEW = "PENDING_NEW"
    WORKING = "WORKING"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
