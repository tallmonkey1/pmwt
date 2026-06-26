r"""Operational-mode runners and the broker factory (SPEC §1.2, §7).

Three operational modes share one decision pipeline (the single-source-of-truth design) and
differ only in their data and broker wiring:

* ``HISTORY_BACKTEST`` -- historical/synthetic data, simulated broker (offline, reproducible).
* ``LIVE_BACKTEST`` -- live data, simulated fills (paper trading; no real orders possible).
* ``ACCOUNT_TRADING`` -- real broker, real capital -- reachable only through the full safety
  gauntlet (see ``execution/SAFETY.md``).

:func:`create_broker` is the single factory that decides which broker to build, and it is
**fail-closed**: it returns a :class:`SimulatedBroker` for every mode except
``ACCOUNT_TRADING``, and even for ``ACCOUNT_TRADING`` it only builds a live broker when live
trading is fully armed *and* an :class:`IBKRConfig` is supplied -- otherwise it refuses (it
does not silently fall back to simulation in live mode, because a silent fallback in the mode
the operator believes is live would itself be dangerous).
"""

from __future__ import annotations

from ..core.config import EngineConfig
from ..core.enums import OperationalMode
from ..core.errors import ConfigurationError
from ..core.logging import get_logger
from ..core.random import RandomFactory
from ..execution.broker import Broker, SimulatedBroker
from ..execution.ibkr import IBKRConfig, create_ibkr_broker
from ..execution.live_guard import LiveTradingArming

__all__ = ["create_broker"]

_logger = get_logger(__name__)


def create_broker(
    *,
    config: EngineConfig,
    arming: LiveTradingArming | None = None,
    ibkr_config: IBKRConfig | None = None,
    rng_factory: RandomFactory | None = None,
) -> Broker:
    """Return the broker appropriate to the operational mode, fail-closed.

    Parameters
    ----------
    config:
        The engine configuration (its ``operational_mode`` selects the broker).
    arming:
        The live-trading arming. Required (and must be armed) for ``ACCOUNT_TRADING``.
        Ignored in non-live modes.
    ibkr_config:
        The live-broker connection config. Required for ``ACCOUNT_TRADING``.
    rng_factory:
        Randomness for the simulated broker (non-live modes).

    Returns
    -------
    Broker
        A :class:`SimulatedBroker` for HISTORY_BACKTEST and LIVE_BACKTEST; an
        :class:`~options_engine.execution.ibkr.IBKRBroker` for ACCOUNT_TRADING when fully
        armed and configured.

    Raises
    ------
    ConfigurationError
        In ``ACCOUNT_TRADING`` when arming is missing/disarmed or ``ibkr_config`` is absent.
        We refuse rather than silently simulate, so the operator is never misled about whether
        real orders can flow.
    """
    mode = config.operational_mode

    if mode in (OperationalMode.HISTORY_BACKTEST, OperationalMode.LIVE_BACKTEST):
        # Both non-live modes use the simulated broker; real orders are physically impossible.
        _logger.info("broker_created_simulated", extra={"operational_mode": mode.value})
        return SimulatedBroker(rng_factory=rng_factory or RandomFactory(config.monte_carlo.seed))

    if mode is OperationalMode.ACCOUNT_TRADING:
        if arming is None or not arming.is_armed:
            raise ConfigurationError(
                "ACCOUNT_TRADING requires fully-armed live trading (enable flag + typed "
                "confirmation phrase); refusing to start. See execution/SAFETY.md.",
                context={"operational_mode": mode.value},
            )
        if ibkr_config is None:
            raise ConfigurationError(
                "ACCOUNT_TRADING requires an IBKRConfig with credentials; refusing to start.",
                context={"operational_mode": mode.value},
            )
        _logger.warning("broker_created_live", extra={"operational_mode": mode.value})
        return create_ibkr_broker(operational_mode=mode, config=ibkr_config, arming=arming)

    raise ConfigurationError(  # pragma: no cover - enum is exhaustive
        "unknown operational mode", context={"operational_mode": str(mode)}
    )
