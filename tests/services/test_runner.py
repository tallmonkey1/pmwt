"""Tests for the fail-closed broker factory."""

from __future__ import annotations

import pytest

from options_engine.core.config import EngineConfig, SecretRef
from options_engine.core.enums import OperationalMode
from options_engine.core.errors import ConfigurationError
from options_engine.execution.broker import SimulatedBroker
from options_engine.execution.ibkr import IBKRConfig
from options_engine.execution.live_guard import LiveTradingArming
from options_engine.services.runner import create_broker


def _ibkr() -> IBKRConfig:
    return IBKRConfig(account_secret=SecretRef(env_var="IBKR_ACCT_TEST"))


def _armed() -> LiveTradingArming:
    return LiveTradingArming(
        enable_live_trading=True, confirmation_phrase=LiveTradingArming.REQUIRED_PHRASE
    )


class TestCreateBroker:
    def test_history_backtest_is_simulated(self) -> None:
        broker = create_broker(
            config=EngineConfig(operational_mode=OperationalMode.HISTORY_BACKTEST)
        )
        assert isinstance(broker, SimulatedBroker)
        assert broker.is_live is False

    def test_live_backtest_is_simulated(self) -> None:
        broker = create_broker(config=EngineConfig(operational_mode=OperationalMode.LIVE_BACKTEST))
        assert isinstance(broker, SimulatedBroker)
        assert broker.is_live is False

    def test_account_trading_refuses_without_arming(self) -> None:
        with pytest.raises(ConfigurationError):
            create_broker(
                config=EngineConfig(operational_mode=OperationalMode.ACCOUNT_TRADING),
                ibkr_config=_ibkr(),
            )

    def test_account_trading_refuses_disarmed(self) -> None:
        with pytest.raises(ConfigurationError):
            create_broker(
                config=EngineConfig(operational_mode=OperationalMode.ACCOUNT_TRADING),
                arming=LiveTradingArming.disarmed(),
                ibkr_config=_ibkr(),
            )

    def test_account_trading_refuses_without_ibkr_config(self) -> None:
        with pytest.raises(ConfigurationError):
            create_broker(
                config=EngineConfig(operational_mode=OperationalMode.ACCOUNT_TRADING),
                arming=_armed(),
            )

    def test_account_trading_does_not_silently_simulate(self) -> None:
        # The dangerous failure mode: silently returning a simulated broker in live mode.
        # The factory must RAISE, never return a simulated broker for ACCOUNT_TRADING.
        with pytest.raises(ConfigurationError):
            create_broker(config=EngineConfig(operational_mode=OperationalMode.ACCOUNT_TRADING))

    def test_account_trading_builds_live_when_fully_armed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("IBKR_ACCT_TEST", "DU7654321")
        broker = create_broker(
            config=EngineConfig(operational_mode=OperationalMode.ACCOUNT_TRADING),
            arming=_armed(),
            ibkr_config=_ibkr(),
        )
        assert broker.is_live is True
