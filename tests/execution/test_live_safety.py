"""Exhaustive tests for the live-trading safety locks (the money-protection layer).

These tests are deliberately thorough: a defect here can route a real order by accident. They
verify that *every* lock independently prevents live trading, and that real orders are
impossible unless all locks are satisfied at once.
"""

from __future__ import annotations

import datetime as dt

import pytest

from options_engine.core.config import SecretRef
from options_engine.core.enums import OperationalMode, OrderSide
from options_engine.core.errors import ConfigurationError
from options_engine.execution.ibkr import IBKRBroker, IBKRConfig, create_ibkr_broker
from options_engine.execution.live_guard import LiveTradingArming
from options_engine.execution.orders import Order
from options_engine.pricing.instruments import IronCondor

UTC = dt.UTC
NOW = dt.datetime(2024, 1, 1, tzinfo=UTC)


def _ibkr_config(env_var: str = "IBKR_ACCT_TEST") -> IBKRConfig:
    return IBKRConfig(account_secret=SecretRef(env_var=env_var))


def _armed() -> LiveTradingArming:
    return LiveTradingArming(
        enable_live_trading=True, confirmation_phrase=LiveTradingArming.REQUIRED_PHRASE
    )


def _order() -> Order:
    return Order(
        client_order_id="X1",
        condor=IronCondor(90.0, 95.0, 105.0, 110.0, 0.05, quantity=1),
        side=OrderSide.SELL,
        quantity=1,
        limit_credit=1.0,
        multiplier=100.0,
        created_at=NOW,
    )


class TestLiveTradingArming:
    def test_default_is_disarmed(self) -> None:
        assert LiveTradingArming().is_armed is False
        assert LiveTradingArming.disarmed().is_armed is False

    def test_boolean_alone_does_not_arm(self) -> None:
        # enable flag without the typed phrase is NOT armed.
        assert LiveTradingArming(enable_live_trading=True).is_armed is False

    def test_phrase_alone_does_not_arm(self) -> None:
        # phrase without the enable flag is NOT armed.
        assert (
            LiveTradingArming(confirmation_phrase=LiveTradingArming.REQUIRED_PHRASE).is_armed
            is False
        )

    def test_wrong_phrase_does_not_arm(self) -> None:
        assert (
            LiveTradingArming(enable_live_trading=True, confirmation_phrase="yes").is_armed
            is False
        )

    def test_exact_phrase_arms(self) -> None:
        assert _armed().is_armed is True

    def test_phrase_is_class_constant_not_field(self) -> None:
        # REQUIRED_PHRASE must be accessible at class level (ClassVar), not a constructor arg.
        assert isinstance(LiveTradingArming.REQUIRED_PHRASE, str)
        with pytest.raises(TypeError):
            LiveTradingArming(REQUIRED_PHRASE="hacked")  # type: ignore[call-arg]

    def test_require_armed_raises_when_disarmed(self) -> None:
        with pytest.raises(ConfigurationError):
            LiveTradingArming.disarmed().require_armed()

    def test_require_armed_error_does_not_echo_phrase(self) -> None:
        # The error must not leak the phrase (so it can't be copy-pasted to bypass typing it).
        try:
            LiveTradingArming(enable_live_trading=True, confirmation_phrase="x").require_armed()
        except ConfigurationError as exc:
            assert LiveTradingArming.REQUIRED_PHRASE not in str(exc)


class TestCreateIBKRBrokerLocks:
    def test_lock_wrong_mode(self) -> None:
        with pytest.raises(ConfigurationError):
            create_ibkr_broker(
                operational_mode=OperationalMode.HISTORY_BACKTEST,
                config=_ibkr_config(),
                arming=_armed(),
            )

    def test_lock_live_backtest_mode(self) -> None:
        with pytest.raises(ConfigurationError):
            create_ibkr_broker(
                operational_mode=OperationalMode.LIVE_BACKTEST,
                config=_ibkr_config(),
                arming=_armed(),
            )

    def test_lock_disarmed(self) -> None:
        with pytest.raises(ConfigurationError):
            create_ibkr_broker(
                operational_mode=OperationalMode.ACCOUNT_TRADING,
                config=_ibkr_config(),
                arming=LiveTradingArming.disarmed(),
            )

    def test_lock_missing_credentials(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("IBKR_ACCT_TEST", raising=False)
        with pytest.raises(ConfigurationError):
            create_ibkr_broker(
                operational_mode=OperationalMode.ACCOUNT_TRADING,
                config=_ibkr_config(),
                arming=_armed(),
            )

    def test_all_locks_satisfied_constructs_but_cannot_place(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # With mode + arming + credentials, the broker constructs -- but the final placement
        # seam is intentionally unimplemented, so a real order STILL cannot be sent.
        monkeypatch.setenv("IBKR_ACCT_TEST", "DU1234567")
        broker = create_ibkr_broker(
            operational_mode=OperationalMode.ACCOUNT_TRADING,
            config=_ibkr_config(),
            arming=_armed(),
        )
        assert broker.is_live is True
        with pytest.raises(NotImplementedError):
            broker.submit(_order())

    def test_direct_construction_still_enforces_arming(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Defense in depth: even constructing IBKRBroker directly cannot bypass arming.
        monkeypatch.setenv("IBKR_ACCT_TEST", "DU1234567")
        with pytest.raises(ConfigurationError):
            IBKRBroker(config=_ibkr_config(), arming=LiveTradingArming.disarmed())
