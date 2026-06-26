"""Tests for typed configuration and secret handling."""

from __future__ import annotations

import pytest
from pydantic import ValidationError as PydanticValidationError

from options_engine.core.config import (
    EngineConfig,
    MonteCarloConfig,
    RiskConfig,
    SecretRef,
)
from options_engine.core.enums import OperationalMode, TradingMode
from options_engine.core.errors import ConfigurationError


def test_default_engine_config_valid() -> None:
    cfg = EngineConfig(operational_mode=OperationalMode.HISTORY_BACKTEST)
    assert cfg.trading_mode is TradingMode.NORMAL
    assert isinstance(cfg.risk, RiskConfig)
    assert isinstance(cfg.monte_carlo, MonteCarloConfig)


def test_extra_keys_forbidden() -> None:
    with pytest.raises(PydanticValidationError):
        EngineConfig(operational_mode=OperationalMode.HISTORY_BACKTEST, typo_field=1)  # type: ignore[call-arg]


def test_config_is_frozen() -> None:
    cfg = EngineConfig(operational_mode=OperationalMode.HISTORY_BACKTEST)
    with pytest.raises(PydanticValidationError):
        cfg.trading_mode = TradingMode.MFD  # type: ignore[misc]


def test_risk_daily_must_dominate_per_trade() -> None:
    with pytest.raises(ConfigurationError):
        RiskConfig(max_risk_fraction_per_trade=0.10, max_risk_fraction_per_day=0.05)


def test_risk_bounds_enforced() -> None:
    with pytest.raises(PydanticValidationError):
        RiskConfig(max_risk_fraction_per_trade=0.5)  # exceeds le=0.25
    with pytest.raises(PydanticValidationError):
        RiskConfig(max_leverage=10.0)  # exceeds le=4.0


def test_monte_carlo_bounds() -> None:
    with pytest.raises(PydanticValidationError):
        MonteCarloConfig(n_paths=10)  # below ge=1000
    cfg = MonteCarloConfig(n_paths=10_000, seed=7)
    assert cfg.seed == 7


def test_fingerprint_is_stable_and_order_independent() -> None:
    a = EngineConfig(operational_mode=OperationalMode.HISTORY_BACKTEST)
    b = EngineConfig(operational_mode=OperationalMode.HISTORY_BACKTEST)
    assert a.fingerprint() == b.fingerprint()


def test_fingerprint_changes_with_content() -> None:
    a = EngineConfig(operational_mode=OperationalMode.HISTORY_BACKTEST)
    b = EngineConfig(operational_mode=OperationalMode.LIVE_BACKTEST)
    assert a.fingerprint() != b.fingerprint()


def test_secret_ref_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    ref = SecretRef(env_var="OPTIONS_ENGINE_TEST_SECRET")
    assert not ref.is_available()
    with pytest.raises(ConfigurationError):
        ref.resolve()
    monkeypatch.setenv("OPTIONS_ENGINE_TEST_SECRET", "s3cr3t")
    assert ref.is_available()
    assert ref.resolve() == "s3cr3t"


def test_secret_ref_empty_value_treated_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPTIONS_ENGINE_TEST_SECRET", "")
    ref = SecretRef(env_var="OPTIONS_ENGINE_TEST_SECRET")
    assert not ref.is_available()
    with pytest.raises(ConfigurationError):
        ref.resolve()
