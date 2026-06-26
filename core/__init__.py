"""Core foundation layer for the options engine.

This package contains the cross-cutting primitives every other module depends on:
structured errors, input validation, reproducible RNG, structured logging, domain enums,
time grids, typed configuration, and the alpha-to-model-parameter framework that the
helper-critic agent optimises. It has no dependencies on higher-level packages, keeping
the dependency graph acyclic (SPEC §13: maintainability, low coupling).
"""

from __future__ import annotations

from .config import EngineConfig, MonteCarloConfig, RiskConfig, SecretRef
from .enums import (
    OperationalMode,
    OptionRight,
    OptionStyle,
    OrderSide,
    OrderStatus,
    StrategicAction,
    TradingMode,
    VolRegime,
)
from .errors import (
    CalibrationError,
    ConfigurationError,
    ConvergenceError,
    DataError,
    ExecutionError,
    ModelStateError,
    NumericalError,
    OptionsEngineError,
    RiskLimitError,
    ValidationError,
)
from .logging import bind_context, configure_logging, get_logger
from .market_alpha import (
    ALPHA_DIM,
    DEFAULT_ALPHA_DIM,
    MarketAlpha,
    alpha_components,
    alpha_to_drift_noise,
    alpha_to_eta,
    alpha_to_hurst,
    alpha_to_jump_intensity,
    alpha_to_jump_size,
    alpha_to_shock_intensity,
    alpha_to_stoikov_noise,
)
from .random import RandomFactory, default_factory
from .timegrid import TRADING_DAYS_PER_YEAR, TimeGrid

__all__ = [
    "ALPHA_DIM",
    "DEFAULT_ALPHA_DIM",
    "MarketAlpha",
    "TRADING_DAYS_PER_YEAR",
    # alpha mappings
    "alpha_components",
    "alpha_to_drift_noise",
    "alpha_to_eta",
    "alpha_to_hurst",
    "alpha_to_jump_intensity",
    "alpha_to_jump_size",
    "alpha_to_shock_intensity",
    "alpha_to_stoikov_noise",
    # errors
    "CalibrationError",
    "ConfigurationError",
    "ConvergenceError",
    "DataError",
    # config
    "EngineConfig",
    "ExecutionError",
    "ModelStateError",
    "MonteCarloConfig",
    "NumericalError",
    # enums
    "OperationalMode",
    "OptionRight",
    "OptionStyle",
    "OptionsEngineError",
    "OrderSide",
    "OrderStatus",
    # random
    "RandomFactory",
    "RiskConfig",
    "RiskLimitError",
    "SecretRef",
    "StrategicAction",
    # timegrid
    "TimeGrid",
    "TradingMode",
    "ValidationError",
    "VolRegime",
    # logging
    "bind_context",
    "configure_logging",
    "default_factory",
    "get_logger",
] 
