"""Typed, validated configuration objects (SPEC §8, §13: validation, reproducibility).

Configuration is the contract between the operator and the engine. We use pydantic v2
models so that:

* every field is type-checked and range-validated at construction (fail fast);
* configs serialize deterministically to/from JSON for run manifests (reproducibility);
* secrets are *never* stored as plain config values — they are referenced by the name of
  the environment variable that holds them and resolved at runtime (security: secrets via
  env, secure defaults).

The top-level :class:`EngineConfig` aggregates the sub-configs needed to launch any of the
three operational modes. Sub-configs for not-yet-built phases (RL, execution, ...) will be
added in their respective phases without breaking this contract.
"""

from __future__ import annotations

import hashlib
import json
import os

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .enums import OperationalMode, TradingMode
from .errors import ConfigurationError

__all__ = [
    "EngineConfig",
    "MonteCarloConfig",
    "RiskConfig",
    "SecretRef",
]


class _StrictModel(BaseModel):
    """Base model with strict, immutable, extra-forbidding semantics.

    ``extra="forbid"`` catches typos in config keys (a common, costly operator error).
    ``frozen=True`` makes configs hashable and prevents accidental mutation after
    validation, supporting reproducibility.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, validate_assignment=True)


class SecretRef(_StrictModel):
    """A reference to a secret held in an environment variable.

    Storing the *name* (not the value) keeps secrets out of configs, logs, and run
    manifests. :meth:`resolve` reads the value at runtime and fails loudly if it is unset,
    so missing credentials are caught at startup rather than mid-trade.
    """

    env_var: str = Field(min_length=1, description="Name of the environment variable.")

    def resolve(self) -> str:
        """Return the secret value from the environment, or raise if absent/empty."""
        value = os.environ.get(self.env_var)
        if not value:
            raise ConfigurationError(
                "required secret is not set in the environment",
                context={"env_var": self.env_var},
            )
        return value

    def is_available(self) -> bool:
        """Return True if the referenced environment variable is set and non-empty."""
        return bool(os.environ.get(self.env_var))


class RiskConfig(_StrictModel):
    """Hard risk limits enforced by the deterministic risk supervisor (SPEC §4.5, §6).

    These are *hard* caps. They are intentionally separate from the RL policy so that the
    learned agent can never widen its own risk budget.
    """

    max_risk_fraction_per_trade: float = Field(
        default=0.02,
        gt=0.0,
        le=0.25,
        description="Max fraction of account equity at risk on a single condor.",
    )
    max_risk_fraction_per_day: float = Field(
        default=0.06,
        gt=0.0,
        le=0.50,
        description="Max fraction of account equity that may be newly risked per day.",
    )
    max_drawdown_kill_switch: float = Field(
        default=0.20,
        gt=0.0,
        le=0.90,
        description="Trailing drawdown fraction that triggers flatten-and-halt.",
    )
    max_leverage: float = Field(
        default=1.0,
        ge=1.0,
        le=4.0,
        description="Absolute ceiling on gross leverage (1.0 = no leverage).",
    )
    kelly_fraction: float = Field(
        default=0.25,
        gt=0.0,
        le=1.0,
        description="Fraction of full-Kelly to apply when sizing (1.0 = full Kelly).",
    )
    news_cooloff_days: int = Field(
        default=5,
        ge=0,
        le=30,
        description="Trading days to suspend new positions after a material news event.",
    )

    @model_validator(mode="after")
    def _check_daily_dominates_per_trade(self) -> RiskConfig:
        if self.max_risk_fraction_per_day < self.max_risk_fraction_per_trade:
            raise ConfigurationError(
                "max_risk_fraction_per_day must be >= max_risk_fraction_per_trade",
                context={
                    "per_day": self.max_risk_fraction_per_day,
                    "per_trade": self.max_risk_fraction_per_trade,
                },
            )
        return self


class MonteCarloConfig(_StrictModel):
    """Monte-Carlo settings governing the price simulator and distribution estimation."""

    n_paths: int = Field(
        default=50_000,
        ge=1_000,
        le=5_000_000,
        description="Number of simulated price paths.",
    )
    antithetic: bool = Field(
        default=True, description="Use antithetic variates for variance reduction."
    )
    use_quasi_random: bool = Field(
        default=True, description="Use Sobol QMC sequences instead of pseudo-random draws."
    )
    max_rel_standard_error: float = Field(
        default=0.01,
        gt=0.0,
        le=0.5,
        description="Max acceptable relative MC standard error for reported estimates.",
    )
    seed: int = Field(default=0, ge=0, description="Master RNG seed for reproducibility.")


class EngineConfig(_StrictModel):
    """Top-level engine configuration aggregating all sub-configs.

    Additional sub-configs (model, regime, RL, execution, broker) are added in later build
    phases. This object is the single thing required to construct and launch a run, and it
    is what gets hashed into the run manifest for reproducibility.
    """

    operational_mode: OperationalMode
    trading_mode: TradingMode = TradingMode.NORMAL
    risk: RiskConfig = Field(default_factory=RiskConfig)
    monte_carlo: MonteCarloConfig = Field(default_factory=MonteCarloConfig)

    @model_validator(mode="after")
    def _check_live_requirements(self) -> EngineConfig:
        # Defensive cross-field invariant: account trading must not silently run with a
        # zero risk budget, which would be a misconfiguration that disables all trading.
        if (
            self.operational_mode is OperationalMode.ACCOUNT_TRADING
            and self.risk.max_risk_fraction_per_day <= 0.0
        ):
            raise ConfigurationError(
                "account trading requires a positive daily risk budget", context={}
            )
        return self

    def fingerprint(self) -> str:
        """Return a stable SHA-256 hex digest of the configuration.

        Used in run manifests so a result can be tied to the exact config that produced
        it (reproducibility, auditability). The digest is order-independent because the
        JSON is dumped with sorted keys.
        """
        payload = self.model_dump(mode="json")
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
