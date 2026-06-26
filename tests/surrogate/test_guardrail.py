"""Tests for the surrogate Monte-Carlo fallback guardrail (SPEC §2.5)."""

from __future__ import annotations

import pytest

from options_engine.core.errors import ValidationError
from options_engine.core.random import RandomFactory
from options_engine.models.rbergomi.results import TerminalDistribution
from options_engine.surrogate.dataset import generate_training_data
from options_engine.surrogate.distribution import SurrogateDistribution
from options_engine.surrogate.guardrail import (
    GuardrailConfig,
    SurrogateGuardrail,
)
from options_engine.surrogate.surrogate import DistributionSurrogate, TrainingConfig


@pytest.fixture(scope="module")
def trained_surrogate() -> DistributionSurrogate:
    data = generate_training_data(
        n_scenarios=120, rng_factory=RandomFactory(1), n_paths=6000, steps_per_day=3
    )
    s = DistributionSurrogate()
    s.fit(
        data,
        config=TrainingConfig(
            hidden_sizes=(64, 64), max_epochs=120, patience=20, learning_rate=2e-3, seed=0
        ),
    )
    return s


class TestGuardrailConfig:
    def test_rejects_bad_values(self) -> None:
        with pytest.raises(ValidationError):
            GuardrailConfig(max_wasserstein=0.0)
        with pytest.raises(ValidationError):
            GuardrailConfig(audit_every=0)


class TestSurrogateGuardrail:
    def test_requires_trained_surrogate(self) -> None:
        with pytest.raises(ValidationError):
            SurrogateGuardrail(DistributionSurrogate(), rng_factory=RandomFactory(0))

    def test_first_request_audits(self, trained_surrogate) -> None:
        g = SurrogateGuardrail(
            trained_surrogate,
            rng_factory=RandomFactory(5),
            config=GuardrailConfig(audit_n_paths=6000, steps_per_day=3),
        )
        g.distribution(hurst=0.1, eta=1.5, rho=-0.7, xi0=0.04, horizon=10 / 252)
        assert g.last_decision is not None
        assert g.last_decision.audited

    def test_non_audit_requests_use_surrogate(self, trained_surrogate) -> None:
        g = SurrogateGuardrail(
            trained_surrogate,
            rng_factory=RandomFactory(5),
            config=GuardrailConfig(audit_every=100, audit_n_paths=6000, steps_per_day=3),
        )
        # First request audits; the next few should not.
        g.distribution(hurst=0.1, eta=1.5, rho=-0.7, xi0=0.04, horizon=10 / 252)
        dist = g.distribution(hurst=0.1, eta=1.5, rho=-0.7, xi0=0.04, horizon=10 / 252)
        assert isinstance(dist, SurrogateDistribution)
        assert g.last_decision is not None
        assert not g.last_decision.audited
        assert g.last_decision.used_surrogate

    def test_degraded_surrogate_falls_back_to_mc(self, trained_surrogate) -> None:
        # With an impossibly tight tolerance, any discrepancy forces a Monte-Carlo fallback.
        g = SurrogateGuardrail(
            trained_surrogate,
            rng_factory=RandomFactory(7),
            config=GuardrailConfig(max_wasserstein=1e-9, audit_n_paths=6000, steps_per_day=3),
        )
        dist = g.distribution(hurst=0.1, eta=1.5, rho=-0.7, xi0=0.04, horizon=10 / 252)
        assert isinstance(dist, TerminalDistribution)  # fell back to MC
        assert g.last_decision is not None
        assert g.last_decision.degraded
        assert g.last_decision.used_monte_carlo

    def test_force_audit(self, trained_surrogate) -> None:
        g = SurrogateGuardrail(
            trained_surrogate,
            rng_factory=RandomFactory(9),
            config=GuardrailConfig(audit_every=1000, audit_n_paths=6000, steps_per_day=3),
        )
        g.distribution(hurst=0.1, eta=1.5, rho=-0.7, xi0=0.04, horizon=10 / 252)  # first audits
        g.distribution(hurst=0.1, eta=1.5, rho=-0.7, xi0=0.04, horizon=10 / 252)  # no audit
        g.distribution(hurst=0.1, eta=1.5, rho=-0.7, xi0=0.04, horizon=10 / 252, force_audit=True)
        assert g.last_decision is not None
        assert g.last_decision.audited
