"""Tests for Monte-Carlo diagnostics and terminal-distribution aggregation."""

from __future__ import annotations

import numpy as np
import pytest

from options_engine.core.errors import ConvergenceError, ValidationError
from options_engine.core.random import RandomFactory
from options_engine.core.timegrid import TimeGrid
from options_engine.models.rbergomi.diagnostics import (
    build_terminal_distribution,
    mean_standard_error,
)
from options_engine.models.rbergomi.params import ForwardVariance, RBergomiParams
from options_engine.models.rbergomi.simulator import HybridSimulator


def _paths(n_paths: int = 5000):
    params = RBergomiParams(
        hurst=0.1, eta=1.0, rho=-0.5, forward_variance=ForwardVariance.flat(0.04)
    )
    sim = HybridSimulator(params, rng_factory=RandomFactory(7))
    grid = TimeGrid(horizon_years=0.5, n_steps=40)
    return sim.simulate(grid=grid, n_paths=n_paths, initial_spot=100.0)


class TestMeanStandardError:
    def test_basic_estimate(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.normal(2.0, 3.0, size=100_000)
        summary = mean_standard_error(x)
        assert summary.estimate == pytest.approx(2.0, abs=0.05)
        # se ~ sigma / sqrt(N).
        assert summary.standard_error == pytest.approx(3.0 / np.sqrt(100_000), rel=0.05)
        assert summary.n_samples == 100_000

    def test_confidence_interval_contains_truth(self) -> None:
        rng = np.random.default_rng(1)
        x = rng.normal(0.0, 1.0, size=50_000)
        summary = mean_standard_error(x)
        lo, hi = summary.confidence_interval(0.95)
        assert lo < 0.0 < hi

    def test_relative_standard_error(self) -> None:
        rng = np.random.default_rng(2)
        x = rng.normal(5.0, 1.0, size=10_000)
        summary = mean_standard_error(x)
        assert summary.relative_standard_error > 0.0

    def test_relative_se_infinite_for_zero_estimate(self) -> None:
        summary = mean_standard_error(np.array([-1.0, 1.0]))
        assert summary.relative_standard_error == float("inf")

    def test_requires_two_samples(self) -> None:
        with pytest.raises(ConvergenceError):
            mean_standard_error(np.array([1.0]))


class TestBuildTerminalDistribution:
    def test_builds_without_tolerance(self) -> None:
        dist = build_terminal_distribution(_paths())
        assert dist.n_paths == 5000
        assert dist.initial_spot == 100.0
        assert dist.mean_standard_error > 0.0

    def test_passes_loose_tolerance(self) -> None:
        dist = build_terminal_distribution(_paths(20_000), max_rel_standard_error=0.5)
        assert dist.n_paths == 20_000

    def test_fails_tight_tolerance(self) -> None:
        # With few paths and an extremely tight tolerance, the gate must trip.
        with pytest.raises(ConvergenceError):
            build_terminal_distribution(_paths(2000), max_rel_standard_error=1e-6)

    def test_tolerance_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            build_terminal_distribution(_paths(2000), max_rel_standard_error=0.0)
