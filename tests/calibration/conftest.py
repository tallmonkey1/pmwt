"""Shared fixtures for calibration tests.

Provides synthetic rBergomi paths with *known* parameters so estimators can be validated by
recovering what was injected (the only honest way to test a calibration).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from options_engine.core.random import RandomFactory
from options_engine.core.timegrid import TimeGrid
from options_engine.models.rbergomi import (
    ForwardVariance,
    HybridSimulator,
    RBergomiParams,
)


@dataclass(frozen=True)
class SyntheticPath:
    """A single simulated rBergomi path with its generating parameters."""

    hurst: float
    eta: float
    rho: float
    xi0: float
    log_price: np.ndarray
    log_variance: np.ndarray
    grid: TimeGrid


def make_path(
    *,
    hurst: float = 0.11,
    eta: float = 1.6,
    rho: float = -0.7,
    xi0: float = 0.04,
    horizon_years: float = 4.0,
    steps_per_year: int = 252 * 4,
    seed: int = 5,
) -> SyntheticPath:
    """Simulate one long, finely-sampled rBergomi path with known parameters."""
    n_steps = int(horizon_years * steps_per_year)
    grid = TimeGrid(horizon_years=horizon_years, n_steps=n_steps)
    params = RBergomiParams(
        hurst=hurst, eta=eta, rho=rho, forward_variance=ForwardVariance.flat(xi0)
    )
    paths = HybridSimulator(params, rng_factory=RandomFactory(seed)).simulate(
        grid=grid, n_paths=1, initial_spot=100.0
    )
    return SyntheticPath(
        hurst=hurst,
        eta=eta,
        rho=rho,
        xi0=xi0,
        log_price=np.log(paths.spot[0]),
        log_variance=np.log(paths.variance[0]),
        grid=grid,
    )


@pytest.fixture(scope="module")
def clean_path() -> SyntheticPath:
    """A clean (latent) synthetic path shared across estimator tests."""
    return make_path()
