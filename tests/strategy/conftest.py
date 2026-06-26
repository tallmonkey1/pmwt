"""Shared fixtures for strategy tests: a distribution, a quoted chain, and helpers."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from options_engine.core.enums import VolRegime
from options_engine.core.random import RandomFactory
from options_engine.core.timegrid import TimeGrid
from options_engine.market import AvellanedaStoikovMaker, MarketMakerConfig, build_synthetic_chain
from options_engine.models.rbergomi import (
    ForwardVariance,
    HybridSimulator,
    RBergomiParams,
    build_terminal_distribution,
)
from options_engine.regime.detector import RegimeNowcast

UTC = dt.UTC


@pytest.fixture(scope="session")
def market_setup():
    """A terminal distribution, its terminal sample, and a quoted synthetic chain."""
    params = RBergomiParams(
        hurst=0.1, eta=1.3, rho=-0.7, forward_variance=ForwardVariance.flat(0.04)
    )
    grid = TimeGrid.from_calendar_days(calendar_days=14, steps_per_day=8)
    paths = HybridSimulator(params, rng_factory=RandomFactory(1), antithetic=True).simulate(
        grid=grid, n_paths=40_000, initial_spot=100.0
    )
    dist = build_terminal_distribution(paths)
    maker = AvellanedaStoikovMaker(
        config=MarketMakerConfig(risk_aversion=0.3, order_flow_intensity=4.0), tick_size=0.05
    )
    chain = build_synthetic_chain(dist, maker=maker, strikes=np.arange(70.0, 131.0, 1.0), rate=0.0)
    return {"dist": dist, "terminal_sample": paths.terminal_spot(), "chain": chain}


def calm_regime() -> RegimeNowcast:
    """A confidently-calm regime nowcast."""
    return RegimeNowcast(
        current_probabilities={VolRegime.LOW: 0.85, VolRegime.MID: 0.10, VolRegime.HIGH: 0.05},
        next_probabilities={VolRegime.LOW: 0.80, VolRegime.MID: 0.15, VolRegime.HIGH: 0.05},
    )


def stressed_regime() -> RegimeNowcast:
    """A high-vol regime nowcast that should fail the gate."""
    return RegimeNowcast(
        current_probabilities={VolRegime.LOW: 0.20, VolRegime.MID: 0.30, VolRegime.HIGH: 0.50},
        next_probabilities={VolRegime.LOW: 0.20, VolRegime.MID: 0.30, VolRegime.HIGH: 0.50},
    )
