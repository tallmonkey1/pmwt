"""Tests exercising the full successful-entry path with a controlled, genuine edge.

To make the test deterministic and independent of synthetic-chain calibration quirks, the
chain here is priced from a *richer*-than-realized distribution and uses tight wings so the
binary-Kelly criterion (deliberately conservative for partial-loss structures) admits a
positive size. This exercises the full entry pipeline -- selection, sizing, position
construction, and risk-supervisor approval/rejection -- end to end.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from options_engine.core.config import RiskConfig
from options_engine.core.random import RandomFactory
from options_engine.core.timegrid import TimeGrid
from options_engine.market import (
    AvellanedaStoikovMaker,
    MarketMakerConfig,
    build_synthetic_chain,
)
from options_engine.models.rbergomi import (
    ForwardVariance,
    HybridSimulator,
    RBergomiParams,
    build_terminal_distribution,
)
from options_engine.news import ReplayEventProvider, ReplayNewsProvider
from options_engine.news.gate import NewsGate
from options_engine.strategy.account import Account
from options_engine.strategy.condor_selection import CondorSelectionConfig
from options_engine.strategy.entry import EntryConfig, EntryEvaluator
from options_engine.strategy.risk_supervisor import RiskSupervisor

from .conftest import calm_regime

UTC = dt.UTC
NOW = dt.datetime(2024, 3, 1, 15, 0, tzinfo=UTC)


@pytest.fixture(scope="module")
def vrp_setup():
    """A richly-priced chain (35% vol) vs a much cheaper realized distribution (~14% vol).

    The large variance risk premium plus narrow (1-strike) wings ensures a positive-Kelly
    condor exists, so the full success path is reachable deterministically.
    """
    grid = TimeGrid.from_calendar_days(calendar_days=14, steps_per_day=8)
    rich_fv = ForwardVariance.flat(0.16)
    real_fv = ForwardVariance.flat(0.04)
    rich = RBergomiParams(hurst=0.1, eta=1.0, rho=-0.7, forward_variance=rich_fv)
    real = RBergomiParams(hurst=0.1, eta=1.0, rho=-0.7, forward_variance=real_fv)
    rich_paths = HybridSimulator(rich, rng_factory=RandomFactory(2), antithetic=True).simulate(
        grid=grid, n_paths=40_000, initial_spot=100.0
    )
    real_paths = HybridSimulator(real, rng_factory=RandomFactory(1), antithetic=True).simulate(
        grid=grid, n_paths=40_000, initial_spot=100.0
    )
    # A very tight maker keeps executable credit close to mid, isolating the variance-risk-
    # premium edge for this deterministic success-path test.
    maker = AvellanedaStoikovMaker(
        config=MarketMakerConfig(
            risk_aversion=0.05, order_flow_intensity=20.0, wing_spread_factor=0.5
        ),
        tick_size=0.05,
    )
    chain = build_synthetic_chain(
        build_terminal_distribution(rich_paths),
        maker=maker,
        strikes=np.arange(70.0, 131.0, 1.0),
        rate=0.0,
    )
    return {
        "real_dist": build_terminal_distribution(real_paths),
        "real_sample": real_paths.terminal_spot(),
        "chain": chain,
    }


def _evaluator(**risk_kw) -> EntryEvaluator:
    risk = RiskConfig(**risk_kw) if risk_kw else RiskConfig()
    return EntryEvaluator(
        risk=risk,
        risk_supervisor=RiskSupervisor(risk=risk),
        news_gate=NewsGate(
            news_provider=ReplayNewsProvider([]),
            event_provider=ReplayEventProvider([]),
            universe=frozenset({"SPX"}),
        ),
        config=EntryConfig(
            min_credit_to_cost_ratio=0.1,
            selection=CondorSelectionConfig(
                target_tail_probability=0.20,
                wing_width_fraction=0.02,
                min_win_probability=0.5,
                min_net_credit=0.01,
            ),
        ),
    )


def _evaluate(ev, vrp_setup, *, available_margin: float):
    return ev.evaluate(
        now=NOW,
        symbol="SPX",
        distribution=vrp_setup["real_dist"],
        chain=vrp_setup["chain"],
        regime=calm_regime(),
        account=Account.open(starting_cash=100_000.0),
        position_id="P1",
        multiplier=100.0,
        terminal_sample=vrp_setup["real_sample"],
        available_margin=available_margin,
    )


def test_full_entry_success(vrp_setup) -> None:
    decision = _evaluate(_evaluator(), vrp_setup, available_margin=50_000.0)
    assert decision.enter, decision.reason
    assert decision.position is not None
    assert decision.position.quantity >= 1
    # The stored condor's quantity matches the position quantity.
    assert decision.position.condor.quantity == decision.position.quantity
    assert decision.candidate is not None
    # Edge should be positive (variance risk premium).
    assert decision.candidate.expected_pnl > 0.0


def test_no_margin_blocks_entry(vrp_setup) -> None:
    # A genuine edge exists, but zero available margin yields no position (sizing/supervisor).
    decision = _evaluate(_evaluator(), vrp_setup, available_margin=0.0)
    assert decision.rejected
