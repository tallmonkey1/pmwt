"""Tests for the growth-optimal reward, proving its anti-collapse properties by arithmetic.

These tests verify -- without training any model -- that the reward's optimum is "trade the
edge", not "do nothing": the free, risk-free zero at FLAT is removed exactly when (and only
when) qualifying edge exists, and a positive-edge trade strictly beats declining that edge.
"""

from __future__ import annotations

import math

import pytest

from options_engine.core.errors import ValidationError
from options_engine.rl.growth_reward import (
    GrowthRewardConfig,
    GrowthRewardInputs,
    compute_growth_reward,
)


def _inputs(**kw) -> GrowthRewardInputs:
    defaults = {
        "pnl_change": 0.0,
        "equity": 100_000.0,
        "traded": False,
        "chosen_expected_edge": 0.0,
        "best_available_edge": 0.0,
        "incremental_cvar": 0.0,
        "limit_breached": False,
    }
    defaults.update(kw)
    return GrowthRewardInputs(**defaults)  # type: ignore[arg-type]


class TestKellyCore:
    def test_flat_no_edge_is_zero(self) -> None:
        # Property 1: sitting out a market with no qualifying edge is free (reward == 0).
        rb = compute_growth_reward(_inputs(traded=False, best_available_edge=0.0))
        assert rb.total == pytest.approx(0.0)

    def test_growth_is_log_wealth(self) -> None:
        # The core term is exactly log(1 + pnl/equity).
        rb = compute_growth_reward(_inputs(pnl_change=1_000.0, equity=100_000.0, traded=True))
        assert rb.growth_term == pytest.approx(math.log(1.0 + 0.01))

    def test_loss_punished_concavely(self) -> None:
        # A loss hurts more than the symmetric gain helps (concave log => built-in risk aversion).
        gain = compute_growth_reward(
            _inputs(pnl_change=5_000.0, equity=100_000.0, traded=True)
        ).growth_term
        loss = compute_growth_reward(
            _inputs(pnl_change=-5_000.0, equity=100_000.0, traded=True)
        ).growth_term
        assert abs(loss) > abs(gain)

    def test_total_loss_is_finite(self) -> None:
        # Even a clamped total loss keeps the log finite (no -inf reward).
        rb = compute_growth_reward(_inputs(pnl_change=-100_000.0, equity=100_000.0, traded=True))
        assert math.isfinite(rb.total)


class TestAntiCollapse:
    def test_flat_with_edge_is_negative(self) -> None:
        # Property 2: declining QUALIFYING positive edge costs something (no free zero).
        rb = compute_growth_reward(_inputs(traded=False, best_available_edge=300.0))
        assert rb.total < 0.0
        assert rb.opportunity_term < 0.0

    def test_opportunity_is_self_gating(self) -> None:
        # No available edge (bad market) => no opportunity penalty => FLAT stays free.
        rb = compute_growth_reward(_inputs(traded=False, best_available_edge=0.0))
        assert rb.opportunity_term == 0.0
        assert rb.total == pytest.approx(0.0)

    def test_positive_edge_trade_beats_declining_it(self) -> None:
        # Property 5 (the calibration invariant): taking a positive-edge condor that realizes
        # its expectation yields strictly higher reward than declining the same edge.
        edge = 400.0  # expected risk-adjusted P&L of the available condor
        equity = 100_000.0
        cfg = GrowthRewardConfig()

        traded = compute_growth_reward(
            _inputs(
                traded=True,
                pnl_change=edge,  # realizes its expectation
                chosen_expected_edge=edge,
                equity=equity,
            ),
            config=cfg,
            shaping_coef=1.0,
        )
        declined = compute_growth_reward(
            _inputs(traded=False, best_available_edge=edge, equity=equity),
            config=cfg,
            shaping_coef=1.0,
        )
        assert traded.total > declined.total

    def test_positive_edge_trade_beats_declining_even_without_shaping(self) -> None:
        # The invariant must hold once shaping has fully decayed (shaping_coef = 0), proving
        # the FINAL policy -- not just the shaped one -- prefers trading real edge.
        edge = 400.0
        equity = 100_000.0
        traded = compute_growth_reward(
            _inputs(traded=True, pnl_change=edge, chosen_expected_edge=edge, equity=equity),
            shaping_coef=0.0,
        )
        declined = compute_growth_reward(
            _inputs(traded=False, best_available_edge=edge, equity=equity),
            shaping_coef=0.0,
        )
        assert traded.total > declined.total


class TestShapingAndPenalties:
    def test_shaping_decays_to_zero(self) -> None:
        full = compute_growth_reward(
            _inputs(traded=True, chosen_expected_edge=500.0, pnl_change=0.0), shaping_coef=1.0
        )
        none = compute_growth_reward(
            _inputs(traded=True, chosen_expected_edge=500.0, pnl_change=0.0), shaping_coef=0.0
        )
        assert full.edge_shaping_term > 0.0
        assert none.edge_shaping_term == 0.0

    def test_only_added_risk_penalized(self) -> None:
        reduced = compute_growth_reward(_inputs(traded=True, incremental_cvar=-1_000.0))
        assert reduced.risk_term == 0.0

    def test_added_risk_penalized(self) -> None:
        rb = compute_growth_reward(
            _inputs(traded=True, incremental_cvar=1_000.0, equity=100_000.0),
            config=GrowthRewardConfig(risk_weight=0.1),
        )
        assert rb.risk_term == pytest.approx(-0.1 * 1_000.0 / 100_000.0)

    def test_breach_penalty_dominates(self) -> None:
        clean = compute_growth_reward(_inputs(traded=True, pnl_change=1000.0))
        breached = compute_growth_reward(
            _inputs(traded=True, pnl_change=1000.0, limit_breached=True), limit_breach_penalty=5.0
        )
        assert breached.breach_term == pytest.approx(-5.0)
        assert breached.total < clean.total

    def test_total_is_sum_of_components(self) -> None:
        rb = compute_growth_reward(
            _inputs(
                traded=True,
                pnl_change=200.0,
                chosen_expected_edge=300.0,
                incremental_cvar=500.0,
                equity=100_000.0,
            )
        )
        components = (
            rb.growth_term
            + rb.opportunity_term
            + rb.edge_shaping_term
            + rb.risk_term
            + rb.breach_term
        )
        assert rb.total == pytest.approx(components)


class TestValidation:
    def test_rejects_nonpositive_equity(self) -> None:
        with pytest.raises(ValidationError):
            GrowthRewardInputs(pnl_change=0.0, equity=0.0, traded=False)

    def test_rejects_bad_shaping_coef(self) -> None:
        with pytest.raises(ValidationError):
            compute_growth_reward(_inputs(), shaping_coef=1.5)

    def test_config_rejects_negative_weight(self) -> None:
        with pytest.raises(ValidationError):
            GrowthRewardConfig(opportunity_weight=-1.0)
