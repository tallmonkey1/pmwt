"""Tests for the RL reward function."""

from __future__ import annotations

import pytest

from options_engine.core.errors import ValidationError
from options_engine.rl.reward import (
    RewardConfig,
    RewardInputs,
    compute_reward,
)


def _inputs(**kw) -> RewardInputs:
    defaults = {
        "pnl_change": 0.0,
        "incremental_cvar": 0.0,
        "transaction_cost": 0.0,
        "margin_utilization": 0.0,
        "theta_captured": 0.0,
        "limit_breached": False,
    }
    defaults.update(kw)
    return RewardInputs(**defaults)  # type: ignore[arg-type]


class TestComputeReward:
    def test_pnl_dominates_positive(self) -> None:
        rb = compute_reward(_inputs(pnl_change=1000.0), config=RewardConfig(reference_risk=1000.0))
        assert rb.pnl_term == pytest.approx(1.0)
        assert rb.total > 0.0

    def test_transaction_cost_penalized(self) -> None:
        rb = compute_reward(
            _inputs(transaction_cost=500.0),
            config=RewardConfig(reference_risk=1000.0, cost_weight=1.0),
        )
        assert rb.cost_term == pytest.approx(-0.5)
        assert rb.total < 0.0

    def test_only_added_risk_penalized(self) -> None:
        # Risk reduction (negative incremental CVaR) is not rewarded as free P&L.
        reduced = compute_reward(_inputs(incremental_cvar=-1000.0))
        assert reduced.risk_term == 0.0

    def test_added_risk_penalized(self) -> None:
        rb = compute_reward(
            _inputs(incremental_cvar=1000.0),
            config=RewardConfig(reference_risk=1000.0, risk_weight=1.0),
        )
        assert rb.risk_term == pytest.approx(-1.0)

    def test_limit_breach_penalty(self) -> None:
        clean = compute_reward(_inputs())
        breached = compute_reward(_inputs(limit_breached=True), limit_breach_penalty=5.0)
        assert breached.tail_term == pytest.approx(-5.0)
        assert breached.total < clean.total

    def test_shaping_decays_to_zero(self) -> None:
        full = compute_reward(_inputs(theta_captured=1000.0), shaping_coef=1.0)
        none = compute_reward(_inputs(theta_captured=1000.0), shaping_coef=0.0)
        assert full.shaping_term > 0.0
        assert none.shaping_term == 0.0

    def test_total_is_sum_of_components(self) -> None:
        rb = compute_reward(
            _inputs(
                pnl_change=100.0,
                incremental_cvar=50.0,
                transaction_cost=10.0,
                margin_utilization=0.5,
                theta_captured=80.0,
                limit_breached=True,
            )
        )
        components = (
            rb.pnl_term
            + rb.risk_term
            + rb.cost_term
            + rb.margin_term
            + rb.tail_term
            + rb.shaping_term
        )
        assert rb.total == pytest.approx(components)

    def test_rejects_bad_shaping_coef(self) -> None:
        with pytest.raises(ValidationError):
            compute_reward(_inputs(), shaping_coef=1.5)

    def test_rejects_negative_cost(self) -> None:
        with pytest.raises(ValidationError):
            RewardInputs(
                pnl_change=0.0,
                incremental_cvar=0.0,
                transaction_cost=-1.0,
                margin_utilization=0.0,
                theta_captured=0.0,
                limit_breached=False,
            )


class TestRewardConfig:
    def test_rejects_negative_weight(self) -> None:
        with pytest.raises(ValidationError):
            RewardConfig(risk_weight=-1.0)

    def test_rejects_nonpositive_reference_risk(self) -> None:
        with pytest.raises(ValidationError):
            RewardConfig(reference_risk=0.0)
