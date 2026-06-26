"""Tests for the backtest engine."""

from __future__ import annotations

import numpy as np
import pytest

from options_engine.agent.ppo import PPOAgent, PPOConfig
from options_engine.backtest.engine import BacktestConfig, run_backtest
from options_engine.backtest.results import BacktestResult
from options_engine.core.errors import ValidationError
from options_engine.rl.action import ACTION_DIM, ActionBounds
from options_engine.rl.env import IronCondorTradingEnv
from options_engine.rl.observation import OBSERVATION_DIM
from options_engine.rl.scenario import ScenarioConfig


def _env() -> IronCondorTradingEnv:
    return IronCondorTradingEnv(
        seed=0,
        scenario_config=ScenarioConfig(
            n_steps=6,
            n_paths=2500,
            vrp_multiplier_range=(1.8, 2.8),
            realized_variance_range=(0.02, 0.05),
        ),
        action_bounds=ActionBounds(
            tail_probability=(0.08, 0.18), wing_width_fraction=(0.025, 0.05)
        ),
    )


def _agent() -> PPOAgent:
    return PPOAgent(
        obs_dim=OBSERVATION_DIM,
        action_dim=ACTION_DIM,
        config=PPOConfig(hidden_sizes=(32,), seed=0),
    )


class _FixedTradePolicy:
    """A deterministic rule policy that always proposes an aggressive theta-harvest trade."""

    def act(self, observation, *, deterministic: bool = True):
        # HARVEST_THETA, low tail (high win prob), full size -> trades when edge allows.
        action = np.array([2.0, 0.0, -1.0, -1.0, 0.0, 1.0], dtype=np.float32)
        return action, 0.0, 0.0


class TestRunBacktest:
    def test_produces_result(self) -> None:
        result = run_backtest(
            env=_env(), policy=_agent(), config=BacktestConfig(n_steps=20), seed=0
        )
        assert isinstance(result, BacktestResult)
        assert result.equity_curve.size == 21  # n_steps + initial
        assert result.returns.size == 20
        assert len(result.steps) == 20

    def test_rule_policy_trades(self) -> None:
        # A policy that proposes favourable trades should execute at least one.
        result = run_backtest(
            env=_env(), policy=_FixedTradePolicy(), config=BacktestConfig(n_steps=30), seed=0
        )
        assert result.n_trades >= 1
        # Trade records carry coherent economics.
        for trade in result.trades:
            assert trade.quantity >= 1
            assert np.isfinite(trade.pnl)

    def test_reproducible(self) -> None:
        a = run_backtest(
            env=_env(), policy=_FixedTradePolicy(), config=BacktestConfig(n_steps=20), seed=7
        )
        b = run_backtest(
            env=_env(), policy=_FixedTradePolicy(), config=BacktestConfig(n_steps=20), seed=7
        )
        np.testing.assert_array_equal(a.equity_curve, b.equity_curve)

    def test_rejects_non_policy(self) -> None:
        with pytest.raises(ValidationError):
            run_backtest(env=_env(), policy=object(), config=BacktestConfig(n_steps=10))  # type: ignore[arg-type]

    def test_config_validation(self) -> None:
        with pytest.raises(ValidationError):
            BacktestConfig(n_steps=1)
        with pytest.raises(ValidationError):
            BacktestConfig(n_trials=0)
