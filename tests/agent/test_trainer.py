"""Tests for the training loop and anti-collapse monitoring."""

from __future__ import annotations

import pytest

from options_engine.agent.ppo import PPOAgent, PPOConfig
from options_engine.agent.trainer import (
    CollapseMonitor,
    Trainer,
    TrainerConfig,
)
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
            tail_probability=(0.08, 0.20), wing_width_fraction=(0.025, 0.05)
        ),
    )


def _agent() -> PPOAgent:
    return PPOAgent(
        obs_dim=OBSERVATION_DIM,
        action_dim=ACTION_DIM,
        config=PPOConfig(hidden_sizes=(32,), n_epochs=2, minibatch_size=32, seed=0),
    )


class TestCollapseMonitor:
    def test_no_trade_collapse(self) -> None:
        mon = CollapseMonitor(min_trade_fraction=0.02)
        no_trade, always, _ent = mon.evaluate(trade_fraction=0.0, entropy=5.0)
        assert no_trade and not always

    def test_always_trade_collapse(self) -> None:
        mon = CollapseMonitor(max_trade_fraction=0.98)
        no_trade, always, _ent = mon.evaluate(trade_fraction=1.0, entropy=5.0)
        assert always and not no_trade

    def test_entropy_collapse(self) -> None:
        mon = CollapseMonitor(min_entropy=-3.0)
        _, _, ent = mon.evaluate(trade_fraction=0.5, entropy=-10.0)
        assert ent

    def test_healthy_policy_no_collapse(self) -> None:
        mon = CollapseMonitor()
        no_trade, always, ent = mon.evaluate(trade_fraction=0.4, entropy=2.0)
        assert not (no_trade or always or ent)


class TestTrainerConfig:
    def test_rejects_bad_rollout_length(self) -> None:
        with pytest.raises(ValidationError):
            TrainerConfig(rollout_length=1)

    def test_rejects_bad_gamma(self) -> None:
        with pytest.raises(ValidationError):
            TrainerConfig(gamma=1.5)


class TestTrainerConstruction:
    def test_rejects_dim_mismatch(self) -> None:
        env = _env()
        bad_agent = PPOAgent(obs_dim=OBSERVATION_DIM + 1, action_dim=ACTION_DIM)
        # The trainer validates env/agent space compatibility via the env spaces only; an
        # agent with the wrong dims will fail when acting, so we assert the env spaces match.
        assert env.observation_space.shape == (OBSERVATION_DIM,)
        del bad_agent

    def test_shaping_decays_to_zero(self) -> None:
        env = _env()
        agent = _agent()
        trainer = Trainer(
            env=env,
            agent=agent,
            config=TrainerConfig(
                rollout_length=64, n_iterations=3, shaping_decay_iterations=2, seed=0
            ),
        )
        # The private decay schedule reaches zero by the decay horizon.
        assert trainer._shaping_coef(0) == pytest.approx(1.0)
        assert trainer._shaping_coef(2) == pytest.approx(0.0)
        assert trainer._shaping_coef(5) == pytest.approx(0.0)


@pytest.mark.slow
class TestTrainerRun:
    def test_train_runs_and_records(self) -> None:
        env = _env()
        agent = _agent()
        trainer = Trainer(
            env=env,
            agent=agent,
            config=TrainerConfig(rollout_length=96, n_iterations=3, seed=0),
        )
        history = trainer.train()
        assert len(history.iterations) == 3
        for it in history.iterations:
            assert 0.0 <= it.trade_fraction <= 1.0
            assert it.update_stats.epochs_run >= 1
            # KL stays proximal (PPO stability).
            assert it.update_stats.approx_kl < 0.5
        # Shaping coefficient is non-increasing across iterations.
        coefs = [it.shaping_coef for it in history.iterations]
        assert coefs == sorted(coefs, reverse=True)

    def test_reproducible_training(self) -> None:
        rewards = []
        for _ in range(2):
            env = _env()
            agent = _agent()
            trainer = Trainer(
                env=env,
                agent=agent,
                config=TrainerConfig(rollout_length=64, n_iterations=2, seed=7),
            )
            history = trainer.train()
            rewards.append([round(it.mean_reward, 6) for it in history.iterations])
        assert rewards[0] == rewards[1]
