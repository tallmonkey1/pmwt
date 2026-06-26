"""Tests for the PPO agent, including a learning-convergence proof on a known-optimum task."""

from __future__ import annotations

import numpy as np
import pytest

from options_engine.agent.ppo import PPOAgent, PPOConfig
from options_engine.agent.rollout import RolloutBuffer
from options_engine.core.errors import ValidationError


class TestPPOAgentBasics:
    def test_act_shapes(self) -> None:
        agent = PPOAgent(obs_dim=4, action_dim=2, config=PPOConfig(hidden_sizes=(16,), seed=0))
        obs = np.zeros(4, dtype=np.float32)
        action, log_prob, value = agent.act(obs)
        assert action.shape == (2,)
        assert np.isfinite(log_prob)
        assert np.isfinite(value)

    def test_deterministic_act_is_mean(self) -> None:
        agent = PPOAgent(obs_dim=4, action_dim=2, config=PPOConfig(hidden_sizes=(16,), seed=0))
        obs = np.ones(4, dtype=np.float32)
        a1, _, _ = agent.act(obs, deterministic=True)
        a2, _, _ = agent.act(obs, deterministic=True)
        np.testing.assert_array_equal(a1, a2)  # deterministic => identical

    def test_cvar_not_above_value(self) -> None:
        agent = PPOAgent(obs_dim=4, action_dim=2, config=PPOConfig(hidden_sizes=(16,), seed=0))
        obs = np.ones(4, dtype=np.float32)
        assert agent.cvar(obs) <= agent.value(obs) + 1e-6

    def test_state_dict_round_trip(self) -> None:
        agent = PPOAgent(obs_dim=4, action_dim=2, config=PPOConfig(hidden_sizes=(16,), seed=0))
        obs = np.ones(4, dtype=np.float32)
        before, _, _ = agent.act(obs, deterministic=True)
        state = agent.state_dict()
        restored = PPOAgent(obs_dim=4, action_dim=2, config=PPOConfig(hidden_sizes=(16,), seed=1))
        restored.load_state_dict(state)
        after, _, _ = restored.act(obs, deterministic=True)
        np.testing.assert_allclose(before, after, rtol=1e-5, atol=1e-6)

    def test_update_returns_stats(self) -> None:
        agent = PPOAgent(
            obs_dim=2, action_dim=1, config=PPOConfig(hidden_sizes=(16,), n_epochs=2, seed=0)
        )
        buf = RolloutBuffer(capacity=64, obs_dim=2, action_dim=1, gamma=0.99, lam=0.95)
        obs = np.zeros(2, dtype=np.float32)
        while not buf.is_full:
            a, lp, v = agent.act(obs)
            buf.add(
                obs=obs,
                action=a.astype(np.float32),
                log_prob=lp,
                reward=0.1,
                value=v,
                next_value=0.0,
                done=True,
            )
        stats = agent.update(buf, rng=np.random.default_rng(0))
        assert stats.epochs_run >= 1
        assert np.isfinite(stats.policy_loss)
        assert np.isfinite(stats.value_loss)
        assert np.isfinite(stats.approx_kl)

    def test_rejects_bad_config(self) -> None:
        with pytest.raises(ValidationError):
            PPOConfig(clip_epsilon=1.5)


@pytest.mark.slow
class TestPPOLearning:
    def test_learns_continuous_bandit(self) -> None:
        # A one-step continuous bandit with a known optimum: reward = -||action - target||^2.
        # PPO must drive the policy mean to the target -- the core "does it actually learn"
        # proof for the brain.
        target = np.array([0.7, -0.4], dtype=np.float32)
        agent = PPOAgent(
            obs_dim=1,
            action_dim=2,
            config=PPOConfig(
                seed=0, n_epochs=8, minibatch_size=64, learning_rate=3e-3, entropy_coef=0.0
            ),
        )
        rng = np.random.default_rng(0)
        obs = np.array([1.0], dtype=np.float32)

        for _ in range(60):
            buf = RolloutBuffer(capacity=256, obs_dim=1, action_dim=2, gamma=0.0, lam=0.0)
            while not buf.is_full:
                action, log_prob, value = agent.act(obs)
                reward = float(-np.sum((action - target) ** 2))
                buf.add(
                    obs=obs,
                    action=action.astype(np.float32),
                    log_prob=log_prob,
                    reward=reward,
                    value=value,
                    next_value=0.0,
                    done=True,
                )
            agent.update(buf, rng=rng)

        learned, _, _ = agent.act(obs, deterministic=True)
        assert np.all(np.abs(learned - target) < 0.1)
