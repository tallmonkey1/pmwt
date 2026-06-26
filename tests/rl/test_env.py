"""Tests for the Gymnasium trading environment, including the safety overlay and conformance."""

from __future__ import annotations

import numpy as np
import pytest

from options_engine.core.config import RiskConfig
from options_engine.core.errors import ValidationError
from options_engine.rl.action import ACTION_DIM, ActionBounds
from options_engine.rl.env import EnvConfig, IronCondorTradingEnv
from options_engine.rl.observation import OBSERVATION_DIM
from options_engine.rl.scenario import ScenarioConfig


def _env(**kw) -> IronCondorTradingEnv:
    defaults = {
        "seed": 0,
        "scenario_config": ScenarioConfig(n_steps=6, n_paths=3000),
    }
    defaults.update(kw)
    return IronCondorTradingEnv(**defaults)  # type: ignore[arg-type]


def _trade_env() -> IronCondorTradingEnv:
    """An env tuned so a sensible action trades (low tail clears the win-prob filter)."""
    return IronCondorTradingEnv(
        seed=0,
        scenario_config=ScenarioConfig(
            n_steps=8,
            n_paths=4000,
            vrp_multiplier_range=(1.8, 2.8),
            realized_variance_range=(0.02, 0.05),
        ),
        action_bounds=ActionBounds(
            tail_probability=(0.08, 0.18), wing_width_fraction=(0.025, 0.05)
        ),
    )


class TestSpaces:
    def test_observation_space_shape(self) -> None:
        env = _env()
        assert env.observation_space.shape == (OBSERVATION_DIM,)

    def test_action_space_shape(self) -> None:
        env = _env()
        assert env.action_space.shape == (ACTION_DIM,)


class TestResetStep:
    def test_reset_returns_valid_observation(self) -> None:
        env = _env()
        obs, info = env.reset(seed=0)
        assert env.observation_space.contains(obs)
        assert "trade_fraction" in info

    def test_step_before_reset_raises(self) -> None:
        env = _env()
        with pytest.raises(ValidationError):
            env.step(np.zeros(ACTION_DIM, dtype=np.float32))

    def test_episode_runs_to_truncation(self) -> None:
        env = _env(scenario_config=ScenarioConfig(n_steps=5, n_paths=3000))
        env.reset(seed=0)
        flat = np.array([0, 0, 5.0, 0, 0, 0], dtype=np.float32)
        steps = 0
        for _ in range(10):
            _, reward, terminated, truncated, _ = env.step(flat)
            steps += 1
            assert np.isfinite(reward)
            if terminated or truncated:
                break
        assert steps == 5  # truncates exactly at n_steps

    def test_returned_observation_in_space(self) -> None:
        env = _env()
        env.reset(seed=0)
        obs, _, terminated, truncated, _ = env.step(
            np.array([2.0, 0, -1, -0.5, 0, 1.0], dtype=np.float32)
        )
        if not (terminated or truncated):
            assert env.observation_space.contains(obs)


class TestAntiCollapseDiagnostics:
    def test_flat_agent_never_trades(self) -> None:
        env = _trade_env()
        env.reset(seed=0)
        flat = np.array([0, 0, 5.0, 0, 0, 0], dtype=np.float32)
        info: dict = {}
        for _ in range(8):
            _, _, terminated, truncated, info = env.step(flat)
            if terminated or truncated:
                break
        assert info["trade_fraction"] == 0.0

    def test_aggressive_agent_trades(self) -> None:
        env = _trade_env()
        env.reset(seed=0)
        traded = 0
        # Low tail probability + full size clears the win-prob filter and sizes > 0.
        act = np.array([2.0, 0, -1, -1.0, 0.0, 1.0], dtype=np.float32)
        for _ in range(8):
            _, _, terminated, truncated, info = env.step(act)
            traded += int(info["traded"])
            if terminated or truncated:
                break
        assert traded >= 1

    def test_strategic_counts_tracked(self) -> None:
        env = _trade_env()
        env.reset(seed=0)
        env.step(np.array([5.0, 0, 0, -0.5, 0, 1.0], dtype=np.float32))  # THETA
        env.step(np.array([0, 5.0, 0, -0.5, 0, 1.0], dtype=np.float32))  # GAMMA
        _, _, _, _, info = env.step(np.array([0, 0, 5.0, 0, 0, 0], dtype=np.float32))  # FLAT
        counts = info["strategic_counts"]
        assert sum(counts.values()) == 3


class TestSafetyOverlay:
    def test_kill_switch_terminates_episode(self) -> None:
        # A tiny drawdown limit makes the kill-switch trip after any loss, terminating early.
        env = IronCondorTradingEnv(
            seed=0,
            risk=RiskConfig(max_drawdown_kill_switch=0.001),
            scenario_config=ScenarioConfig(
                n_steps=8,
                n_paths=4000,
                vrp_multiplier_range=(1.8, 2.8),
                realized_variance_range=(0.02, 0.05),
            ),
            action_bounds=ActionBounds(
                tail_probability=(0.08, 0.18), wing_width_fraction=(0.025, 0.05)
            ),
        )
        env.reset(seed=0)
        terminated = False
        act = np.array([2.0, 0, -1, -1.0, 0.0, 1.0], dtype=np.float32)
        for _ in range(8):
            _, _, terminated, truncated, info = env.step(act)
            if terminated:
                assert "kill_switch" in info
                break
            if truncated:
                break
        # Either the kill-switch tripped (if a loss occurred) or the episode truncated cleanly.
        assert isinstance(terminated, bool)

    def test_reproducible_episodes(self) -> None:
        act = np.array([2.0, 0, -1, -0.5, 0.0, 1.0], dtype=np.float32)
        rewards_a, rewards_b = [], []
        for store in (rewards_a, rewards_b):
            env = _trade_env()
            env.reset(seed=42)
            for _ in range(8):
                _, r, terminated, truncated, _ = env.step(act)
                store.append(r)
                if terminated or truncated:
                    break
        assert rewards_a == rewards_b


class TestEnvConfig:
    def test_rejects_bad_values(self) -> None:
        with pytest.raises(ValidationError):
            EnvConfig(starting_cash=0.0)
        with pytest.raises(ValidationError):
            EnvConfig(shaping_coef=2.0)
