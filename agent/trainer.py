r"""On-policy training loop with anti-collapse monitoring (SPEC §4.2, §4.4).

Drives the PPO agent against the trading environment: collect a rollout, update, repeat. The
loop is deterministic given the seed and records per-iteration diagnostics, including the
**anti-collapse monitors** the spec mandates -- it actively watches for the two degenerate
failure modes of a short-premium RL agent and the policy-entropy collapse that precedes them:

* **collapse-to-no-trade** -- the policy learns to always stay FLAT (trivially avoids the
  cost/risk penalties but earns nothing);
* **collapse-to-always-trade** -- the policy ignores the gates and trades every step;
* **entropy collapse** -- the action distribution becomes near-deterministic too early.

These are surfaced as explicit booleans/values on each :class:`TrainingIteration`, so a
training run (or a test) can assert the agent is exploring a healthy range of behaviour rather
than degenerating -- which is the difference between a model that *looks* trained and one that
actually learned.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.errors import ValidationError
from ..core.logging import get_logger
from ..rl.action import ACTION_DIM
from ..rl.env import IronCondorTradingEnv
from ..rl.observation import OBSERVATION_DIM
from .ppo import PPOAgent, PPOUpdateStats
from .rollout import RolloutBuffer

__all__ = ["CollapseMonitor", "Trainer", "TrainerConfig", "TrainingIteration"]

_logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class TrainerConfig:
    """Configuration for the training loop."""

    rollout_length: int = 1024
    n_iterations: int = 50
    gamma: float = 0.99
    gae_lambda: float = 0.95
    shaping_decay_iterations: int = 30  # iterations over which reward shaping decays to 0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.rollout_length < 2 or self.n_iterations < 1:
            raise ValidationError("rollout_length must be >= 2 and n_iterations >= 1", context={})
        if not 0.0 <= self.gamma <= 1.0 or not 0.0 <= self.gae_lambda <= 1.0:
            raise ValidationError("gamma and gae_lambda must lie in [0, 1]", context={})
        if self.shaping_decay_iterations < 1:
            raise ValidationError("shaping_decay_iterations must be >= 1", context={})


@dataclass(frozen=True, slots=True)
class CollapseMonitor:
    """Thresholds defining degenerate (collapsed) policy behaviour."""

    min_trade_fraction: float = 0.02  # below => collapse-to-no-trade
    max_trade_fraction: float = 0.98  # above => collapse-to-always-trade
    min_entropy: float = -3.0  # below => entropy collapse (near-deterministic policy)

    def evaluate(self, *, trade_fraction: float, entropy: float) -> tuple[bool, bool, bool]:
        """Return ``(no_trade_collapse, always_trade_collapse, entropy_collapse)`` flags."""
        no_trade = trade_fraction < self.min_trade_fraction
        always_trade = trade_fraction > self.max_trade_fraction
        entropy_collapsed = entropy < self.min_entropy
        return no_trade, always_trade, entropy_collapsed


@dataclass
class TrainingIteration:
    """Per-iteration training diagnostics, including anti-collapse signals."""

    iteration: int
    mean_reward: float
    trade_fraction: float
    update_stats: PPOUpdateStats
    shaping_coef: float
    no_trade_collapse: bool
    always_trade_collapse: bool
    entropy_collapse: bool

    @property
    def collapsed(self) -> bool:
        """True if any collapse mode is active this iteration."""
        return self.no_trade_collapse or self.always_trade_collapse or self.entropy_collapse


@dataclass
class TrainingHistory:
    """The full record of a training run."""

    iterations: list[TrainingIteration] = field(default_factory=list)

    def mean_rewards(self) -> list[float]:
        """Return the per-iteration mean rewards."""
        return [it.mean_reward for it in self.iterations]

    def any_collapse(self) -> bool:
        """True if any iteration showed a collapse mode."""
        return any(it.collapsed for it in self.iterations)


class Trainer:
    """Collects rollouts and trains a :class:`PPOAgent` with anti-collapse monitoring.

    Parameters
    ----------
    env:
        The trading environment.
    agent:
        The PPO agent (its dimensions must match the env spaces).
    config:
        Training-loop configuration.
    monitor:
        Collapse thresholds.
    """

    def __init__(
        self,
        *,
        env: IronCondorTradingEnv,
        agent: PPOAgent,
        config: TrainerConfig | None = None,
        monitor: CollapseMonitor | None = None,
    ) -> None:
        if env.observation_space.shape != (OBSERVATION_DIM,):
            raise ValidationError("env observation space mismatch", context={})
        if env.action_space.shape != (ACTION_DIM,):
            raise ValidationError("env action space mismatch", context={})
        self._env = env
        self._agent = agent
        self._config = config or TrainerConfig()
        self._monitor = monitor or CollapseMonitor()
        self._rng = np.random.default_rng(self._config.seed)
        self._buffer = RolloutBuffer(
            capacity=self._config.rollout_length,
            obs_dim=OBSERVATION_DIM,
            action_dim=ACTION_DIM,
            gamma=self._config.gamma,
            lam=self._config.gae_lambda,
        )

    def _shaping_coef(self, iteration: int) -> float:
        """Linearly decay the reward-shaping coefficient to zero (potential-based discipline)."""
        decay = self._config.shaping_decay_iterations
        return float(max(0.0, 1.0 - iteration / decay))

    def _collect_rollout(self) -> tuple[float, float]:
        """Fill the rollout buffer from the environment; return ``(mean_reward, trade_fraction)``.

        The environment is reset at the start and whenever an episode ends, so the buffer holds
        a continuous stream of on-policy transitions with correct episode boundaries.
        """
        self._buffer.reset()
        obs, _ = self._env.reset()
        total_reward = 0.0
        trade_steps = 0
        steps = 0

        while not self._buffer.is_full:
            action, log_prob, value = self._agent.act(obs)
            next_obs, reward, terminated, truncated, info = self._env.step(action)
            done = bool(terminated or truncated)
            # Bootstrap value: 0 on a true terminal, else the critic's estimate of next_obs.
            next_value = 0.0 if terminated else self._agent.value(next_obs)
            self._buffer.add(
                obs=obs.astype(np.float32),
                action=action.astype(np.float32),
                log_prob=log_prob,
                reward=reward,
                value=value,
                next_value=next_value,
                done=terminated,  # only true terminals stop the bootstrap chain
            )
            total_reward += reward
            trade_steps += int(info.get("traded", False))
            steps += 1
            if done:
                obs, _ = self._env.reset()
            else:
                obs = next_obs

        mean_reward = total_reward / max(1, steps)
        trade_fraction = trade_steps / max(1, steps)
        return mean_reward, trade_fraction

    def train(self) -> TrainingHistory:
        """Run the full training loop and return its history."""
        history = TrainingHistory()
        for iteration in range(self._config.n_iterations):
            shaping = self._shaping_coef(iteration)
            self._env.set_shaping_coef(shaping)

            mean_reward, trade_fraction = self._collect_rollout()
            update_stats = self._agent.update(self._buffer, rng=self._rng)

            no_trade, always_trade, entropy_collapse = self._monitor.evaluate(
                trade_fraction=trade_fraction, entropy=update_stats.entropy
            )
            record = TrainingIteration(
                iteration=iteration,
                mean_reward=mean_reward,
                trade_fraction=trade_fraction,
                update_stats=update_stats,
                shaping_coef=shaping,
                no_trade_collapse=no_trade,
                always_trade_collapse=always_trade,
                entropy_collapse=entropy_collapse,
            )
            history.iterations.append(record)
            _logger.info(
                "training_iteration",
                extra={
                    "iteration": iteration,
                    "mean_reward": mean_reward,
                    "trade_fraction": trade_fraction,
                    "entropy": update_stats.entropy,
                    "approx_kl": update_stats.approx_kl,
                    "collapsed": record.collapsed,
                },
            )
        return history
