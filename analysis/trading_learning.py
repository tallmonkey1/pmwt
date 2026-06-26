"""Measure how the RL agent learns on the REAL trading environment, vs. baselines.

We train the PPO agent and record per-iteration: mean reward, trade fraction, policy entropy,
KL, value loss, and the distributional critic's CVaR. We then compare the trained agent's
backtest performance against three baselines on held-out episodes:

  * random policy (samples the action space),
  * always-FLAT policy (never trades -> reward ~0; the "do nothing" floor),
  * a fixed aggressive theta-harvest rule.

This tells us whether learning actually improves decisions, not just whether loss goes down.
"""

from __future__ import annotations

import time

import numpy as np

from options_engine.agent.ppo import PPOAgent, PPOConfig
from options_engine.agent.trainer import Trainer, TrainerConfig
from options_engine.backtest.engine import BacktestConfig, run_backtest
from options_engine.rl.action import ACTION_DIM, ActionBounds
from options_engine.rl.env import IronCondorTradingEnv
from options_engine.rl.observation import OBSERVATION_DIM
from options_engine.rl.scenario import ScenarioConfig

SCENARIO = ScenarioConfig(
    n_steps=8, n_paths=3000, vrp_multiplier_range=(1.8, 2.8), realized_variance_range=(0.02, 0.05)
)
BOUNDS = ActionBounds(tail_probability=(0.08, 0.18), wing_width_fraction=(0.025, 0.05))


def make_env(seed: int) -> IronCondorTradingEnv:
    return IronCondorTradingEnv(seed=seed, scenario_config=SCENARIO, action_bounds=BOUNDS)


class RandomPolicy:
    def __init__(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def act(self, obs, *, deterministic: bool = False):  # noqa: ARG002
        a = self._rng.uniform(-1.0, 1.0, size=ACTION_DIM).astype(np.float32)
        return a, 0.0, 0.0


class FlatPolicy:
    def act(self, obs, *, deterministic: bool = True):  # noqa: ARG002
        return np.array([0.0, 0.0, 5.0, 0.0, 0.0, 0.0], dtype=np.float32), 0.0, 0.0


class ThetaRule:
    def act(self, obs, *, deterministic: bool = True):  # noqa: ARG002
        return np.array([2.0, 0.0, -1.0, -1.0, 0.0, 1.0], dtype=np.float32), 0.0, 0.0


def cumulative_reward(env, policy, *, steps: int, seed: int) -> float:
    res = run_backtest(env=env, policy=policy, config=BacktestConfig(n_steps=steps, n_trials=1), seed=seed)
    return float(np.sum(res.returns)) * 100.0  # scaled for readability


def main() -> None:
    print("Training PPO on the trading environment...\n")
    env = make_env(seed=0)
    agent = PPOAgent(
        obs_dim=OBSERVATION_DIM,
        action_dim=ACTION_DIM,
        config=PPOConfig(hidden_sizes=(64, 64), n_epochs=4, minibatch_size=64, seed=0),
    )
    trainer = Trainer(
        env=env, agent=agent, config=TrainerConfig(rollout_length=256, n_iterations=25, seed=0)
    )
    t0 = time.time()
    history = trainer.train()
    train_secs = time.time() - t0

    print(f"{'iter':>4} {'reward':>9} {'trade_frac':>10} {'entropy':>8} {'kl':>7} {'val_loss':>9}")
    for it in history.iterations:
        print(
            f"{it.iteration:>4} {it.mean_reward:>9.4f} {it.trade_fraction:>10.2f} "
            f"{it.update_stats.entropy:>8.3f} {it.update_stats.approx_kl:>7.4f} "
            f"{it.update_stats.value_loss:>9.4f}"
        )

    rewards = history.mean_rewards()
    first5 = float(np.mean(rewards[:5]))
    last5 = float(np.mean(rewards[-5:]))
    print(f"\nTrained {len(history.iterations)} iters in {train_secs:.0f}s "
          f"({trainer._buffer.capacity * len(history.iterations)} env steps).")  # noqa: SLF001
    print(f"Mean reward: first 5 iters = {first5:.4f}, last 5 iters = {last5:.4f}, "
          f"improvement = {last5 - first5:+.4f}")
    print("Any collapse flagged:", history.any_collapse())

    # Held-out evaluation vs baselines (fresh seeds the agent never trained on).
    print("\nHeld-out backtest (cumulative scaled P&L over 60 steps, mean of 5 fresh seeds):")
    eval_steps = 60
    eval_seeds = [101, 202, 303, 404, 505]
    policies = {
        "PPO (trained)": agent,
        "theta-rule": ThetaRule(),
        "random": RandomPolicy(seed=1),
        "always-FLAT": FlatPolicy(),
    }
    for name, pol in policies.items():
        vals = [cumulative_reward(make_env(seed=s), pol, steps=eval_steps, seed=s) for s in eval_seeds]
        print(f"  {name:>16}: mean={np.mean(vals):>9.3f}  std={np.std(vals):>7.3f}")


if __name__ == "__main__":
    main()
