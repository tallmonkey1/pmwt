"""Show that retuning reward weights makes the agent learn to TRADE profitably.

The default reward over-penalizes risk/cost relative to the (positive) P&L edge, so the agent
rationally learns to stay flat. Here we lower the risk/cost weights so the genuine edge
survives, and confirm the agent then learns to trade and beats the do-nothing floor.
"""

from __future__ import annotations

import numpy as np

from options_engine.agent.ppo import PPOAgent, PPOConfig
from options_engine.agent.trainer import Trainer, TrainerConfig
from options_engine.backtest.engine import BacktestConfig, run_backtest
from options_engine.rl.action import ACTION_DIM, ActionBounds
from options_engine.rl.env import IronCondorTradingEnv
from options_engine.rl.observation import OBSERVATION_DIM
from options_engine.rl.reward import RewardConfig
from options_engine.rl.scenario import ScenarioConfig

SCENARIO = ScenarioConfig(
    n_steps=8, n_paths=3000, vrp_multiplier_range=(1.8, 2.8), realized_variance_range=(0.02, 0.05)
)
BOUNDS = ActionBounds(tail_probability=(0.08, 0.18), wing_width_fraction=(0.025, 0.05))
# Reward retuned so the positive P&L edge is not swamped by risk/cost penalties.
RETUNED = RewardConfig(risk_weight=0.10, cost_weight=0.25, margin_weight=0.05)


def make_env(seed: int) -> IronCondorTradingEnv:
    return IronCondorTradingEnv(
        seed=seed, scenario_config=SCENARIO, action_bounds=BOUNDS, reward_config=RETUNED
    )


def cum_pnl(env, policy, *, steps: int, seed: int) -> float:
    res = run_backtest(env=env, policy=policy, config=BacktestConfig(n_steps=steps, n_trials=1), seed=seed)
    return float(np.sum(res.returns)) * 100.0


class FlatPolicy:
    def act(self, obs, *, deterministic: bool = True):  # noqa: ARG002
        return np.array([0.0, 0.0, 5.0, 0.0, 0.0, 0.0], dtype=np.float32), 0.0, 0.0


def main() -> None:
    env = make_env(seed=0)
    agent = PPOAgent(
        obs_dim=OBSERVATION_DIM, action_dim=ACTION_DIM,
        config=PPOConfig(hidden_sizes=(64, 64), n_epochs=4, minibatch_size=64, seed=0),
    )
    trainer = Trainer(env=env, agent=agent, config=TrainerConfig(rollout_length=256, n_iterations=25, seed=0))
    history = trainer.train()

    print("Retuned reward (risk_weight=0.10, cost_weight=0.25):\n")
    print(f"{'iter':>4} {'reward':>9} {'trade_frac':>10} {'entropy':>8}")
    for it in history.iterations[::3]:
        print(f"{it.iteration:>4} {it.mean_reward:>9.4f} {it.trade_fraction:>10.2f} {it.update_stats.entropy:>8.3f}")
    rewards = history.mean_rewards()
    print(f"\nMean reward: first 5 = {np.mean(rewards[:5]):.4f}, last 5 = {np.mean(rewards[-5:]):.4f}")
    print(f"Final trade fraction: {history.iterations[-1].trade_fraction:.2f}")

    eval_seeds = [101, 202, 303, 404, 505]
    ppo = [cum_pnl(make_env(s), agent, steps=60, seed=s) for s in eval_seeds]
    flat = [cum_pnl(make_env(s), FlatPolicy(), steps=60, seed=s) for s in eval_seeds]
    print("\nHeld-out cumulative scaled P&L (mean of 5 fresh seeds):")
    print(f"  PPO (trained): {np.mean(ppo):.3f} (std {np.std(ppo):.3f})")
    print(f"  always-FLAT  : {np.mean(flat):.3f}")
    print(f"  PPO beats do-nothing: {np.mean(ppo) > np.mean(flat) + 1e-9}")


if __name__ == "__main__":
    main()
