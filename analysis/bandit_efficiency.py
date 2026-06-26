"""Measure PPO sample efficiency on a continuous bandit with a KNOWN optimum.

Reward = -||action - target||^2 ; the optimal deterministic action is exactly `target` and the
optimal reward is 0. This isolates the learning algorithm from the noisy trading environment,
so we can quantify *how fast* and *how reliably* the agent learns, and how that scales with the
amount of data per update (sample efficiency).
"""

from __future__ import annotations

import time

import numpy as np

from options_engine.agent.ppo import PPOAgent, PPOConfig
from options_engine.agent.rollout import RolloutBuffer

TARGET = np.array([0.7, -0.4], dtype=np.float32)
OBS = np.array([1.0], dtype=np.float32)


def run(*, rollout: int, iterations: int, lr: float, seed: int) -> dict[str, object]:
    agent = PPOAgent(
        obs_dim=1,
        action_dim=2,
        config=PPOConfig(seed=seed, n_epochs=8, minibatch_size=64, learning_rate=lr, entropy_coef=0.0),
    )
    rng = np.random.default_rng(seed)
    errors: list[float] = []
    t0 = time.time()
    samples_to_converge: int | None = None
    for it in range(iterations):
        buf = RolloutBuffer(capacity=rollout, obs_dim=1, action_dim=2, gamma=0.0, lam=0.0)
        while not buf.is_full:
            a, lp, v = agent.act(OBS)
            r = float(-np.sum((a - TARGET) ** 2))
            buf.add(obs=OBS, action=a.astype(np.float32), log_prob=lp, reward=r, value=v, next_value=0.0, done=True)
        agent.update(buf, rng=rng)
        learned, _, _ = agent.act(OBS, deterministic=True)
        err = float(np.linalg.norm(learned - TARGET))
        errors.append(err)
        if samples_to_converge is None and err < 0.05:
            samples_to_converge = (it + 1) * rollout
    learned, _, _ = agent.act(OBS, deterministic=True)
    return {
        "final_error": float(np.linalg.norm(learned - TARGET)),
        "errors": errors,
        "samples_to_converge": samples_to_converge,
        "wall_seconds": time.time() - t0,
        "learned": learned.tolist(),
    }


if __name__ == "__main__":
    print("PPO sample efficiency on a known-optimum continuous bandit (optimal error = 0)\n")
    print(f"{'rollout':>8} {'iters':>6} {'final_err':>10} {'samples->conv':>14} {'sec':>6}")
    for rollout in (64, 128, 256, 512):
        res = run(rollout=rollout, iterations=80, lr=3e-3, seed=0)
        conv = res["samples_to_converge"]
        conv_s = str(conv) if conv is not None else "no"
        print(
            f"{rollout:>8} {80:>6} {res['final_error']:>10.4f} {conv_s:>14} {res['wall_seconds']:>6.1f}"
        )

    # Reliability across seeds at a fixed budget.
    print("\nReliability across 8 seeds (rollout=256, 60 iters):")
    finals = []
    for seed in range(8):
        res = run(rollout=256, iterations=60, lr=3e-3, seed=seed)
        finals.append(res["final_error"])
    finals_arr = np.array(finals)
    print(f"  final error: mean={finals_arr.mean():.4f} max={finals_arr.max():.4f} "
          f"all<0.1: {bool(np.all(finals_arr < 0.1))}")

    # Learning curve sample (rollout=256).
    curve = run(rollout=256, iterations=60, lr=3e-3, seed=0)["errors"]
    marks = [0, 4, 9, 19, 29, 39, 59]
    print("\nLearning curve (error vs iteration), rollout=256:")
    for m in marks:
        bar = "#" * int(40 * (1 - min(curve[m], 1.0)))
        print(f"  iter {m:>2}: err={curve[m]:.4f} {bar}")
