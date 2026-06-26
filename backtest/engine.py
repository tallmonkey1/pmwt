r"""Backtest engine: runs a policy through the trading environment (SPEC §9).

The engine drives any policy that maps an observation to an action through the
:class:`~options_engine.rl.env.IronCondorTradingEnv`, recording the equity curve, per-step
audit log, and trade ledger, then computing the performance metrics. The *same* engine runs
the history backtest and the paper (live-replay) backtest -- the only difference is which data
the environment is configured with, which is the single-source-of-truth design that
guarantees backtest <-> live parity (SPEC §8).

A policy is anything with ``act(observation) -> action`` (the :class:`PolicyProtocol`); the
trained :class:`~options_engine.agent.ppo.PPOAgent` satisfies it, as do simple rule policies
used for baselines and testing. Episodes are run back-to-back until the configured number of
steps is reached or the kill-switch halts trading.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from ..core.errors import ValidationError
from ..core.logging import get_logger
from ..core.validation import check_positive
from ..rl.env import IronCondorTradingEnv
from .metrics import compute_performance_metrics
from .results import BacktestResult, StepRecord, TradeRecord

__all__ = ["BacktestConfig", "PolicyProtocol", "run_backtest"]

_logger = get_logger(__name__)


@runtime_checkable
class PolicyProtocol(Protocol):
    """Minimal policy interface the engine drives."""

    def act(
        self, observation: NDArray[np.float32], *, deterministic: bool = ...
    ) -> tuple[NDArray[np.float32], float, float]:
        """Return ``(action, log_prob, value)`` for an observation."""
        ...


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    """Configuration for a backtest run."""

    n_steps: int = 500
    deterministic_policy: bool = True  # evaluate the policy's mean action (no exploration)
    periods_per_year: int = 252
    n_trials: int = 1  # multiple-testing count for the deflated Sharpe

    def __post_init__(self) -> None:
        if self.n_steps < 2:
            raise ValidationError("n_steps must be >= 2", context={"n_steps": self.n_steps})
        check_positive(self.periods_per_year, name="periods_per_year")
        if self.n_trials < 1:
            raise ValidationError("n_trials must be >= 1", context={"n_trials": self.n_trials})


def run_backtest(
    *,
    env: IronCondorTradingEnv,
    policy: PolicyProtocol,
    config: BacktestConfig | None = None,
    seed: int = 0,
) -> BacktestResult:
    """Run ``policy`` through ``env`` for ``config.n_steps`` steps and return the result.

    Parameters
    ----------
    env:
        The trading environment (configured for history or paper data).
    policy:
        The decision policy (RL agent or rule).
    config:
        Backtest configuration.
    seed:
        Reset seed for reproducibility.
    """
    cfg = config or BacktestConfig()
    if not isinstance(policy, PolicyProtocol):
        raise ValidationError("policy must implement act(observation)", context={})

    obs, _ = env.reset(seed=seed)
    starting_equity = float(env.account_equity())
    equity_curve: list[float] = [starting_equity]
    returns: list[float] = []
    steps: list[StepRecord] = []
    trades: list[TradeRecord] = []
    kill_switch = False

    for step in range(cfg.n_steps):
        action, _log_prob, _value = policy.act(obs, deterministic=cfg.deterministic_policy)
        next_obs, reward, terminated, truncated, info = env.step(action)

        equity = float(info["equity"])
        prev_equity = equity_curve[-1]
        period_return = (equity - prev_equity) / prev_equity if prev_equity > 0 else 0.0
        equity_curve.append(equity)
        returns.append(period_return)

        steps.append(
            StepRecord(
                step=step,
                equity=equity,
                reward=float(reward),
                traded=bool(info.get("traded", False)),
                limit_breached=bool(info.get("limit_breached", False)),
                strategic_action=_dominant_strategic(info),
                regime_low_prob=float(info.get("regime_low_prob", 0.0)),
            )
        )
        trade = _extract_trade(env, step)
        if trade is not None:
            trades.append(trade)

        if terminated and "kill_switch" in info:
            kill_switch = True

        if terminated or truncated:
            obs, _ = env.reset()
        else:
            obs = next_obs

    ending_equity = equity_curve[-1]
    returns_arr = np.asarray(returns, dtype=np.float64)
    equity_arr = np.asarray(equity_curve, dtype=np.float64)

    metrics = compute_performance_metrics(
        returns=returns_arr,
        equity_curve=equity_arr,
        trade_pnls=np.asarray([t.pnl for t in trades], dtype=np.float64) if trades else None,
        periods_per_year=cfg.periods_per_year,
    )

    _logger.info(
        "backtest_complete",
        extra={
            "n_steps": cfg.n_steps,
            "n_trades": len(trades),
            "ending_equity": ending_equity,
            "sharpe": metrics.sharpe,
            "max_drawdown": metrics.max_drawdown,
            "kill_switch": kill_switch,
        },
    )

    return BacktestResult(
        equity_curve=equity_arr,
        returns=returns_arr,
        steps=tuple(steps),
        trades=tuple(trades),
        metrics=metrics,
        starting_equity=starting_equity,
        ending_equity=ending_equity,
        kill_switch_triggered=kill_switch,
        metadata={"n_trials": float(cfg.n_trials)},
    )


def _dominant_strategic(info: dict[str, object]) -> str:
    """Return the strategic action taken most often so far (for the audit log)."""
    counts = info.get("strategic_counts")
    if not isinstance(counts, dict) or not counts:
        return "UNKNOWN"
    best = max(counts.items(), key=lambda kv: kv[1])
    key = best[0]
    return key.value if hasattr(key, "value") else str(key)


def _extract_trade(env: IronCondorTradingEnv, step: int) -> TradeRecord | None:
    """Return the trade record for the just-executed step, if a trade occurred."""
    last = env.last_trade()
    if last is None:
        return None
    return TradeRecord(
        step=step,
        pnl=last["pnl"],
        credit=last["credit"],
        transaction_cost=last["transaction_cost"],
        quantity=int(last["quantity"]),
    )
