r"""Gymnasium trading environment for the RL agent (SPEC §4.1-4.5).

This wraps the entire engine -- distribution, regime, chain, condor selection, sizing, and the
deterministic risk supervisor -- into a single POMDP the agent learns in. Critically, the
**risk supervisor overlay is always active** (SPEC §4.5): the agent proposes, but the
supervisor disposes, so the policy is trained *inside* the real constraint set and can never
learn to breach hard limits. Limit breaches are surfaced to the reward as a penalty, not as a
silently-allowed action.

Episode flow
------------
1. ``reset`` draws a domain-randomized :class:`Episode` and returns the first observation.
2. Each ``step`` decodes the action, and:
   * if FLAT, holds; otherwise constructs an iron condor at the agent's tactical parameters,
   * sizes it under fractional Kelly, clears it through the risk supervisor (veto -> no
     trade + breach penalty), and books realized P&L at the position's expiry payoff,
   * advances the market one step and returns ``(obs, reward, terminated, truncated, info)``.
3. The episode terminates early if the kill-switch trips (drawdown limit).

The environment is deterministic given its seed and tracks anti-collapse diagnostics
(fraction of steps traded, strategic-action histogram) in ``info`` so training can detect
collapse-to-no-trade or collapse-to-always-trade (SPEC §4.4).
"""

from __future__ import annotations

import datetime as _dt
from typing import Any

import numpy as np
from numpy.typing import NDArray

try:  # Gymnasium is an optional (rl) dependency; fail clearly if used without it.
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover - exercised only without the rl extra
    raise ImportError(
        "the RL environment requires gymnasium; install the 'rl' extra: pip install -e '.[rl]'"
    ) from exc

from ..core.config import RiskConfig
from ..core.enums import StrategicAction, VolRegime
from ..core.errors import ValidationError
from ..core.random import RandomFactory
from ..pricing.instruments import IronCondor
from ..pricing.payoff import iron_condor_pnl
from ..strategy.account import Account, OpenPosition
from ..strategy.condor_selection import CondorSelectionConfig, select_iron_condor
from ..strategy.risk_supervisor import RiskSupervisor, RiskSupervisorConfig
from ..strategy.sizing import SizingInputs, size_position
from .action import ACTION_DIM, ActionBounds, DecodedAction, decode_action
from .observation import (
    OBSERVATION_DIM,
    ObservationInputs,
    build_observation,
    observation_bounds,
)
from .reward import RewardConfig, RewardInputs, compute_reward
from .scenario import Episode, ScenarioConfig, ScenarioStep, generate_episode

__all__ = ["EnvConfig", "IronCondorTradingEnv"]


class EnvConfig:
    """Top-level environment configuration."""

    __slots__ = (
        "limit_breach_penalty",
        "max_open_positions",
        "multiplier",
        "shaping_coef",
        "starting_cash",
        "recalibrate_interval_minutes",
    )

    def __init__(
        self,
        *,
        starting_cash: float = 100_000.0,
        multiplier: float = 100.0,
        max_open_positions: int = 10,
        shaping_coef: float = 1.0,
        limit_breach_penalty: float = 5.0,
        recalibrate_interval_minutes: int = 15,
    ) -> None:
        if starting_cash <= 0.0 or multiplier <= 0.0:
            raise ValidationError("starting_cash and multiplier must be positive", context={})
        if max_open_positions < 1:
            raise ValidationError("max_open_positions must be >= 1", context={})
        if not 0.0 <= shaping_coef <= 1.0:
            raise ValidationError("shaping_coef must be in [0, 1]", context={})
        if limit_breach_penalty < 0.0:
            raise ValidationError("limit_breach_penalty must be non-negative", context={})
        if recalibrate_interval_minutes < 1:
            raise ValidationError("recalibrate_interval_minutes must be >= 1", context={})
        self.starting_cash = starting_cash
        self.multiplier = multiplier
        self.max_open_positions = max_open_positions
        self.shaping_coef = shaping_coef
        self.limit_breach_penalty = limit_breach_penalty
        self.recalibrate_interval_minutes = recalibrate_interval_minutes


class IronCondorTradingEnv(gym.Env[NDArray[np.float32], NDArray[np.float32]]):
    """A Gymnasium POMDP for learning to trade under a hard risk overlay."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        risk: RiskConfig | None = None,
        seed: int = 0,
        env_config: EnvConfig | None = None,
        scenario_config: ScenarioConfig | None = None,
        reward_config: RewardConfig | None = None,
        selection_config: CondorSelectionConfig | None = None,
        action_bounds: ActionBounds | None = None,
    ) -> None:
        super().__init__()
        self._risk = risk or RiskConfig()
        self._env_config = env_config or EnvConfig()
        self._scenario_config = scenario_config or ScenarioConfig()
        self._reward_config = reward_config or RewardConfig()
        self._selection_config = selection_config or CondorSelectionConfig()
        self._action_bounds = action_bounds or ActionBounds()
        self._rng_factory = RandomFactory(seed)
        self._supervisor = RiskSupervisor(
            risk=self._risk,
            config=RiskSupervisorConfig(max_open_positions=self._env_config.max_open_positions),
        )

        low, high = observation_bounds()
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32)

        self._episode: Episode | None = None
        self._step_index: int = 0
        self._episode_index: int = 0
        self._account: Account = Account.open(starting_cash=self._env_config.starting_cash)
        self._risked_today: float = 0.0
        self._trade_steps: int = 0
        self._strategic_counts: dict[StrategicAction, int] = {}
        self._shaping_coef: float = self._env_config.shaping_coef
        self._last_trade: dict[str, float] | None = None

    def set_shaping_coef(self, value: float) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValidationError("shaping_coef must lie in [0, 1]", context={"value": value})
        self._shaping_coef = float(value)

    def account_equity(self) -> float:
        return self._account.equity()

    def last_trade(self) -> dict[str, float] | None:
        return self._last_trade

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng_factory = RandomFactory(seed)
            self._episode_index = 0
        else:
            self._episode_index += 1

        # Honor an alpha supplied via options so the helper critic can drive the env.
        alpha = None
        if isinstance(options, dict):
            candidate = options.get("alpha")
            if candidate is not None:
                from ..core.market_alpha import MarketAlpha as _MA

                if isinstance(candidate, _MA):
                    alpha = candidate
                else:
                    alpha = _MA.scalar(float(candidate))

        self._episode = generate_episode(
            rng_factory=self._rng_factory,
            config=self._scenario_config,
            episode_index=self._episode_index,
            alpha=alpha,
        )
        self._step_index = 0
        self._account = Account.open(starting_cash=self._env_config.starting_cash)
        self._risked_today = 0.0
        self._trade_steps = 0
        self._strategic_counts = dict.fromkeys(StrategicAction, 0)

        obs = self._build_observation(self._current_step())
        return obs, self._info(reward_breakdown=None, traded=False, breached=False)

    def step(
        self, action: NDArray[np.float32]
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        if self._episode is None:
            raise ValidationError("step called before reset", context={})

        self._last_trade = None
        decoded = decode_action(np.asarray(action, dtype=np.float64), bounds=self._action_bounds)
        self._strategic_counts[decoded.strategic] += 1
        step = self._current_step()

        pnl_change, transaction_cost, theta_captured, traded, breached = self._execute_iron_condor(
            decoded, step
        )

        equity = self._account.equity()
        margin_util = self._account.total_margin / equity if equity > 0 else 2.0
        incremental_cvar = self._estimate_incremental_cvar(decoded, step) if traded else 0.0

        reward_breakdown = compute_reward(
            RewardInputs(
                pnl_change=pnl_change,
                incremental_cvar=incremental_cvar,
                transaction_cost=transaction_cost,
                margin_utilization=margin_util,
                theta_captured=theta_captured,
                limit_breached=breached,
            ),
            config=self._reward_config,
            shaping_coef=self._shaping_coef,
            limit_breach_penalty=self._env_config.limit_breach_penalty,
        )

        kill = self._supervisor.check_kill_switch(self._account)
        terminated = kill.kill_switch_triggered

        self._step_index += 1
        truncated = self._step_index >= len(self._episode)

        if traded:
            self._trade_steps += 1

        next_step = None if (terminated or truncated) else self._current_step()
        obs = (
            self._build_observation(next_step)
            if next_step is not None
            else np.zeros(OBSERVATION_DIM, dtype=np.float32)
        )
        info = self._info(reward_breakdown=reward_breakdown, traded=traded, breached=breached)
        if terminated:
            info["kill_switch"] = kill.reason
        return obs, float(reward_breakdown.total), terminated, truncated, info

    def _current_step(self) -> ScenarioStep:
        assert self._episode is not None
        return self._episode.steps[self._step_index]

    def _execute_iron_condor(
        self, decoded: DecodedAction, step: ScenarioStep
    ) -> tuple[float, float, float, bool, bool]:
        if not decoded.is_trade:
            return 0.0, 0.0, 0.0, False, False

        candidate = select_iron_condor(
            step.realized_distribution,
            step.chain,
            config=self._selection_with(decoded),
            terminal_sample=step.realized_terminal_sample,
        )
        if candidate is None:
            return 0.0, 0.0, 0.0, False, False

        max_loss_per_condor = candidate.condor.max_spread_width - candidate.net_credit
        if max_loss_per_condor <= 0.0:
            return 0.0, 0.0, 0.0, False, False

        equity = self._account.equity()
        available_margin = max(0.0, equity - self._account.total_margin)
        per_condor_pnl = iron_condor_pnl(
            candidate.condor,
            step.realized_terminal_sample,
            entry_credit=candidate.net_credit,
            multiplier=1.0,
        )
        pnl_per_unit_risk = per_condor_pnl / max_loss_per_condor
        sizing = size_position(
            SizingInputs(
                account_equity=equity,
                win_probability=candidate.win_probability,
                net_credit=candidate.net_credit,
                max_loss_per_condor=max_loss_per_condor,
                multiplier=self._env_config.multiplier,
                available_margin=available_margin,
                risked_today=self._risked_today,
                pnl_per_unit_risk=pnl_per_unit_risk,
            ),
            risk=self._risk,
        )
        quantity = int(sizing.quantity * decoded.size_fraction)
        if quantity < 1:
            return 0.0, 0.0, 0.0, False, False

        condor = IronCondor(
            put_long_strike=candidate.condor.put_long_strike,
            put_short_strike=candidate.condor.put_short_strike,
            call_short_strike=candidate.condor.call_short_strike,
            call_long_strike=candidate.condor.call_long_strike,
            expiry=candidate.condor.expiry,
            quantity=quantity,
        )
        position = OpenPosition(
            position_id=f"E{self._episode_index}-S{self._step_index}",
            condor=condor,
            entry_credit=candidate.net_credit,
            quantity=quantity,
            multiplier=self._env_config.multiplier,
            entry_time=_epoch_plus_steps(self._step_index),
            entry_spot=step.spot,
        )

        approval = self._supervisor.approve_new_position(
            self._account,
            position,
            risked_today=self._risked_today,
            available_margin=available_margin,
        )
        if not approval.approved:
            return 0.0, 0.0, 0.0, False, True

        terminal_spot = float(
            step.realized_terminal_sample[
                self._rng_factory.generator("rl.settle").integers(
                    0, step.realized_terminal_sample.size
                )
            ]
        )
        pnl = float(
            iron_condor_pnl(
                condor,
                np.array([terminal_spot]),
                entry_credit=candidate.net_credit,
                multiplier=self._env_config.multiplier,
            )[0]
        )
        transaction_cost = candidate.spread_cost * quantity * self._env_config.multiplier
        theta_captured = candidate.net_credit * quantity * self._env_config.multiplier

        self._account = self._account.add_position(
            position, premium_received=theta_captured
        ).close_position(position.position_id, realized=pnl - theta_captured)
        self._risked_today += position.margin_requirement
        self._last_trade = {
            "pnl": pnl,
            "credit": theta_captured,
            "transaction_cost": transaction_cost,
            "quantity": float(quantity),
        }
        return pnl, transaction_cost, theta_captured, True, False

    def _selection_with(self, decoded: DecodedAction) -> CondorSelectionConfig:
        base = self._selection_config
        return CondorSelectionConfig(
            target_tail_probability=decoded.tail_probability,
            wing_width_fraction=decoded.wing_width_fraction,
            cvar_weight=base.cvar_weight,
            cost_weight=base.cost_weight,
            cvar_alpha=base.cvar_alpha,
            min_win_probability=base.min_win_probability,
            min_net_credit=base.min_net_credit,
        )

    def _estimate_incremental_cvar(self, decoded: DecodedAction, step: ScenarioStep) -> float:
        if not decoded.is_trade:
            return 0.0
        candidate = select_iron_condor(
            step.realized_distribution,
            step.chain,
            config=self._selection_with(decoded),
            terminal_sample=step.realized_terminal_sample,
        )
        if candidate is None:
            return 0.0
        return max(0.0, candidate.cvar) * self._env_config.multiplier

    def _build_observation(self, step: ScenarioStep) -> NDArray[np.float32]:
        dist = step.realized_distribution
        expected_move = float(np.std(dist.log_returns))
        sigma = expected_move if expected_move > 0 else 1e-6
        prob_in_one_sigma = dist.probability_in_range(-sigma, sigma)
        left_tail = dist.probability_below(-2.0 * sigma)
        right_tail = 1.0 - dist.probability_below(2.0 * sigma)

        equity = self._account.equity()
        margin_util = self._account.total_margin / equity if equity > 0 else 2.0
        open_fraction = self._account.position_count() / self._env_config.max_open_positions
        ttx = 1.0 - (self._step_index / max(1, len(self._episode or ())))

        inputs = ObservationInputs(
            prob_in_one_sigma=prob_in_one_sigma,
            left_tail_prob=left_tail,
            right_tail_prob=right_tail,
            expected_move=min(expected_move, 1.0),
            regime=step.regime,
            atm_relative_spread=min(step.atm_relative_spread, 1.0),
            margin_utilization=margin_util,
            drawdown=self._account.drawdown(),
            open_position_fraction=min(open_fraction, 1.0),
            time_to_expiry_fraction=ttx,
            news_cooloff_active=False,
        )
        return build_observation(inputs)

    def _info(self, *, reward_breakdown: object, traded: bool, breached: bool) -> dict[str, Any]:
        total_steps = max(1, self._step_index)
        info: dict[str, Any] = {
            "equity": self._account.equity(),
            "realized_pnl": self._account.realized_pnl,
            "trade_fraction": self._trade_steps / total_steps,
            "strategic_counts": dict(self._strategic_counts),
            "traded": traded,
            "limit_breached": breached,
        }
        if reward_breakdown is not None:
            rb = reward_breakdown
            info["reward_breakdown"] = {
                "pnl": rb.pnl_term,  # type: ignore[attr-defined]
                "risk": rb.risk_term,  # type: ignore[attr-defined]
                "cost": rb.cost_term,  # type: ignore[attr-defined]
                "margin": rb.margin_term,  # type: ignore[attr-defined]
                "tail": rb.tail_term,  # type: ignore[attr-defined]
                "shaping": rb.shaping_term,  # type: ignore[attr-defined]
            }
        if self._episode is not None:
            info["regime_low_prob"] = self._current_step_safe_regime()
        return info

    def _current_step_safe_regime(self) -> float:
        assert self._episode is not None
        if self._step_index >= len(self._episode):
            return 0.0
        return self._episode.steps[self._step_index].regime.current_prob(VolRegime.LOW)


def _epoch_plus_steps(step_index: int) -> _dt.datetime:
    return _dt.datetime(2024, 1, 1, tzinfo=_dt.UTC) + _dt.timedelta(days=step_index)
