"""Reinforcement-learning environment for the iron-condor agent (SPEC §4.1-4.5).

A Gymnasium POMDP wraps the full engine (distribution, regime, chain, strategy, risk
supervisor) into a learnable trading environment with a risk-sensitive reward and an
always-on deterministic risk overlay.

Public surface:

* Action: :class:`ActionBounds`, :class:`DecodedAction`, :func:`decode_action`,
  :data:`ACTION_DIM`.
* Observation: :class:`ObservationInputs`, :func:`build_observation`,
  :func:`observation_bounds`, :data:`OBSERVATION_DIM`.
* Reward: :class:`RewardConfig`, :class:`RewardInputs`, :class:`RewardBreakdown`,
  :func:`compute_reward`.
* Scenario: :class:`ScenarioConfig`, :class:`ScenarioStep`, :class:`Episode`,
  :func:`generate_episode`.
* Environment: :class:`IronCondorTradingEnv`, :class:`EnvConfig`.
"""

from __future__ import annotations

from .action import ACTION_DIM, ActionBounds, DecodedAction, decode_action
from .growth_reward import (
    GrowthRewardBreakdown,
    GrowthRewardConfig,
    GrowthRewardInputs,
    compute_growth_reward,
)
from .observation import (
    OBSERVATION_DIM,
    ObservationInputs,
    build_observation,
    observation_bounds,
)
from .reward import RewardBreakdown, RewardConfig, RewardInputs, compute_reward
from .scenario import Episode, ScenarioConfig, ScenarioStep, generate_episode

__all__ = [
    "ACTION_DIM",
    "OBSERVATION_DIM",
    "ActionBounds",
    "DecodedAction",
    "Episode",
    "GrowthRewardBreakdown",
    "GrowthRewardConfig",
    "GrowthRewardInputs",
    "ObservationInputs",
    "RewardBreakdown",
    "RewardConfig",
    "RewardInputs",
    "ScenarioConfig",
    "ScenarioStep",
    "build_observation",
    "compute_growth_reward",
    "compute_reward",
    "decode_action",
    "generate_episode",
    "observation_bounds",
]


def make_env(**kwargs: object) -> object:
    """Construct the trading environment, importing the gym-dependent module lazily.

    Keeping :class:`IronCondorTradingEnv` out of the eager package imports means the rest of
    :mod:`options_engine.rl` (action/observation/reward/scenario) is usable without gymnasium
    installed; only this factory pulls it in.
    """
    from .env import IronCondorTradingEnv  # noqa: PLC0415 (deliberate lazy gym import)

    return IronCondorTradingEnv(**kwargs)  # type: ignore[arg-type]
