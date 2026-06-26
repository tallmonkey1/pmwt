"""The reinforcement-learning agent: PPO with a distributional, risk-sensitive critic.

This is the system's decision brain (SPEC §4.2): a from-scratch, fully-tested PPO
implementation with a quantile (distributional) value head that exposes a CVaR tail estimate,
trained with GAE advantages and a quantile-Huber value loss, under anti-collapse monitoring.

Two agent variants are exported:

* :class:`PPOAgent` -- the original MLP-based PPO (fast for unit tests and small
  baselines). No memory; sees only the current observation.
* :class:`PPOTransformerAgent` -- a memory-enabled variant whose actor and critic
  share a causal Transformer backbone over the last ``seq_len`` observations. This
  is the variant that can exploit the non-Markovian structure of the rBergomi
  price model (the Volterra kernel means future variance depends on the path of past
  variance), which a single-observation MLP cannot.

The **helper critic** (:class:`HelperCritic`) is the meta-controller that learns
the :class:`MarketAlpha` vector driving every internal diagnostic of the main agent
toward ``1``.

Public surface:

* Networks: :class:`GaussianActor`, :class:`DistributionalCritic`, :func:`quantile_fractions`,
  :class:`TransformerBackbone`.
* Advantages: :func:`compute_gae`.
* Rollout: :class:`RolloutBuffer`, :class:`RolloutBatch`.
* Losses: :func:`ppo_clip_loss`, :func:`quantile_huber_loss`.
* Agent (MLP): :class:`PPOAgent`, :class:`PPOConfig`, :class:`PPOUpdateStats`.
* Agent (Transformer): :class:`PPOTransformerAgent`, :class:`PPOTransformerConfig`,
  :class:`PPOTransformerStats`.
* Helper critic: :class:`HelperCritic`, :class:`HelperCriticConfig`,
  :func:`alpha_lattice`, :func:`feature_score_from_components`.
* Training: :class:`Trainer`, :class:`TrainerConfig`, :class:`TrainingIteration`,
  :class:`TrainingHistory`, :class:`CollapseMonitor`.
"""

from __future__ import annotations

from .gae import compute_gae
from .helper_critic import (
    DEFAULT_HELPER_LATTICE_SIZE,
    HelperCritic,
    HelperCriticConfig,
    alpha_lattice,
    feature_score_from_components,
)
from .losses import ppo_clip_loss, quantile_huber_loss
from .networks import DistributionalCritic, GaussianActor, quantile_fractions
from .ppo import PPOAgent, PPOConfig, PPOUpdateStats
from .ppo_transformer import (
    PPOTransformerAgent,
    PPOTransformerConfig,
    PPOTransformerStats,
)
from .rollout import RolloutBatch, RolloutBuffer
from .trainer import (
    CollapseMonitor,
    Trainer,
    TrainerConfig,
    TrainingHistory,
    TrainingIteration,
)
from .transformer_backbone import SinusoidalPositionalEncoding, TransformerBackbone

__all__ = [
    "CollapseMonitor",
    "DEFAULT_HELPER_LATTICE_SIZE",
    "DistributionalCritic",
    "GaussianActor",
    "HelperCritic",
    "HelperCriticConfig",
    "PPOAgent",
    "PPOConfig",
    "PPOTransformerAgent",
    "PPOTransformerConfig",
    "PPOTransformerStats",
    "PPOUpdateStats",
    "RolloutBatch",
    "RolloutBuffer",
    "SinusoidalPositionalEncoding",
    "Trainer",
    "TrainerConfig",
    "TrainingHistory",
    "TrainingIteration",
    "TransformerBackbone",
    "alpha_lattice",
    "compute_gae",
    "feature_score_from_components",
    "ppo_clip_loss",
    "quantile_fractions",
    "quantile_huber_loss",
] 
