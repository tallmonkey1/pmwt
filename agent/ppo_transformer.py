"""PPO-Transformer: PPO with a transformer backbone that conditions on past observations.

The rBergomi price model is non-Markovian (the conditional distribution of future
variance depends on the path of past variance through the Volterra kernel). A
policy that sees only the current observation cannot exploit this dependency,
no matter how expressive its trunk is. The :class:`PPOTransformerAgent` closes
exactly that gap: a transformer encoder runs over the last ``seq_len`` observations
and produces a context vector that both the actor head and the distributional
(quantile/CVaR) critic head condition on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from ..core.errors import ValidationError
from ..core.logging import get_logger
from .losses import ppo_clip_loss, quantile_huber_loss
from .networks import DistributionalCritic, quantile_fractions
from .transformer_backbone import TransformerBackbone

__all__ = ["PPOTransformerAgent", "PPOTransformerConfig", "PPOTransformerStats"]

_logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class PPOTransformerConfig:
    """Hyper-parameters for the PPO-Transformer agent."""

    learning_rate: float = 3e-4
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    n_epochs: int = 10
    minibatch_size: int = 64
    max_grad_norm: float = 0.5
    target_kl: float = 0.03
    huber_kappa: float = 1.0
    n_quantiles: int = 32
    cvar_alpha: float = 0.1
    seed: int = 0
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 3
    dim_feedforward: int = 256
    dropout: float = 0.0
    max_seq_len: int = 64
    seq_len: int = 16
    log_std_min: float = -5.0
    log_std_max: float = 2.0

    def __post_init__(self) -> None:
        if self.learning_rate <= 0.0:
            raise ValidationError("learning_rate must be positive", context={})
        if not 0.0 < self.clip_epsilon < 1.0:
            raise ValidationError("clip_epsilon must lie in (0, 1)", context={})
        if self.entropy_coef < 0.0 or self.value_coef < 0.0:
            raise ValidationError(
                "entropy_coef and value_coef must be non-negative", context={}
            )
        if self.n_epochs < 1 or self.minibatch_size < 1:
            raise ValidationError(
                "n_epochs and minibatch_size must be >= 1", context={}
            )
        if self.seq_len < 1:
            raise ValidationError("seq_len must be >= 1", context={})
        if self.seq_len > self.max_seq_len:
            raise ValidationError(
                "seq_len must not exceed max_seq_len",
                context={"seq_len": self.seq_len, "max_seq_len": self.max_seq_len},
            )


@dataclass
class PPOTransformerStats:
    """Per-update diagnostics for the PPO-Transformer."""

    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    clip_fraction: float
    epochs_run: int


class PPOTransformerAgent:
    """PPO with a shared transformer backbone (memory over past observations)."""

    def __init__(
        self,
        *,
        obs_dim: int,
        action_dim: int,
        config: PPOTransformerConfig | None = None,
    ) -> None:
        if obs_dim < 1 or action_dim < 1:
            raise ValidationError("obs_dim and action_dim must be >= 1", context={})
        self._config = config or PPOTransformerConfig()
        torch.manual_seed(self._config.seed)
        self._obs_dim = obs_dim
        self._action_dim = action_dim

        self.backbone = TransformerBackbone(
            input_dim=obs_dim,
            d_model=self._config.d_model,
            nhead=self._config.nhead,
            num_layers=self._config.num_layers,
            dim_feedforward=self._config.dim_feedforward,
            dropout=self._config.dropout,
            max_seq_len=self._config.max_seq_len,
        )
        self.actor_mean = nn.Linear(self._config.d_model, action_dim)
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.zeros_(self.actor_mean.bias)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.critic = DistributionalCritic(
            obs_dim=self._config.d_model,
            n_quantiles=self._config.n_quantiles,
            hidden_sizes=(self._config.d_model,),
            cvar_alpha=self._config.cvar_alpha,
        )
        params = (
            list(self.backbone.parameters())
            + list(self.actor_mean.parameters())
            + [self.log_std]
            + list(self.critic.parameters())
        )
        self._optimizer = torch.optim.Adam(params, lr=self._config.learning_rate)
        self._taus = quantile_fractions(self._config.n_quantiles)
        self._torch_rng = torch.Generator().manual_seed(self._config.seed)

    @property
    def config(self) -> PPOTransformerConfig:
        """The PPO-Transformer configuration."""
        return self._config

    @torch.no_grad()
    def encode(self, obs_sequence: NDArray[np.float32]) -> NDArray[np.float32]:
        """Encode ``(seq_len, obs_dim)`` -> ``(d_model,)`` context vector."""
        if obs_sequence.ndim == 1:
            obs_sequence = obs_sequence[np.newaxis, ...]
        if obs_sequence.ndim == 2:
            obs_sequence = obs_sequence[np.newaxis, ...]
        x = torch.as_tensor(obs_sequence, dtype=torch.float32)
        ctx = self.backbone(x)
        return ctx.squeeze(0).numpy()

    def _dist(self, ctx: torch.Tensor) -> torch.distributions.Normal:
        mean = self.actor_mean(ctx)
        log_std = torch.clamp(self.log_std, self._config.log_std_min, self._config.log_std_max)
        std = torch.exp(log_std).expand_as(mean)
        return torch.distributions.Normal(mean, std)

    @torch.no_grad()
    def act(
        self,
        obs_sequence: NDArray[np.float32],
        *,
        deterministic: bool = False,
    ) -> tuple[NDArray[np.float32], float, float]:
        """Return ``(action, log_prob, value)`` for a single observation sequence."""
        if obs_sequence.ndim == 2:
            obs_sequence = obs_sequence[np.newaxis, ...]
        x = torch.as_tensor(obs_sequence, dtype=torch.float32)
        ctx = self.backbone(x)
        dist = self._dist(ctx)
        action = dist.mean if deterministic else dist.sample()
        log_prob = float(dist.log_prob(action).sum(dim=-1).item())
        value = float(self.critic.value(ctx).item())
        return action.squeeze(0).numpy(), log_prob, value

    @torch.no_grad()
    def value(self, obs_sequence: NDArray[np.float32]) -> float:
        """Return the critic's value baseline for an observation sequence."""
        if obs_sequence.ndim == 2:
            obs_sequence = obs_sequence[np.newaxis, ...]
        x = torch.as_tensor(obs_sequence, dtype=torch.float32)
        ctx = self.backbone(x)
        return float(self.critic.value(ctx).item())

    @torch.no_grad()
    def cvar(self, obs_sequence: NDArray[np.float32]) -> float:
        """Return the critic's left-tail CVaR estimate for an observation sequence."""
        if obs_sequence.ndim == 2:
            obs_sequence = obs_sequence[np.newaxis, ...]
        x = torch.as_tensor(obs_sequence, dtype=torch.float32)
        ctx = self.backbone(x)
        return float(self.critic.cvar(ctx).item())

    def update_from_sequences(
        self,
        *,
        obs_sequences: NDArray[np.float32],
        actions: NDArray[np.float32],
        old_log_probs: NDArray[np.float32],
        advantages: NDArray[np.float32],
        returns: NDArray[np.float32],
        rng: np.random.Generator,
    ) -> PPOTransformerStats:
        """Run one PPO update over a batch of pre-collected sequences."""
        cfg = self._config
        if obs_sequences.ndim != 3 or obs_sequences.shape[1] != cfg.seq_len:
            raise ValidationError(
                "obs_sequences must have shape (N, seq_len, obs_dim)",
                context={"shape": obs_sequences.shape, "seq_len": cfg.seq_len},
            )
        n = obs_sequences.shape[0]
        if n < cfg.minibatch_size:
            raise ValidationError(
                "batch too small for minibatch_size",
                context={"n": int(n), "minibatch_size": cfg.minibatch_size},
            )

        obs_t = torch.as_tensor(obs_sequences, dtype=torch.float32)
        actions_t = torch.as_tensor(actions, dtype=torch.float32)
        old_log_probs_t = torch.as_tensor(old_log_probs, dtype=torch.float32)
        advantages_t = torch.as_tensor(advantages, dtype=torch.float32)
        returns_t = torch.as_tensor(returns, dtype=torch.float32)

        adv_std = float(advantages_t.std().item())
        if adv_std > 1e-8:
            advantages_t = (advantages_t - advantages_t.mean()) / adv_std

        last_policy_loss = 0.0
        last_value_loss = 0.0
        last_entropy = 0.0
        last_clip_fraction = 0.0
        approx_kl = 0.0
        epochs_run = 0

        for _ in range(cfg.n_epochs):
            epoch_kls: list[float] = []
            indices = rng.permutation(n)
            for start in range(0, n, cfg.minibatch_size):
                idx = indices[start : start + cfg.minibatch_size]
                obs_b = obs_t[idx]
                actions_b = actions_t[idx]
                old_log_probs_b = old_log_probs_t[idx]
                advantages_b = advantages_t[idx]
                returns_b = returns_t[idx]

                ctx = self.backbone(obs_b)
                dist = self._dist(ctx)
                new_log_probs = dist.log_prob(actions_b).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1)

                policy_loss, clip_fraction = ppo_clip_loss(
                    new_log_probs=new_log_probs,
                    old_log_probs=old_log_probs_b,
                    advantages=advantages_b,
                    clip_epsilon=cfg.clip_epsilon,
                )
                predicted_quantiles = self.critic.quantiles(ctx)
                value_loss = quantile_huber_loss(
                    predicted_quantiles=predicted_quantiles,
                    target_returns=returns_b,
                    taus=self._taus,
                    kappa=cfg.huber_kappa,
                )
                loss = (
                    policy_loss
                    + cfg.value_coef * value_loss
                    - cfg.entropy_coef * entropy.mean()
                )

                self._optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.backbone.parameters())
                    + list(self.actor_mean.parameters())
                    + [self.log_std]
                    + list(self.critic.parameters()),
                    cfg.max_grad_norm,
                )
                self._optimizer.step()

                with torch.no_grad():
                    log_ratio = new_log_probs - old_log_probs_b
                    batch_kl = float(
                        torch.mean(torch.exp(log_ratio) - 1.0 - log_ratio).item()
                    )
                epoch_kls.append(batch_kl)
                last_policy_loss = float(policy_loss.item())
                last_value_loss = float(value_loss.item())
                last_entropy = float(entropy.mean().item())
                last_clip_fraction = float(clip_fraction.item())

            epochs_run += 1
            approx_kl = float(np.mean(epoch_kls)) if epoch_kls else 0.0
            if approx_kl > cfg.target_kl:
                break

        return PPOTransformerStats(
            policy_loss=last_policy_loss,
            value_loss=last_value_loss,
            entropy=last_entropy,
            approx_kl=approx_kl,
            clip_fraction=last_clip_fraction,
            epochs_run=epochs_run,
        )
