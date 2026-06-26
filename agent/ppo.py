r"""PPO agent with a distributional (risk-sensitive) critic (SPEC §4.2).

This is the brain's learning algorithm, implemented from first principles on top of the
verified components (:mod:`networks`, :mod:`gae`, :mod:`rollout`, :mod:`losses`). It is
proximal policy optimization with:

* a diagonal-Gaussian actor and a quantile (distributional) critic,
* GAE advantages and quantile-Huber value targets,
* the PPO-clip surrogate with an entropy bonus (exploration / anti-mode-collapse) and a
  configurable value-loss coefficient,
* global gradient clipping and an optional KL early-stop, the standard PPO stability guards.

Everything is deterministic given the seed. Each :meth:`update` consumes one full rollout and
returns a :class:`PPOUpdateStats` record (losses, KL, clip fraction, entropy) used by the
training loop's anti-collapse monitors (SPEC §4.4).
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
from .networks import DistributionalCritic, GaussianActor, quantile_fractions
from .rollout import RolloutBuffer

__all__ = ["PPOAgent", "PPOConfig", "PPOUpdateStats"]

_logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class PPOConfig:
    """Hyper-parameters for the PPO agent."""

    learning_rate: float = 3e-4
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    n_epochs: int = 10
    minibatch_size: int = 64
    max_grad_norm: float = 0.5
    target_kl: float = 0.03  # early-stop the epoch loop if mean KL exceeds this
    huber_kappa: float = 1.0
    hidden_sizes: tuple[int, ...] = (128, 128)
    n_quantiles: int = 32
    cvar_alpha: float = 0.1
    seed: int = 0

    def __post_init__(self) -> None:
        if self.learning_rate <= 0.0:
            raise ValidationError("learning_rate must be positive", context={})
        if not 0.0 < self.clip_epsilon < 1.0:
            raise ValidationError("clip_epsilon must lie in (0, 1)", context={})
        if self.entropy_coef < 0.0 or self.value_coef < 0.0:
            raise ValidationError("entropy_coef and value_coef must be non-negative", context={})
        if self.n_epochs < 1 or self.minibatch_size < 1:
            raise ValidationError("n_epochs and minibatch_size must be >= 1", context={})
        if self.max_grad_norm <= 0.0 or self.target_kl <= 0.0:
            raise ValidationError("max_grad_norm and target_kl must be positive", context={})


@dataclass
class PPOUpdateStats:
    """Diagnostics from one PPO update over a rollout."""

    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    clip_fraction: float
    epochs_run: int


class PPOAgent:
    """Proximal Policy Optimization with a distributional critic.

    Parameters
    ----------
    obs_dim, action_dim:
        Environment dimensions.
    config:
        PPO hyper-parameters.
    """

    def __init__(self, *, obs_dim: int, action_dim: int, config: PPOConfig | None = None) -> None:
        if obs_dim < 1 or action_dim < 1:
            raise ValidationError("obs_dim and action_dim must be >= 1", context={})
        self._config = config or PPOConfig()
        torch.manual_seed(self._config.seed)
        self._obs_dim = obs_dim
        self._action_dim = action_dim
        self.actor = GaussianActor(
            obs_dim=obs_dim, action_dim=action_dim, hidden_sizes=self._config.hidden_sizes
        )
        self.critic = DistributionalCritic(
            obs_dim=obs_dim,
            n_quantiles=self._config.n_quantiles,
            hidden_sizes=self._config.hidden_sizes,
            cvar_alpha=self._config.cvar_alpha,
        )
        self._optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=self._config.learning_rate,
        )
        self._taus = quantile_fractions(self._config.n_quantiles)
        self._torch_rng = torch.Generator().manual_seed(self._config.seed)

    @property
    def config(self) -> PPOConfig:
        """The PPO configuration."""
        return self._config

    # -- acting ----------------------------------------------------------------------

    @torch.no_grad()
    def act(
        self, observation: NDArray[np.float32], *, deterministic: bool = False
    ) -> tuple[NDArray[np.float32], float, float]:
        """Return ``(action, log_prob, value)`` for a single observation.

        With ``deterministic=True`` the policy mean is returned (for evaluation); otherwise an
        action is sampled from the Gaussian (for exploration during rollouts).
        """
        obs = torch.as_tensor(observation, dtype=torch.float32).reshape(1, -1)
        dist = self.actor.distribution(obs)
        # torch.distributions methods are untyped in the stubs.
        action = dist.mean if deterministic else dist.sample()  # type: ignore[no-untyped-call]
        log_prob = float(dist.log_prob(action).sum(dim=-1).item())  # type: ignore[no-untyped-call]
        value = float(self.critic.value(obs).item())
        return action.numpy().reshape(-1), log_prob, value

    @torch.no_grad()
    def value(self, observation: NDArray[np.float32]) -> float:
        """Return the scalar value baseline for an observation (for GAE bootstrapping)."""
        obs = torch.as_tensor(observation, dtype=torch.float32).reshape(1, -1)
        return float(self.critic.value(obs).item())

    @torch.no_grad()
    def cvar(self, observation: NDArray[np.float32]) -> float:
        """Return the critic's CVaR (left-tail return estimate) for an observation."""
        obs = torch.as_tensor(observation, dtype=torch.float32).reshape(1, -1)
        return float(self.critic.cvar(obs).item())

    # -- learning --------------------------------------------------------------------

    def update(self, buffer: RolloutBuffer, *, rng: np.random.Generator) -> PPOUpdateStats:
        """Run the PPO update over one full rollout buffer and return diagnostics."""
        buffer.compute_advantages()
        cfg = self._config

        last_policy_loss = 0.0
        last_value_loss = 0.0
        last_entropy = 0.0
        last_clip_fraction = 0.0
        approx_kl = 0.0
        epochs_run = 0

        for _epoch in range(cfg.n_epochs):
            epoch_kls: list[float] = []
            for batch in buffer.iter_minibatches(batch_size=cfg.minibatch_size, rng=rng):
                obs = torch.as_tensor(batch.observations, dtype=torch.float32)
                actions = torch.as_tensor(batch.actions, dtype=torch.float32)
                old_log_probs = torch.as_tensor(batch.old_log_probs, dtype=torch.float32)
                advantages = torch.as_tensor(batch.advantages, dtype=torch.float32)
                returns = torch.as_tensor(batch.returns, dtype=torch.float32)

                new_log_probs, entropy = self.actor.evaluate_actions(obs, actions)
                policy_loss, clip_fraction = ppo_clip_loss(
                    new_log_probs=new_log_probs,
                    old_log_probs=old_log_probs,
                    advantages=advantages,
                    clip_epsilon=cfg.clip_epsilon,
                )
                predicted_quantiles = self.critic.quantiles(obs)
                value_loss = quantile_huber_loss(
                    predicted_quantiles=predicted_quantiles,
                    target_returns=returns,
                    taus=self._taus,
                    kappa=cfg.huber_kappa,
                )
                entropy_mean = entropy.mean()
                loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy_mean

                self._optimizer.zero_grad()
                loss.backward()  # type: ignore[no-untyped-call]
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    cfg.max_grad_norm,
                )
                self._optimizer.step()

                with torch.no_grad():
                    # Schulman's low-variance approximate KL estimator.
                    log_ratio = new_log_probs - old_log_probs
                    batch_kl = float(torch.mean(torch.exp(log_ratio) - 1.0 - log_ratio).item())
                epoch_kls.append(batch_kl)
                last_policy_loss = float(policy_loss.item())
                last_value_loss = float(value_loss.item())
                last_entropy = float(entropy_mean.item())
                last_clip_fraction = float(clip_fraction.item())

            epochs_run += 1
            approx_kl = float(np.mean(epoch_kls)) if epoch_kls else 0.0
            if approx_kl > cfg.target_kl:
                # Early-stop the epoch loop to keep the policy update proximal.
                break

        stats = PPOUpdateStats(
            policy_loss=last_policy_loss,
            value_loss=last_value_loss,
            entropy=last_entropy,
            approx_kl=approx_kl,
            clip_fraction=last_clip_fraction,
            epochs_run=epochs_run,
        )
        _logger.debug(
            "ppo_update",
            extra={
                "policy_loss": stats.policy_loss,
                "value_loss": stats.value_loss,
                "entropy": stats.entropy,
                "approx_kl": stats.approx_kl,
                "clip_fraction": stats.clip_fraction,
                "epochs_run": stats.epochs_run,
            },
        )
        return stats

    # -- serialization ---------------------------------------------------------------

    def state_dict(self) -> dict[str, object]:
        """Return a serializable snapshot of the agent's parameters."""
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "optimizer": self._optimizer.state_dict(),
            "obs_dim": self._obs_dim,
            "action_dim": self._action_dim,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        """Restore parameters from :meth:`state_dict`."""
        self.actor.load_state_dict(state["actor"])  # type: ignore[arg-type]
        self.critic.load_state_dict(state["critic"])  # type: ignore[arg-type]
        self._optimizer.load_state_dict(state["optimizer"])  # type: ignore[arg-type]
