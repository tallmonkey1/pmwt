r"""Actor and distributional-critic networks for the RL agent (SPEC §4.2).

Two networks, both implemented from first principles so every line is understood, seeded, and
testable (no black-box RL library is used for the core learning logic):

* :class:`GaussianActor` -- a diagonal-Gaussian policy over the continuous action box. It
  outputs a state-dependent mean and a state-independent (learnable) log-standard-deviation,
  the standard, stable PPO parameterization. Actions are sampled from this Gaussian; the
  environment's decoder clips them into admissible ranges, so the log-probabilities are exact
  for the (unbounded) sampling distribution.

* :class:`DistributionalCritic` -- predicts ``K`` quantiles of the *return distribution*
  ``Z(s)`` rather than a single scalar value (a QR-DQN/IQN-style value head). The mean of the
  quantiles is the value baseline used by GAE; the lower quantiles give a Conditional
  Value-at-Risk (CVaR) estimate used for tail-risk monitoring and risk-sensitive shaping
  (SPEC §4.2, §4.4).

Both use the same small, layer-normalized MLP backbone with orthogonal initialization
(the well-established choice for stable on-policy RL).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ..core.errors import ValidationError

__all__ = ["DistributionalCritic", "GaussianActor", "quantile_fractions"]

# Bounds on the learnable log-std keep the policy from collapsing (std -> 0, premature
# determinism / mode collapse) or exploding (std -> inf, no learning signal).
_LOG_STD_MIN = -5.0
_LOG_STD_MAX = 2.0


def _mlp(
    in_dim: int, hidden_sizes: tuple[int, ...], *, activation: type[nn.Module]
) -> nn.Sequential:
    """Build a layer-normalized MLP trunk with orthogonal initialization."""
    layers: list[nn.Module] = []
    last = in_dim
    for width in hidden_sizes:
        linear = nn.Linear(last, width)
        nn.init.orthogonal_(linear.weight, gain=float(np.sqrt(2.0)))
        nn.init.zeros_(linear.bias)
        layers.append(linear)
        layers.append(nn.LayerNorm(width))
        layers.append(activation())
        last = width
    return nn.Sequential(*layers)


def quantile_fractions(n_quantiles: int) -> torch.Tensor:
    r"""Return the ``K`` midpoint quantile fractions ``tau_i = (i + 0.5) / K``.

    Midpoint fractions (rather than endpoints) are the standard QR-DQN choice; they avoid the
    degenerate 0/1 quantiles and give an unbiased grid for the value distribution.
    """
    if n_quantiles < 1:
        raise ValidationError("n_quantiles must be >= 1", context={"n_quantiles": n_quantiles})
    return (torch.arange(n_quantiles, dtype=torch.float32) + 0.5) / n_quantiles


class GaussianActor(nn.Module):
    """Diagonal-Gaussian policy network.

    Parameters
    ----------
    obs_dim:
        Observation dimension.
    action_dim:
        Action dimension.
    hidden_sizes:
        Hidden-layer widths for the shared trunk.
    """

    def __init__(
        self, *, obs_dim: int, action_dim: int, hidden_sizes: tuple[int, ...] = (128, 128)
    ) -> None:
        super().__init__()
        if obs_dim < 1 or action_dim < 1:
            raise ValidationError("obs_dim and action_dim must be >= 1", context={})
        if not hidden_sizes:
            raise ValidationError("hidden_sizes must be non-empty", context={})
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.trunk = _mlp(obs_dim, hidden_sizes, activation=nn.Tanh)
        self.mean_head = nn.Linear(hidden_sizes[-1], action_dim)
        # Small mean-head gain so the initial policy is near-uniform-ish (stable start).
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.zeros_(self.mean_head.bias)
        # State-independent log-std, initialized so the initial std ~ 1 (broad exploration).
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def distribution(self, obs: torch.Tensor) -> torch.distributions.Normal:
        """Return the diagonal-Gaussian action distribution for a batch of observations."""
        if obs.dim() != 2 or obs.shape[1] != self.obs_dim:
            raise ValidationError(
                "obs must have shape (batch, obs_dim)",
                context={"shape": tuple(obs.shape), "obs_dim": self.obs_dim},
            )
        mean = self.mean_head(self.trunk(obs))
        log_std = torch.clamp(self.log_std, _LOG_STD_MIN, _LOG_STD_MAX)
        std = torch.exp(log_std).expand_as(mean)
        return torch.distributions.Normal(mean, std)

    def forward(self, obs: torch.Tensor) -> torch.distributions.Normal:
        """Alias for :meth:`distribution` (nn.Module convention)."""
        return self.distribution(obs)

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(log_prob, entropy)`` for given observations and actions.

        Log-probabilities and entropy are summed over the action dimensions (a diagonal
        Gaussian factorizes), giving per-sample scalars of shape ``(batch,)``.
        """
        dist = self.distribution(obs)
        # torch.distributions methods are untyped in the stubs; the shapes are asserted by
        # the tests (per-sample scalars after summing over the action dimension).
        log_prob = dist.log_prob(actions).sum(dim=-1)  # type: ignore[no-untyped-call]
        entropy = dist.entropy().sum(dim=-1)  # type: ignore[no-untyped-call]
        return log_prob, entropy


class DistributionalCritic(nn.Module):
    """Quantile value network predicting ``K`` quantiles of the return distribution.

    Parameters
    ----------
    obs_dim:
        Observation dimension.
    n_quantiles:
        Number of value-distribution quantiles ``K`` (``>= 1``).
    hidden_sizes:
        Hidden-layer widths.
    cvar_alpha:
        Tail probability for the CVaR estimate (e.g. 0.1 => mean of the lowest 10% quantiles).
    """

    def __init__(
        self,
        *,
        obs_dim: int,
        n_quantiles: int = 32,
        hidden_sizes: tuple[int, ...] = (128, 128),
        cvar_alpha: float = 0.1,
    ) -> None:
        super().__init__()
        if obs_dim < 1:
            raise ValidationError("obs_dim must be >= 1", context={})
        if n_quantiles < 1:
            raise ValidationError("n_quantiles must be >= 1", context={})
        if not 0.0 < cvar_alpha <= 1.0:
            raise ValidationError(
                "cvar_alpha must lie in (0, 1]", context={"cvar_alpha": cvar_alpha}
            )
        self.obs_dim = obs_dim
        self.n_quantiles = n_quantiles
        self.cvar_alpha = cvar_alpha
        self.trunk = _mlp(obs_dim, hidden_sizes, activation=nn.Tanh)
        self.quantile_head = nn.Linear(hidden_sizes[-1], n_quantiles)
        nn.init.orthogonal_(self.quantile_head.weight, gain=1.0)
        nn.init.zeros_(self.quantile_head.bias)
        # Number of quantiles forming the CVaR tail (at least one).
        self._n_tail = max(1, round(cvar_alpha * n_quantiles))

    def quantiles(self, obs: torch.Tensor) -> torch.Tensor:
        """Return predicted return quantiles, shape ``(batch, n_quantiles)``, sorted ascending.

        The raw head outputs are sorted so the returned quantiles are monotone (a valid
        inverse-CDF), which keeps the mean/CVaR reductions well-defined regardless of training
        state.
        """
        if obs.dim() != 2 or obs.shape[1] != self.obs_dim:
            raise ValidationError(
                "obs must have shape (batch, obs_dim)",
                context={"shape": tuple(obs.shape), "obs_dim": self.obs_dim},
            )
        raw = self.quantile_head(self.trunk(obs))
        sorted_q, _ = torch.sort(raw, dim=-1)
        return sorted_q

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        """Return the scalar value baseline ``V(s) = mean(quantiles)``, shape ``(batch,)``."""
        return self.quantiles(obs).mean(dim=-1)

    def cvar(self, obs: torch.Tensor) -> torch.Tensor:
        """Return the CVaR (mean of the lowest ``alpha`` quantiles), shape ``(batch,)``.

        A lower (more negative) CVaR means a heavier left tail of returns -- the quantity a
        short-gamma book must watch. Used for tail-risk monitoring and risk-sensitive shaping.
        """
        return self.quantiles(obs)[:, : self._n_tail].mean(dim=-1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Alias for :meth:`quantiles`."""
        return self.quantiles(obs)
