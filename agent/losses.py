r"""PPO and distributional-critic loss functions (SPEC §4.2).

Two losses, both implemented from scratch and unit-tested against their defining properties:

* :func:`ppo_clip_loss` -- the PPO-clip surrogate (Schulman et al., 2017):

  .. math::

      L^{CLIP} = -\,\mathbb{E}\big[\min(\rho_t A_t,\;
                  \mathrm{clip}(\rho_t, 1-\epsilon, 1+\epsilon)\, A_t)\big],

  where :math:`\rho_t = \exp(\log\pi_\theta(a_t|s_t) - \log\pi_{\theta_\text{old}})` is the
  importance ratio. The clip removes the incentive to move the policy too far in one update,
  which is what makes PPO stable.

* :func:`quantile_huber_loss` -- the quantile-regression Huber loss (Dabney et al., 2018) for
  training the distributional critic toward the return targets:

  .. math::

      \mathcal{L} = \frac{1}{K}\sum_i \big|\tau_i - \mathbb{1}\{u_i < 0\}\big|\,
                    \mathcal{H}_\kappa(u_i), \qquad u_i = R - \theta_i,

  with :math:`\mathcal{H}_\kappa` the Huber loss. This asymmetric weighting is what makes each
  predicted quantile converge to the correct quantile of the return distribution.
"""

from __future__ import annotations

import torch

from ..core.errors import ValidationError

__all__ = ["ppo_clip_loss", "quantile_huber_loss"]


def ppo_clip_loss(
    *,
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(policy_loss, clip_fraction)`` for the PPO-clip surrogate.

    Parameters
    ----------
    new_log_probs, old_log_probs:
        Per-sample log-probabilities under the current and behaviour policies, shape ``(N,)``.
    advantages:
        Per-sample advantages, shape ``(N,)`` (already normalized by the caller).
    clip_epsilon:
        PPO clip range ``epsilon`` (e.g. 0.2).

    Returns
    -------
    (policy_loss, clip_fraction):
        The scalar surrogate loss (to minimize) and the fraction of samples whose ratio was
        clipped (a diagnostic for update-size health).
    """
    if not 0.0 < clip_epsilon < 1.0:
        raise ValidationError(
            "clip_epsilon must lie in (0, 1)", context={"clip_epsilon": clip_epsilon}
        )
    if new_log_probs.shape != old_log_probs.shape or new_log_probs.shape != advantages.shape:
        raise ValidationError(
            "log-prob and advantage shapes must match",
            context={
                "new": tuple(new_log_probs.shape),
                "old": tuple(old_log_probs.shape),
                "adv": tuple(advantages.shape),
            },
        )
    ratio = torch.exp(new_log_probs - old_log_probs)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    policy_loss = -torch.mean(torch.minimum(unclipped, clipped))
    with torch.no_grad():
        clip_fraction = torch.mean((torch.abs(ratio - 1.0) > clip_epsilon).float())
    return policy_loss, clip_fraction


def quantile_huber_loss(
    *,
    predicted_quantiles: torch.Tensor,
    target_returns: torch.Tensor,
    taus: torch.Tensor,
    kappa: float = 1.0,
) -> torch.Tensor:
    r"""Return the mean quantile-regression Huber loss for the distributional critic.

    Parameters
    ----------
    predicted_quantiles:
        The critic's predicted quantiles, shape ``(N, K)``.
    target_returns:
        Scalar return targets per sample, shape ``(N,)``. Each target is regressed by all ``K``
        quantiles (the standard QR loss broadcasts the scalar target across quantiles).
    taus:
        The ``K`` quantile fractions, shape ``(K,)``, each in ``(0, 1)``.
    kappa:
        Huber threshold ``kappa`` (``> 0``). ``kappa -> 0`` recovers the (non-smooth) quantile
        loss; ``kappa = 1`` is the common default.

    Returns
    -------
    torch.Tensor
        Scalar loss (mean over samples and quantiles).
    """
    if predicted_quantiles.dim() != 2:
        raise ValidationError(
            "predicted_quantiles must be 2-D (N, K)",
            context={"shape": tuple(predicted_quantiles.shape)},
        )
    n, k = predicted_quantiles.shape
    if target_returns.shape != (n,):
        raise ValidationError(
            "target_returns must have shape (N,)",
            context={"shape": tuple(target_returns.shape), "n": n},
        )
    if taus.shape != (k,):
        raise ValidationError(
            "taus must have shape (K,)", context={"shape": tuple(taus.shape), "k": k}
        )
    if kappa <= 0.0:
        raise ValidationError("kappa must be positive", context={"kappa": kappa})

    # u[n, i] = target_n - theta_{n,i}  (pairwise temporal-difference per quantile).
    target = target_returns.unsqueeze(1)  # (N, 1)
    u = target - predicted_quantiles  # (N, K)

    abs_u = torch.abs(u)
    huber = torch.where(abs_u <= kappa, 0.5 * u.pow(2), kappa * (abs_u - 0.5 * kappa))
    # Asymmetric quantile weight |tau - 1{u < 0}|.
    tau_row = taus.unsqueeze(0)  # (1, K)
    weight = torch.abs(tau_row - (u.detach() < 0.0).float())
    loss = weight * huber / kappa
    return loss.mean()
