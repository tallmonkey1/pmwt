r"""Quantile (pinball) loss for training the distribution surrogate.

The surrogate is trained by minimizing the **pinball / quantile loss**, the proper scoring
rule whose minimizer is the true quantile. For a target ``y``, prediction ``q`` at level
``tau``,

.. math::

    L_\tau(y, q) = \max\big(\tau (y - q),\; (\tau - 1)(y - q)\big).

Averaging over a grid of quantile levels yields a consistent estimator of the whole
quantile function and, in the limit of a dense grid, approximates the Continuous Ranked
Probability Score (CRPS) -- the standard distributional accuracy metric (SPEC §2.5).

This module is deliberately framework-thin: it operates on torch tensors and contains no
training-loop logic, so it is trivially unit-testable against the known closed-form
optimum.
"""

from __future__ import annotations

import torch

from ..core.errors import ValidationError

__all__ = ["pinball_loss"]


def pinball_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    quantile_levels: torch.Tensor,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    r"""Return the pinball loss between predicted and target quantiles.

    Parameters
    ----------
    predictions:
        Predicted quantiles, shape ``(batch, Q)``.
    targets:
        Target quantile values, shape ``(batch, Q)`` (the empirical MC quantiles).
    quantile_levels:
        The probability levels ``tau``, shape ``(Q,)``, each in ``(0, 1)``.
    reduction:
        ``"mean"`` (default), ``"sum"``, or ``"none"`` (returns the per-element loss).

    Returns
    -------
    torch.Tensor
        Scalar loss for ``mean``/``sum``; tensor of shape ``(batch, Q)`` for ``none``.
    """
    if predictions.shape != targets.shape:
        raise ValidationError(
            "predictions and targets must share shape",
            context={"pred": tuple(predictions.shape), "target": tuple(targets.shape)},
        )
    if predictions.dim() != 2:
        raise ValidationError(
            "predictions must be 2-D (batch, Q)", context={"ndim": predictions.dim()}
        )
    if quantile_levels.dim() != 1 or quantile_levels.shape[0] != predictions.shape[1]:
        raise ValidationError(
            "quantile_levels must be 1-D of length Q",
            context={"got": tuple(quantile_levels.shape), "q": predictions.shape[1]},
        )
    if reduction not in {"mean", "sum", "none"}:
        raise ValidationError("reduction must be mean|sum|none", context={"reduction": reduction})

    errors = targets - predictions  # (batch, Q)
    levels = quantile_levels.unsqueeze(0)  # (1, Q)
    loss = torch.maximum(levels * errors, (levels - 1.0) * errors)

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss
