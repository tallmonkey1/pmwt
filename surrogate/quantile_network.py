r"""Monotone quantile network for the distribution surrogate (SPEC §2.5).

A feed-forward network maps the (scaled) feature vector to the terminal log-return
quantiles on a fixed probability grid. The key design property is **monotonicity by
construction**: instead of predicting the ``Q`` quantile values directly (which can cross,
violating the definition of a quantile function), the network predicts

* ``base`` -- the value of the lowest quantile, and
* ``Q - 1`` raw increments passed through ``softplus`` (hence strictly positive),

and the quantiles are the cumulative sum ``base, base + d_1, base + d_1 + d_2, ...``. This
guarantees non-crossing quantiles for *every* input, with no penalty term, no post-hoc
sorting, and no failure mode -- exactly the kind of structural guarantee an institutional
model needs (you cannot accidentally ship crossing quantiles).

The architecture is intentionally small and fully specified: an MLP with configurable
hidden widths, GELU activations, and layer norm for training stability. It is deterministic
given a seed.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from ..core.errors import ValidationError

__all__ = ["MonotoneQuantileNetwork"]


class MonotoneQuantileNetwork(nn.Module):
    """MLP producing non-crossing quantiles via a base value plus positive increments.

    Parameters
    ----------
    n_features:
        Input feature dimension.
    n_quantiles:
        Number of quantile levels ``Q`` to output (``>= 2``).
    hidden_sizes:
        Widths of the hidden layers.
    dropout:
        Dropout probability applied after each hidden activation (0 disables it).
    """

    def __init__(
        self,
        *,
        n_features: int,
        n_quantiles: int,
        hidden_sizes: tuple[int, ...] = (128, 128),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if n_features < 1:
            raise ValidationError("n_features must be >= 1", context={"n_features": n_features})
        if n_quantiles < 2:
            raise ValidationError("n_quantiles must be >= 2", context={"n_quantiles": n_quantiles})
        if not hidden_sizes:
            raise ValidationError("hidden_sizes must be non-empty", context={})
        if not 0.0 <= dropout < 1.0:
            raise ValidationError("dropout must lie in [0, 1)", context={"dropout": dropout})

        self.n_features = n_features
        self.n_quantiles = n_quantiles

        layers: list[nn.Module] = []
        in_dim = n_features
        for width in hidden_sizes:
            if width < 1:
                raise ValidationError("hidden widths must be >= 1", context={"width": width})
            layers.append(nn.Linear(in_dim, width))
            layers.append(nn.LayerNorm(width))
            layers.append(nn.GELU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = width
        self.backbone = nn.Sequential(*layers)

        # Two heads: the base (lowest) quantile and the positive increments.
        self.base_head = nn.Linear(in_dim, 1)
        self.increment_head = nn.Linear(in_dim, n_quantiles - 1)
        # Softplus turns raw increments into strictly positive gaps between quantiles.
        self.softplus = nn.Softplus()

        self._initialize_output_heads()

    def _initialize_output_heads(self) -> None:
        r"""Initialize the output heads so predictions start as a narrow, sane distribution.

        Without this, the default-initialized increment head produces ``softplus(0) ~ 0.69``
        per gap; with ``Q`` quantiles the implied distribution spans ``~0.69 * (Q - 1)``,
        which is absurdly wide (tens of log-return units) and forces the optimizer to spend
        many epochs just shrinking it. We instead bias the increment head so the initial
        gaps are ``softplus(b) ~ total_width / (Q - 1)`` for a realistic ``total_width`` of
        order one, and zero the base bias (centred distribution). Weights start near zero so
        the initial output is feature-independent and stable.
        """
        target_total_width = 1.0
        n_gaps = self.n_quantiles - 1
        target_gap = target_total_width / n_gaps
        # Invert softplus: b such that softplus(b) = target_gap  =>  b = log(exp(gap) - 1).
        bias = math.log(math.expm1(target_gap)) if target_gap > 0 else 0.0
        with torch.no_grad():
            nn.init.zeros_(self.increment_head.weight)
            nn.init.constant_(self.increment_head.bias, bias)
            nn.init.zeros_(self.base_head.weight)
            nn.init.constant_(self.base_head.bias, -0.5 * target_total_width)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return predicted quantiles of shape ``(batch, n_quantiles)``, sorted ascending.

        Parameters
        ----------
        features:
            Float tensor of shape ``(batch, n_features)``.
        """
        if features.dim() != 2 or features.shape[1] != self.n_features:
            raise ValidationError(
                "features must have shape (batch, n_features)",
                context={"got": tuple(features.shape), "n_features": self.n_features},
            )
        hidden = self.backbone(features)
        base = self.base_head(hidden)  # (batch, 1)
        # Positive increments; a small floor keeps strict monotonicity away from zero gaps.
        increments = self.softplus(self.increment_head(hidden)) + 1e-6  # (batch, Q-1)
        cumulative = torch.cumsum(increments, dim=1)  # (batch, Q-1)
        quantiles = torch.cat([base, base + cumulative], dim=1)  # (batch, Q)
        return quantiles
