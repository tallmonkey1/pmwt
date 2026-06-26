"""Helper critic: a meta-controller that learns the alpha vector.

The main PPO-Transformer agent learns *within* a market characterised by a
:class:`MarketAlpha` vector. The helper critic agent is the outer-loop
meta-controller: it observes the main agent's *internal diagnostics* (every
internal feature is in ``[0, 1]``) and proposes an alpha that drives them toward
``1``.

The optimisation target is precisely: **find alpha such that every internal
feature of the main agent is close to 1**. With ``alpha = ones()`` the market is
fully calm (near-Black-Scholes, no noise, no jumps, no shocks) and the main
agent's features naturally converge toward their ideal values. The helper
critic's job is to find the *minimum-noise* alpha at which the main agent is
*still* fully profitable.

Design
------
* **Small tabular bandit** over a discrete alpha lattice (default ``K = 8``
  alphas spanning ``zeros()`` to ``ones()``). A simple Q-update picks the next
  alpha to evaluate; the reward is the sum of internal features produced by the
  main agent under that alpha (each in ``[0, 1]``).
* **Reproducible**: the critic's RNG is injected; the alpha lattice is fixed.
* **Bounded update**: every alpha returned by the helper critic is clipped to
  ``[0, 1]`` per component before being passed to the main environment.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..core.errors import ValidationError
from ..core.market_alpha import (
    DEFAULT_ALPHA_DIM,
    MarketAlpha,
    alpha_components,
)
from ..core.logging import get_logger

__all__ = [
    "DEFAULT_HELPER_LATTICE_SIZE",
    "HelperCritic",
    "HelperCriticConfig",
    "alpha_lattice",
    "feature_score_from_components",
]

_logger = get_logger(__name__)

DEFAULT_HELPER_LATTICE_SIZE: int = 8


def alpha_lattice(*, size: int = DEFAULT_HELPER_LATTICE_SIZE) -> tuple[MarketAlpha, ...]:
    """Return a deterministic, evenly-spaced lattice of ``size`` alphas in [0, 1]^K.

    The lattice spans the diagonal from ``MarketAlpha.zeros()`` (alpha=0, maximally
    rough) to ``MarketAlpha.ones()`` (alpha=1, maximally calm). Each alpha is
    right-padded with ones so a scalar alpha and a multi-dimensional alpha are
    represented consistently in the lattice.
    """
    if size < 1:
        raise ValidationError("lattice size must be >= 1", context={"size": size})
    if size == 1:
        return (MarketAlpha.ones(),)
    return tuple(
        MarketAlpha(features=tuple([t] + [1.0] * (DEFAULT_ALPHA_DIM - 1)))
        for t in np.linspace(0.0, 1.0, size, dtype=np.float64)
    )


@dataclass(frozen=True, slots=True)
class HelperCriticConfig:
    """Hyper-parameters for the helper critic."""

    lattice_size: int = DEFAULT_HELPER_LATTICE_SIZE
    learning_rate: float = 0.1
    exploration_rate: float = 0.1
    seed: int = 0

    def __post_init__(self) -> None:
        if self.lattice_size < 1:
            raise ValidationError(
                "lattice_size must be >= 1", context={"lattice_size": self.lattice_size}
            )
        if not 0.0 <= self.exploration_rate <= 1.0:
            raise ValidationError(
                "exploration_rate must be in [0, 1]",
                context={"exploration_rate": self.exploration_rate},
            )
        if not 0.0 < self.learning_rate <= 1.0:
            raise ValidationError(
                "learning_rate must be in (0, 1]",
                context={"learning_rate": self.learning_rate},
            )


class HelperCritic:
    """A tabular bandit over the alpha lattice, updated by reward = sum of internals."""

    def __init__(self, *, config: HelperCriticConfig | None = None) -> None:
        self._config = config or HelperCriticConfig()
        self._lattice = alpha_lattice(size=self._config.lattice_size)
        self._q: NDArray[np.float64] = np.zeros(len(self._lattice), dtype=np.float64)
        self._n_pulls: NDArray[np.int64] = np.zeros(len(self._lattice), dtype=np.int64)
        self._rng = np.random.default_rng(self._config.seed)
        self._last_alpha: MarketAlpha | None = None

    @property
    def lattice(self) -> tuple[MarketAlpha, ...]:
        """The candidate alpha values the helper critic can choose between."""
        return self._lattice

    @property
    def q_values(self) -> NDArray[np.float64]:
        """The current estimate of each lattice alpha's expected sum-of-internals."""
        return self._q.copy()

    def best_alpha(self) -> MarketAlpha:
        """Return the lattice alpha with the highest current Q-value."""
        if not np.any(self._n_pulls):
            return MarketAlpha.ones()
        best_idx = int(np.argmax(self._q))
        return self._lattice[best_idx]

    def select_alpha(
        self, *, features: NDArray[np.float64] | None = None
    ) -> MarketAlpha:
        """Pick the next alpha to evaluate under (epsilon-greedy on the lattice)."""
        if features is not None and features.ndim != 1:
            raise ValidationError(
                "features must be 1-D", context={"shape": features.shape}
            )
        if self._rng.random() < self._config.exploration_rate:
            idx = int(self._rng.integers(0, len(self._lattice)))
        else:
            best_idx = int(np.argmax(self._q))
            candidates = np.where(self._q >= self._q[best_idx] - 1e-6)[0]
            idx = int(self._rng.choice(candidates))
        alpha = self._lattice[idx].clipped()
        self._last_alpha = alpha
        return alpha

    def update(self, *, features: NDArray[np.float64]) -> float:
        """Record the observed sum-of-internals for the last chosen alpha and update Q."""
        if self._last_alpha is None:
            raise ValidationError(
                "select_alpha must be called before update", context={}
            )
        if features.ndim != 1:
            raise ValidationError(
                "features must be 1-D", context={"shape": features.shape}
            )
        clipped = np.clip(np.asarray(features, dtype=np.float64), 0.0, 1.0)
        reward = float(clipped.sum())
        idx = self._find_lattice_index(self._last_alpha)
        lr = float(self._config.learning_rate)
        prev_q = float(self._q[idx])
        self._q[idx] = (1.0 - lr) * prev_q + lr * reward
        self._n_pulls[idx] += 1
        _logger.debug(
            "helper_critic_update",
            extra={
                "alpha": str(self._last_alpha),
                "reward": reward,
                "q_after": float(self._q[idx]),
                "pulls": int(self._n_pulls[idx]),
            },
        )
        return reward

    def _find_lattice_index(self, alpha: MarketAlpha) -> int:
        """Return the index of ``alpha`` in the lattice (exact equality expected)."""
        target = alpha.padded().as_array()
        for i, candidate in enumerate(self._lattice):
            if np.array_equal(candidate.padded().as_array(), target):
                return i
        dists = [
            float(np.linalg.norm(c.padded().as_array() - target)) for c in self._lattice
        ]
        return int(np.argmin(dists))

    def diagnostics(self) -> dict[str, float]:
        """Return a snapshot of the helper critic's learning state for logging."""
        return {
            "best_q": float(self._q.max()) if self._n_pulls.any() else 0.0,
            "mean_q": float(self._q.mean()),
            "total_pulls": int(self._n_pulls.sum()),
            "n_explored": int(np.count_nonzero(self._n_pulls)),
            "n_lattice": len(self._lattice),
        }


def feature_score_from_components(
    *,
    win_probability: float,
    cvar_safety: float,
    profit_factor: float,
    stability: float,
    margin_safety: float,
    drawdown_safety: float,
) -> NDArray[np.float64]:
    """Pack a 6-tuple of internal diagnostics (each already in ``[0, 1]``) into the
    feature vector the helper critic expects.

    Each component is clipped to ``[0, 1]`` for safety.
    """
    return np.clip(
        np.asarray(
            [
                float(win_probability),
                float(cvar_safety),
                float(profit_factor),
                float(stability),
                float(margin_safety),
                float(drawdown_safety),
            ],
            dtype=np.float64,
        ),
        0.0,
        1.0,
    )
