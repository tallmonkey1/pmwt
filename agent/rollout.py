r"""On-policy rollout buffer for PPO (SPEC §4.2).

Collects a fixed-length batch of transitions from the environment, then computes GAE
advantages and value-distribution targets and yields shuffled minibatches for the PPO update.
Keeping this as a small, explicit, NumPy-backed structure (rather than hiding it in a
framework) makes the data flow auditable: the exact tensors PPO trains on are visible and
testable.

Storage is pre-allocated to a known capacity for determinism and speed; the buffer asserts it
is full before computing advantages, so a partial/mis-sized rollout cannot silently produce a
bad update.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..core.errors import ValidationError
from .gae import compute_gae

__all__ = ["RolloutBatch", "RolloutBuffer"]


@dataclass(frozen=True, slots=True)
class RolloutBatch:
    """A minibatch of transitions for the PPO update (float32 tensors as NumPy arrays)."""

    observations: NDArray[np.float32]
    actions: NDArray[np.float32]
    old_log_probs: NDArray[np.float32]
    advantages: NDArray[np.float32]
    returns: NDArray[np.float32]
    values: NDArray[np.float32]


class RolloutBuffer:
    """Fixed-capacity on-policy transition store with GAE post-processing.

    Parameters
    ----------
    capacity:
        Number of transitions per rollout.
    obs_dim, action_dim:
        Dimensions used to pre-allocate storage.
    gamma, lam:
        Discount and GAE lambda.
    """

    def __init__(
        self, *, capacity: int, obs_dim: int, action_dim: int, gamma: float, lam: float
    ) -> None:
        if capacity < 1 or obs_dim < 1 or action_dim < 1:
            raise ValidationError("capacity/obs_dim/action_dim must be >= 1", context={})
        self._capacity = capacity
        self._obs_dim = obs_dim
        self._action_dim = action_dim
        self._gamma = gamma
        self._lam = lam

        self._obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self._log_probs = np.zeros(capacity, dtype=np.float32)
        self._rewards = np.zeros(capacity, dtype=np.float64)
        self._values = np.zeros(capacity, dtype=np.float64)
        self._next_values = np.zeros(capacity, dtype=np.float64)
        self._dones = np.zeros(capacity, dtype=np.bool_)

        self._pos = 0
        self._advantages: NDArray[np.float64] | None = None
        self._returns: NDArray[np.float64] | None = None

    @property
    def capacity(self) -> int:
        """Number of transitions the buffer holds."""
        return self._capacity

    @property
    def is_full(self) -> bool:
        """True once ``capacity`` transitions have been added."""
        return self._pos >= self._capacity

    def reset(self) -> None:
        """Clear the buffer for a new rollout."""
        self._pos = 0
        self._advantages = None
        self._returns = None

    def add(
        self,
        *,
        obs: NDArray[np.float32],
        action: NDArray[np.float32],
        log_prob: float,
        reward: float,
        value: float,
        next_value: float,
        done: bool,
    ) -> None:
        """Append one transition. Raises if the buffer is already full."""
        if self._pos >= self._capacity:
            raise ValidationError("rollout buffer is full", context={"capacity": self._capacity})
        i = self._pos
        self._obs[i] = obs
        self._actions[i] = action
        self._log_probs[i] = log_prob
        self._rewards[i] = reward
        self._values[i] = value
        self._next_values[i] = next_value
        self._dones[i] = done
        self._pos += 1

    def compute_advantages(self) -> None:
        """Compute GAE advantages and returns for the full rollout (must be full)."""
        if not self.is_full:
            raise ValidationError(
                "buffer must be full before computing advantages",
                context={"pos": self._pos, "capacity": self._capacity},
            )
        self._advantages, self._returns = compute_gae(
            self._rewards,
            self._values,
            self._dones,
            self._next_values,
            gamma=self._gamma,
            lam=self._lam,
        )

    def iter_minibatches(
        self, *, batch_size: int, rng: np.random.Generator, normalize_advantages: bool = True
    ) -> Iterator[RolloutBatch]:
        """Yield shuffled minibatches of the post-processed rollout.

        Advantages are (optionally) normalized to zero mean / unit std across the whole
        rollout before batching -- the standard PPO variance-reduction step. The final
        minibatch may be smaller than ``batch_size`` if the capacity is not a multiple of it.
        """
        if self._advantages is None or self._returns is None:
            raise ValidationError("call compute_advantages before iterating", context={})
        if batch_size < 1:
            raise ValidationError("batch_size must be >= 1", context={})

        advantages = self._advantages.copy()
        if normalize_advantages:
            std = advantages.std()
            if std > 1e-8:
                advantages = (advantages - advantages.mean()) / std
            else:
                advantages = advantages - advantages.mean()

        indices = rng.permutation(self._capacity)
        for start in range(0, self._capacity, batch_size):
            idx = indices[start : start + batch_size]
            yield RolloutBatch(
                observations=self._obs[idx],
                actions=self._actions[idx],
                old_log_probs=self._log_probs[idx],
                advantages=advantages[idx].astype(np.float32),
                returns=self._returns[idx].astype(np.float32),
                values=self._values[idx].astype(np.float32),
            )
