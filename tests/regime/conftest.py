"""Shared fixtures for regime tests: synthetic data with known hidden states."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest


@dataclass(frozen=True)
class SyntheticRegimes:
    """A synthetic regime sequence with observations and ground-truth states."""

    observations: np.ndarray  # (T, 1) log-variance-like feature
    states: np.ndarray  # (T,) true state indices (0=low, 1=mid, 2=high)
    means: np.ndarray  # (3, 1) true state means
    transition: np.ndarray  # (3, 3)


def make_regimes(
    *,
    n: int = 3000,
    means: tuple[float, ...] = (-6.0, -4.5, -3.0),
    variances: tuple[float, ...] = (0.1, 0.15, 0.25),
    persistence: float = 0.95,
    seed: int = 0,
) -> SyntheticRegimes:
    """Generate a sticky 3-state Gaussian regime sequence with known parameters."""
    rng = np.random.default_rng(seed)
    k = len(means)
    off = (1.0 - persistence) / (k - 1)
    transition = np.full((k, k), off)
    np.fill_diagonal(transition, persistence)
    states = np.zeros(n, dtype=int)
    for t in range(1, n):
        states[t] = rng.choice(k, p=transition[states[t - 1]])
    obs = np.array([[rng.normal(means[s], np.sqrt(variances[s]))] for s in states], dtype=float)
    return SyntheticRegimes(
        observations=obs,
        states=states,
        means=np.array(means).reshape(-1, 1),
        transition=transition,
    )


@pytest.fixture(scope="module")
def regimes() -> SyntheticRegimes:
    """A shared synthetic regime dataset."""
    return make_regimes()
