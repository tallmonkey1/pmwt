r"""Generalized Advantage Estimation (GAE) and returns (SPEC §4.2).

GAE (Schulman et al., 2016) is the variance-reduced advantage estimator PPO trains on:

.. math::

    \delta_t = r_t + \gamma\, V(s_{t+1})\,(1 - d_t) - V(s_t), \qquad
    A_t = \sum_{l \ge 0} (\gamma\lambda)^l\, \delta_{t+l},

computed by the backward recursion :math:`A_t = \delta_t + \gamma\lambda\,(1-d_t)\,A_{t+1}`.
The value target (return) is :math:`R_t = A_t + V(s_t)`.

This module is pure NumPy, deterministic, and depends on nothing else, so it can be tested in
isolation against hand-computed values -- which is exactly how its correctness is verified
(getting the ``(1 - done)`` masking or the bootstrap term wrong is the classic, silent RL
bug). Episode boundaries are honoured via the per-step ``dones`` mask and a per-step
``next_value`` bootstrap, so it is correct for both truncated and terminated episodes.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..core.errors import ValidationError
from ..core.validation import check_unit_interval

__all__ = ["compute_gae"]


def compute_gae(
    rewards: NDArray[np.float64],
    values: NDArray[np.float64],
    dones: NDArray[np.bool_],
    next_values: NDArray[np.float64],
    *,
    gamma: float,
    lam: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Return ``(advantages, returns)`` via the GAE backward recursion.

    Parameters
    ----------
    rewards:
        Per-step rewards, shape ``(T,)``.
    values:
        Critic value estimates ``V(s_t)`` for each step, shape ``(T,)``.
    dones:
        Per-step terminal flags ``d_t`` (True if ``s_{t+1}`` is terminal / the episode ended
        with no bootstrap), shape ``(T,)``.
    next_values:
        Bootstrap values ``V(s_{t+1})`` for each step, shape ``(T,)``. For a terminal step the
        corresponding entry is ignored (masked by ``1 - done``); for a truncation it should be
        the critic's estimate of the next state so the return is correctly bootstrapped.
    gamma:
        Discount factor in ``[0, 1]``.
    lam:
        GAE lambda in ``[0, 1]``.

    Returns
    -------
    (advantages, returns):
        Both shape ``(T,)``. ``returns = advantages + values``.
    """
    r = np.asarray(rewards, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    nv = np.asarray(next_values, dtype=np.float64)
    d = np.asarray(dones, dtype=np.bool_)
    check_unit_interval(gamma, name="gamma")
    check_unit_interval(lam, name="lam")
    n = r.shape[0]
    if not (v.shape == nv.shape == d.shape == (n,)) or n == 0:
        raise ValidationError(
            "rewards, values, next_values, dones must be 1-D of equal non-zero length",
            context={
                "rewards": r.shape,
                "values": v.shape,
                "next_values": nv.shape,
                "dones": d.shape,
            },
        )
    if not (np.all(np.isfinite(r)) and np.all(np.isfinite(v)) and np.all(np.isfinite(nv))):
        raise ValidationError("rewards/values/next_values contain non-finite entries", context={})

    advantages = np.zeros(n, dtype=np.float64)
    not_done = (~d).astype(np.float64)
    gae = 0.0
    for t in range(n - 1, -1, -1):
        delta = r[t] + gamma * nv[t] * not_done[t] - v[t]
        gae = delta + gamma * lam * not_done[t] * gae
        advantages[t] = gae
    returns = advantages + v
    return advantages, returns
