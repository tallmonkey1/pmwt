r"""Standard-normal generation with variance-reduction support.

The Monte-Carlo accuracy of the simulator is governed by how its driving Gaussian noise is
generated (SPEC §2.3). This module centralizes that concern and offers two
variance-reduction techniques that compose:

* **Antithetic variates** — for every primal draw :math:`x` we also use :math:`-x`. Because
  the standard normal is symmetric, this exactly halves the variance of any estimator of an
  odd functional and reduces variance for many even ones, at no extra cost.
* **Quasi-Monte-Carlo (Sobol)** — a low-discrepancy sequence transformed to normals via the
  inverse CDF. Sobol points cover the unit hypercube far more uniformly than pseudo-random
  points, typically improving convergence from :math:`O(N^{-1/2})` toward :math:`O(N^{-1})`
  for smooth integrands.

Both techniques are *optional* and selected explicitly so that a plain, well-understood
pseudo-random Monte-Carlo run is always available as a baseline for validation.
"""

from __future__ import annotations

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray
from scipy.stats import norm, qmc

from ...core.errors import ValidationError

__all__ = ["draw_standard_normals"]

# Clip Sobol uniforms away from {0, 1} before the inverse-CDF transform so that the
# resulting normals are always finite (the normal inverse-CDF diverges at the endpoints).
_UNIFORM_EPS = 1e-12


def draw_standard_normals(
    n_paths: int,
    dim: int,
    *,
    rng: Generator,
    antithetic: bool = False,
    quasi_random: bool = False,
) -> NDArray[np.float64]:
    r"""Return an ``(n_paths, dim)`` array of standard-normal variates.

    Parameters
    ----------
    n_paths:
        Number of paths (rows). Must be positive. When ``antithetic`` is enabled and
        ``n_paths`` is odd, the final row is generated independently so the exact count is
        always honoured.
    dim:
        Number of independent normals per path (columns). Must be positive.
    rng:
        NumPy generator supplying randomness (also seeds the scrambled Sobol engine, so QMC
        runs remain reproducible).
    antithetic:
        If True, the second half of the rows are the negation of the first half.
    quasi_random:
        If True, use a scrambled Sobol sequence transformed by the normal inverse CDF
        instead of pseudo-random draws.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_paths, dim)`` of finite ``float64`` standard normals.
    """
    if n_paths < 1:
        raise ValidationError("n_paths must be >= 1", context={"n_paths": n_paths})
    if dim < 1:
        raise ValidationError("dim must be >= 1", context={"dim": dim})

    if not antithetic:
        return _draw_block(n_paths, dim, rng=rng, quasi_random=quasi_random)

    # Antithetic: generate ceil(n/2) primal rows, mirror them, then trim to n_paths.
    half = (n_paths + 1) // 2
    primal = _draw_block(half, dim, rng=rng, quasi_random=quasi_random)
    combined = np.empty((2 * half, dim), dtype=np.float64)
    combined[:half] = primal
    combined[half:] = -primal
    return combined[:n_paths]


def _draw_block(n: int, dim: int, *, rng: Generator, quasi_random: bool) -> NDArray[np.float64]:
    """Draw an ``(n, dim)`` block of standard normals (pseudo-random or Sobol)."""
    if not quasi_random:
        return rng.standard_normal(size=(n, dim))

    # Seed the Sobol engine from the provided generator to preserve reproducibility while
    # keeping the QMC stream independent of subsequent ``rng`` usage.
    seed = int(rng.integers(0, 2**63 - 1))
    engine = qmc.Sobol(d=dim, scramble=True, seed=seed)
    uniforms = engine.random(n)
    np.clip(uniforms, _UNIFORM_EPS, 1.0 - _UNIFORM_EPS, out=uniforms)
    return np.asarray(norm.ppf(uniforms), dtype=np.float64)
