r"""Volterra-kernel mathematics for the rBergomi model.

The rough driver is the Volterra (Riemann-Liouville) process

.. math::

    Y_t = \sqrt{2H}\int_0^t (t-s)^{H-1/2}\, dZ_s,
    \qquad \mathbb{E}[Y_t^2] = t^{2H}.

This module provides the two ingredients needed to simulate it:

1. **Closed-form Gaussian covariances** (:func:`volterra_autocovariance`,
   :func:`cross_covariance`) used by the *exact* Cholesky simulator. The auto-covariance
   has the closed form, for :math:`s \le t`,

   .. math::

       \mathbb{E}[Y_s Y_t] = \frac{2H}{H+\tfrac12}\, s^{H+1/2}\, t^{H-1/2}\,
       {}_2F_1\!\Big(\tfrac12 - H,\, 1;\, H + \tfrac32;\, \tfrac{s}{t}\Big),

   which can be verified to collapse to :math:`t^{2H}` on the diagonal (``s = t``). The
   cross-covariance with the driving Brownian motion :math:`Z` is

   .. math::

       \mathbb{E}[Y_t Z_s] = \frac{\sqrt{2H}}{H+\tfrac12}
       \Big(t^{H+1/2} - (t - \min(s,t))^{H+1/2}\Big).

2. **Hybrid-scheme coefficients** (:func:`hybrid_discrete_covariance`,
   :func:`hybrid_weights`) for the production simulator of Bennedsen, Lunde & Pakkanen
   (2017) with :math:`\kappa = 1`.

References
----------
* Bayer, Friz, Gatheral (2016), *Pricing under rough volatility*.
* Bennedsen, Lunde, Pakkanen (2017), *Hybrid scheme for Brownian semistationary processes*.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import hyp2f1

from ...core.errors import ValidationError
from ...core.validation import check_in_range, check_positive

__all__ = [
    "cross_covariance",
    "hybrid_discrete_covariance",
    "hybrid_weights",
    "volterra_autocovariance",
]


def volterra_autocovariance(
    s: NDArray[np.float64], t: NDArray[np.float64], hurst: float
) -> NDArray[np.float64]:
    r"""Return :math:`\mathbb{E}[Y_s Y_t]` for the Volterra process, vectorized.

    Parameters
    ----------
    s, t:
        Broadcastable arrays of non-negative times in years.
    hurst:
        Hurst exponent :math:`H \in (0, 1/2)`.

    Returns
    -------
    numpy.ndarray
        The covariance evaluated elementwise over the broadcast of ``s`` and ``t``.

    Notes
    -----
    The formula is symmetric in its arguments; internally we order each pair so that the
    smaller time is the numerator of the hypergeometric argument (which lies in
    :math:`[0, 1]`, where :func:`scipy.special.hyp2f1` is well-behaved).
    """
    check_in_range(hurst, name="hurst", low=0.0, high=0.5, inclusive=False)
    s_arr = np.asarray(s, dtype=np.float64)
    t_arr = np.asarray(t, dtype=np.float64)
    if np.any(s_arr < 0.0) or np.any(t_arr < 0.0):
        raise ValidationError("times must be non-negative", context={})

    lower = np.minimum(s_arr, t_arr)
    upper = np.maximum(s_arr, t_arr)

    h = hurst
    coef = 2.0 * h / (h + 0.5)
    out = np.zeros(np.broadcast(lower, upper).shape, dtype=np.float64)

    # Where the smaller time is zero the covariance is zero (Y_0 = 0); avoid 0**negative.
    nonzero = (lower > 0.0) & (upper > 0.0)
    if np.any(nonzero):
        lo = np.broadcast_to(lower, out.shape)[nonzero]
        hi = np.broadcast_to(upper, out.shape)[nonzero]
        z = lo / hi
        hyp = hyp2f1(0.5 - h, 1.0, h + 1.5, z)
        out[nonzero] = coef * lo ** (h + 0.5) * hi ** (h - 0.5) * hyp
    return out


def cross_covariance(
    t: NDArray[np.float64], s: NDArray[np.float64], hurst: float
) -> NDArray[np.float64]:
    r"""Return :math:`\mathbb{E}[Y_t Z_s]`, the Volterra-vs-driver covariance, vectorized.

    Parameters
    ----------
    t:
        Time(s) at which the Volterra process is evaluated (years, non-negative).
    s:
        Time(s) at which the driving Brownian motion :math:`Z` is evaluated.
    hurst:
        Hurst exponent :math:`H \in (0, 1/2)`.
    """
    check_in_range(hurst, name="hurst", low=0.0, high=0.5, inclusive=False)
    t_arr = np.asarray(t, dtype=np.float64)
    s_arr = np.asarray(s, dtype=np.float64)
    if np.any(t_arr < 0.0) or np.any(s_arr < 0.0):
        raise ValidationError("times must be non-negative", context={})

    h = hurst
    coef = np.sqrt(2.0 * h) / (h + 0.5)
    m = np.minimum(s_arr, t_arr)
    result = coef * (t_arr ** (h + 0.5) - (t_arr - m) ** (h + 0.5))
    return np.asarray(result, dtype=np.float64)


def hybrid_discrete_covariance(dt: float, hurst: float) -> NDArray[np.float64]:
    r"""Return the :math:`2\times2` covariance of the hybrid-scheme step increments.

    For each step of width ``dt`` the hybrid scheme (:math:`\kappa = 1`) draws a bivariate
    Gaussian :math:`(W^{(1)}, W^{(2)})` with

    .. math::

        \mathrm{Var}(W^{(1)}) = \Delta, \quad
        \mathrm{Cov}(W^{(1)}, W^{(2)}) = \frac{\Delta^{\,\alpha+1}}{\alpha+1}, \quad
        \mathrm{Var}(W^{(2)}) = \frac{\Delta^{\,2\alpha+1}}{2\alpha+1},

    where :math:`\alpha = H - 1/2`. :math:`W^{(1)}` is the plain Brownian increment and
    :math:`W^{(2)} = \int_0^\Delta s^{\alpha}\, dZ_s` is the kernel-weighted increment used
    for the exact treatment of the singular term.
    """
    check_positive(dt, name="dt")
    check_in_range(hurst, name="hurst", low=0.0, high=0.5, inclusive=False)
    alpha = hurst - 0.5
    cov = np.empty((2, 2), dtype=np.float64)
    cov[0, 0] = dt
    cov[0, 1] = dt ** (alpha + 1.0) / (alpha + 1.0)
    cov[1, 0] = cov[0, 1]
    cov[1, 1] = dt ** (2.0 * alpha + 1.0) / (2.0 * alpha + 1.0)
    return cov


def hybrid_weights(n_steps: int, dt: float, hurst: float) -> NDArray[np.float64]:
    r"""Return the convolution weights :math:`G_k` for the hybrid scheme (:math:`\kappa=1`).

    The Riemann-sum part of the discretized Volterra process is
    :math:`\sum_{k\ge 2} G_k\, W^{(1)}_{i-k}` with

    .. math::

        G_k = \big(b_k\,\Delta\big)^{\alpha}, \qquad
        b_k = \left(\frac{k^{\alpha+1} - (k-1)^{\alpha+1}}{\alpha+1}\right)^{1/\alpha},

    evaluated at the optimal intra-interval point :math:`b_k`. By construction
    ``G[0] = G[1] = 0`` so the most recent interval is handled by the exact term instead.

    Returns
    -------
    numpy.ndarray
        Weights of length ``n_steps + 1`` indexed by lag ``k`` (``G[0]`` and ``G[1]`` are
        zero). The :math:`\sqrt{2H}` normalisation is applied later by the simulator.
    """
    check_positive(dt, name="dt")
    check_in_range(hurst, name="hurst", low=0.0, high=0.5, inclusive=False)
    if n_steps < 1:
        raise ValidationError("n_steps must be >= 1", context={"n_steps": n_steps})

    alpha = hurst - 0.5
    weights = np.zeros(n_steps + 1, dtype=np.float64)
    k = np.arange(2, n_steps + 1, dtype=np.float64)
    if k.size:
        b_k = ((k ** (alpha + 1.0) - (k - 1.0) ** (alpha + 1.0)) / (alpha + 1.0)) ** (1.0 / alpha)
        weights[2:] = (b_k * dt) ** alpha
    return weights
