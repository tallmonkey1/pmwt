"""Parameters of the rough-Bergomi (rBergomi) model.

The rBergomi model (Bayer, Friz & Gatheral, 2016, *Pricing under rough volatility*) is a
non-Markovian stochastic-volatility model in which the instantaneous variance is driven by
a fractional (Riemann-Liouville / Volterra) Brownian motion with Hurst exponent
``H < 1/2``. Under the pricing measure the dynamics are

.. math::

    dS_t / S_t   &= r\\, dt + \\sqrt{v_t}\\,\\big(\\rho\\, dZ_t
                     + \\sqrt{1-\\rho^2}\\, dB_t\\big), \\\\
    v_t          &= \\xi_0(t)\\, \\exp\\!\\Big(\\eta\\, Y_t
                     - \\tfrac{1}{2}\\eta^2\\, t^{2H}\\Big), \\\\
    Y_t          &= \\sqrt{2H}\\int_0^t (t-s)^{H-1/2}\\, dZ_s,

where :math:`Z` and :math:`B` are independent Brownian motions, :math:`Y` is the
Volterra/rough driver (with :math:`\\mathbb{E}[Y_t^2] = t^{2H}`), and the exponential is
*martingale corrected* so that :math:`\\mathbb{E}[v_t] = \\xi_0(t)` exactly.

This module defines the immutable, fully-validated parameter container. The forward
variance curve :math:`\\xi_0(\\cdot)` is represented by :class:`ForwardVariance`, which
supports the common flat case as well as a piecewise-linear term structure.

References
----------
* C. Bayer, P. Friz, J. Gatheral. *Pricing under rough volatility.* Quantitative Finance
  16(6), 2016.
* M. Bennedsen, A. Lunde, M. Pakkanen. *Hybrid scheme for Brownian semistationary
  processes.* Finance and Stochastics 21(4), 2017.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from ...core.errors import ValidationError
from ...core.validation import (
    check_array_finite,
    check_correlation,
    check_finite,
    check_in_range,
    check_positive,
)

__all__ = ["ForwardVariance", "RBergomiParams"]


@dataclass(frozen=True, slots=True)
class ForwardVariance:
    """The forward-variance curve :math:`\\xi_0(t)` (the model's term structure of variance).

    Two representations are supported:

    * **Flat** — a single constant level (the most common research/teaching case), built
      with :meth:`flat`.
    * **Piecewise-linear** — values given on an increasing grid of maturities (in years),
      linearly interpolated and flat-extrapolated beyond the last knot. This is what an
      IV-surface calibration produces (SPEC §2.4).

    The curve must be strictly positive everywhere (variance cannot be zero or negative).

    Parameters
    ----------
    knot_times:
        Strictly increasing maturities in years, starting at ``t = 0``. For a flat curve
        this is ``[0.0]``.
    knot_values:
        Forward-variance level at each knot. Same length as ``knot_times``; all strictly
        positive.
    """

    knot_times: NDArray[np.float64]
    knot_values: NDArray[np.float64]

    def __post_init__(self) -> None:
        times = np.ascontiguousarray(self.knot_times, dtype=np.float64)
        values = np.ascontiguousarray(self.knot_values, dtype=np.float64)
        check_array_finite(times, name="knot_times")
        check_array_finite(values, name="knot_values")
        if times.ndim != 1 or values.ndim != 1:
            raise ValidationError(
                "knot arrays must be one-dimensional",
                context={"times_ndim": times.ndim, "values_ndim": values.ndim},
            )
        if times.size == 0:
            raise ValidationError("forward-variance curve needs at least one knot", context={})
        if times.size != values.size:
            raise ValidationError(
                "knot_times and knot_values must have equal length",
                context={"n_times": int(times.size), "n_values": int(values.size)},
            )
        if times[0] != 0.0:
            raise ValidationError(
                "knot_times must start at t = 0", context={"first": float(times[0])}
            )
        if times.size > 1 and not np.all(np.diff(times) > 0.0):
            raise ValidationError("knot_times must be strictly increasing", context={})
        if np.any(values <= 0.0):
            raise ValidationError(
                "forward variance must be strictly positive everywhere",
                context={"min_value": float(values.min())},
            )
        # Store canonical, immutable copies (defend against external mutation).
        times.setflags(write=False)
        values.setflags(write=False)
        object.__setattr__(self, "knot_times", times)
        object.__setattr__(self, "knot_values", values)

    @classmethod
    def flat(cls, level: float) -> ForwardVariance:
        """Build a flat forward-variance curve at the given (positive) level.

        ``level`` is a *variance* (e.g. ``0.04`` for a 20% vol), not a volatility.
        """
        check_positive(level, name="level")
        return cls(knot_times=np.array([0.0]), knot_values=np.array([float(level)]))

    @property
    def is_flat(self) -> bool:
        """True if the curve is a single constant level."""
        return self.knot_times.size == 1

    def __call__(self, t: NDArray[np.float64] | float) -> NDArray[np.float64]:
        """Evaluate :math:`\\xi_0(t)` at time(s) ``t`` (years), vectorized.

        Linear interpolation between knots; flat extrapolation outside the knot range.
        Negative times are rejected.
        """
        t_arr = np.atleast_1d(np.asarray(t, dtype=np.float64))
        check_array_finite(t_arr, name="t")
        if np.any(t_arr < 0.0):
            raise ValidationError(
                "forward variance is undefined for negative time",
                context={"min_t": float(t_arr.min())},
            )
        if self.is_flat:
            return np.full(t_arr.shape, self.knot_values[0], dtype=np.float64)
        # np.interp performs flat extrapolation beyond the endpoints by default.
        return np.interp(t_arr, self.knot_times, self.knot_values)

    def at(self, t: float) -> float:
        """Scalar convenience accessor for :math:`\\xi_0(t)`."""
        return float(self(np.array([t]))[0])


@dataclass(frozen=True, slots=True)
class RBergomiParams:
    """Immutable, validated parameter set for the rBergomi model.

    Parameters
    ----------
    hurst:
        Hurst exponent :math:`H \\in (0, 1/2)`. Rough volatility corresponds to small
        :math:`H` (empirically :math:`H \\approx 0.05\\text{--}0.2`). The endpoints are
        excluded: :math:`H = 1/2` is classical (non-rough) Brownian behaviour and
        :math:`H = 0` is degenerate.
    eta:
        Volatility-of-volatility :math:`\\eta > 0`.
    rho:
        Spot-vol correlation :math:`\\rho \\in [-1, 1]`. The leverage effect makes
        :math:`\\rho < 0` typical for equity indices.
    forward_variance:
        The forward-variance curve :math:`\\xi_0(\\cdot)`.
    rate:
        Continuously-compounded risk-free rate used for the spot drift under the pricing
        measure. Defaults to ``0.0``. (The *physical* dynamic drift model of SPEC §2.1 is
        a separate component; this field is the deterministic carry term.)
    """

    hurst: float
    eta: float
    rho: float
    forward_variance: ForwardVariance
    rate: float = 0.0
    # Cached, derived quantity (set in __post_init__): the Volterra normalisation
    # alpha = H - 1/2, which recurs throughout the kernel math.
    alpha: float = field(init=False)

    def __post_init__(self) -> None:
        check_in_range(self.hurst, name="hurst", low=0.0, high=0.5, inclusive=False)
        check_positive(self.eta, name="eta")
        check_correlation(self.rho, name="rho")
        check_finite(self.rate, name="rate")
        if not isinstance(self.forward_variance, ForwardVariance):
            raise ValidationError(
                "forward_variance must be a ForwardVariance instance",
                context={"type": type(self.forward_variance).__name__},
            )
        object.__setattr__(self, "alpha", self.hurst - 0.5)

    @property
    def gamma(self) -> float:
        """Kernel singularity exponent :math:`\\gamma = 1/2 - H \\in (0, 1/2)`.

        The Volterra kernel behaves like :math:`s^{-\\gamma}` near zero; ``gamma`` is the
        natural quantity for the hybrid scheme's singularity treatment.
        """
        return 0.5 - self.hurst

    def xi0(self, t: NDArray[np.float64] | float) -> NDArray[np.float64]:
        """Evaluate the forward-variance curve :math:`\\xi_0(t)`."""
        return self.forward_variance(t)

    def with_eta(self, eta: float) -> RBergomiParams:
        """Return a copy with a different ``eta`` (useful for analytic-limit tests)."""
        return RBergomiParams(
            hurst=self.hurst,
            eta=eta,
            rho=self.rho,
            forward_variance=self.forward_variance,
            rate=self.rate,
        )
