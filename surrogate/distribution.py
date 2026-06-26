r"""Quantile-based representation of a terminal log-return distribution.

The surrogate (SPEC §2.5) outputs a distribution as a set of **monotone, non-crossing
quantiles** on a fixed probability grid. This module wraps those quantiles in a
:class:`SurrogateDistribution` that exposes the *same* query interface as the Monte-Carlo
:class:`~options_engine.models.rbergomi.results.TerminalDistribution`
(``probability_in_range``, ``quantile``, ``terminal_spot`` sampling, ...), so downstream
trading code can consume either interchangeably -- the whole point of a surrogate.

A distribution defined by quantiles is the inverse-CDF (quantile function) sampled on a
grid. We interpolate it piecewise-linearly to evaluate the CDF and to draw samples by the
inverse-transform method, both of which are exact at the grid points and monotone between
them (because the quantiles are monotone by construction).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..core.errors import ValidationError
from ..core.validation import check_array_finite, check_positive

__all__ = ["SurrogateDistribution"]


@dataclass(frozen=True, slots=True)
class SurrogateDistribution:
    """A terminal log-return distribution represented by monotone quantiles.

    Parameters
    ----------
    quantile_levels:
        Strictly increasing probabilities in the open interval ``(0, 1)``, shape ``(Q,)``.
    quantile_values:
        Log-return value at each quantile level, shape ``(Q,)``. Must be non-decreasing
        (the no-crossing constraint); a small numerical tolerance is allowed and repaired.
    horizon:
        Horizon in years.
    initial_spot:
        Initial spot price used to map log-returns back to price space.
    """

    quantile_levels: NDArray[np.float64]
    quantile_values: NDArray[np.float64]
    horizon: float
    initial_spot: float

    # Tolerance for repairing tiny non-monotonicities from floating-point noise.
    _MONOTONE_TOL = 1e-9

    def __post_init__(self) -> None:
        levels = np.ascontiguousarray(self.quantile_levels, dtype=np.float64)
        values = np.ascontiguousarray(self.quantile_values, dtype=np.float64)
        check_array_finite(levels, name="quantile_levels")
        check_array_finite(values, name="quantile_values")
        if levels.ndim != 1 or values.ndim != 1:
            raise ValidationError("quantile arrays must be 1-D", context={})
        if levels.size < 2 or levels.size != values.size:
            raise ValidationError(
                "need >= 2 quantile levels and matching values",
                context={"n_levels": int(levels.size), "n_values": int(values.size)},
            )
        if np.any(levels <= 0.0) or np.any(levels >= 1.0):
            raise ValidationError("quantile levels must lie in (0, 1)", context={})
        if not np.all(np.diff(levels) > 0.0):
            raise ValidationError("quantile levels must be strictly increasing", context={})
        # Enforce / repair monotonicity of the quantile values (no-crossing).
        diffs = np.diff(values)
        if np.any(diffs < -self._MONOTONE_TOL):
            raise ValidationError(
                "quantile values cross (decreasing beyond tolerance)",
                context={"min_diff": float(diffs.min())},
            )
        repaired = np.maximum.accumulate(values)
        check_positive(self.horizon, name="horizon")
        check_positive(self.initial_spot, name="initial_spot")
        levels.setflags(write=False)
        repaired.setflags(write=False)
        object.__setattr__(self, "quantile_levels", levels)
        object.__setattr__(self, "quantile_values", repaired)

    @property
    def n_quantiles(self) -> int:
        """Number of quantile knots."""
        return int(self.quantile_levels.size)

    def quantile(self, q: float | NDArray[np.float64]) -> NDArray[np.float64]:
        """Return interpolated quantile value(s) at probability level(s) ``q``.

        Levels are clamped to the representable grid range ``[levels[0], levels[-1]]`` and
        interpolated linearly between knots (the quantile function is monotone there).
        """
        q_arr = np.atleast_1d(np.asarray(q, dtype=np.float64))
        if np.any((q_arr < 0.0) | (q_arr > 1.0)):
            raise ValidationError("quantile levels must lie in [0, 1]", context={})
        clipped = np.clip(q_arr, self.quantile_levels[0], self.quantile_levels[-1])
        return np.interp(clipped, self.quantile_levels, self.quantile_values)

    def cdf(self, log_return: float | NDArray[np.float64]) -> NDArray[np.float64]:
        r"""Return the CDF :math:`P(R \le x)` evaluated at ``log_return`` value(s).

        Implemented as the inverse of the (monotone) quantile function via linear
        interpolation, with flat extrapolation beyond the represented quantile range.
        """
        x = np.atleast_1d(np.asarray(log_return, dtype=np.float64))
        check_array_finite(x, name="log_return")
        # Interpolate level as a function of value (both monotone non-decreasing).
        result = np.interp(
            x,
            self.quantile_values,
            self.quantile_levels,
            left=self.quantile_levels[0],
            right=self.quantile_levels[-1],
        )
        return np.asarray(result, dtype=np.float64)

    def probability_below(self, log_return_threshold: float) -> float:
        """Return the estimated probability that the terminal log-return is below a level."""
        return float(self.cdf(np.array([log_return_threshold]))[0])

    def probability_in_range(self, lower: float, upper: float) -> float:
        """Return the estimated probability the terminal log-return is in ``[lower, upper)``.

        The iron-condor win-probability building block (mirrors
        :meth:`TerminalDistribution.probability_in_range`).
        """
        if lower > upper:
            raise ValidationError(
                "lower must not exceed upper", context={"lower": lower, "upper": upper}
            )
        return float(self.cdf(np.array([upper]))[0] - self.cdf(np.array([lower]))[0])

    def sample(self, n: int, *, rng: np.random.Generator) -> NDArray[np.float64]:
        """Draw ``n`` log-return samples via inverse-transform sampling.

        Uniform draws are mapped through the interpolated quantile function. Exact at the
        grid, monotone between knots.
        """
        if n < 1:
            raise ValidationError("n must be >= 1", context={"n": n})
        u = rng.uniform(self.quantile_levels[0], self.quantile_levels[-1], size=n)
        return np.interp(u, self.quantile_levels, self.quantile_values)

    def terminal_spot_quantile(self, q: float | NDArray[np.float64]) -> NDArray[np.float64]:
        """Return price-space quantile(s): ``S_0 * exp(quantile(q))``."""
        return self.initial_spot * np.exp(self.quantile(q))

    def mean(self) -> float:
        r"""Return a trapezoidal estimate of the mean log-return.

        Approximates :math:`\mathbb{E}[R] = \int_0^1 Q(p)\, dp` by the trapezoid rule over
        the quantile grid, augmented with the clamped tails outside ``[levels[0],
        levels[-1]]`` (treated as flat, a conservative tail assumption).
        """
        levels = self.quantile_levels
        values = self.quantile_values
        # Tail mass below the first / above the last level uses the boundary value.
        integral = float(values[0] * levels[0] + values[-1] * (1.0 - levels[-1]))
        integral += float(np.trapezoid(values, levels))
        return integral
