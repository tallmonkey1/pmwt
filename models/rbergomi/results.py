"""Containers for simulated paths and Monte-Carlo distribution estimates.

These immutable result objects are the contract between the simulator and downstream
consumers (pricing, distribution surrogate, RL environment). They validate their own
invariants on construction so a malformed simulation can never silently propagate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ...core.errors import ValidationError
from ...core.validation import check_array_finite

__all__ = ["SimulationPaths", "TerminalDistribution"]


@dataclass(frozen=True, slots=True)
class SimulationPaths:
    """Simulated spot and variance paths on a shared time grid.

    Attributes
    ----------
    times:
        Grid time points in years, shape ``(n_steps + 1,)``, starting at ``0``.
    spot:
        Spot price paths, shape ``(n_paths, n_steps + 1)``. Column 0 is the initial spot.
    variance:
        Instantaneous-variance paths :math:`v_t`, shape ``(n_paths, n_steps + 1)``.
    """

    times: NDArray[np.float64]
    spot: NDArray[np.float64]
    variance: NDArray[np.float64]

    def __post_init__(self) -> None:
        times = np.ascontiguousarray(self.times, dtype=np.float64)
        spot = np.ascontiguousarray(self.spot, dtype=np.float64)
        variance = np.ascontiguousarray(self.variance, dtype=np.float64)
        check_array_finite(times, name="times")
        check_array_finite(spot, name="spot")
        check_array_finite(variance, name="variance")
        if times.ndim != 1:
            raise ValidationError("times must be 1-D", context={"ndim": times.ndim})
        if spot.ndim != 2 or variance.ndim != 2:
            raise ValidationError(
                "spot and variance must be 2-D (n_paths, n_points)",
                context={"spot_ndim": spot.ndim, "variance_ndim": variance.ndim},
            )
        if spot.shape != variance.shape:
            raise ValidationError(
                "spot and variance must share shape",
                context={"spot": spot.shape, "variance": variance.shape},
            )
        if spot.shape[1] != times.size:
            raise ValidationError(
                "path length must match number of time points",
                context={"n_points": int(times.size), "path_len": int(spot.shape[1])},
            )
        if np.any(spot <= 0.0):
            raise ValidationError("spot prices must be strictly positive", context={})
        if np.any(variance < 0.0):
            raise ValidationError("variance must be non-negative", context={})
        for arr in (times, spot, variance):
            arr.setflags(write=False)
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "spot", spot)
        object.__setattr__(self, "variance", variance)

    @property
    def n_paths(self) -> int:
        """Number of simulated paths."""
        return int(self.spot.shape[0])

    @property
    def n_steps(self) -> int:
        """Number of time steps (one fewer than the number of grid points)."""
        return int(self.spot.shape[1] - 1)

    @property
    def horizon(self) -> float:
        """Simulation horizon in years (final grid time)."""
        return float(self.times[-1])

    @property
    def initial_spot(self) -> float:
        """The common initial spot price ``S_0``."""
        return float(self.spot[0, 0])

    def terminal_spot(self) -> NDArray[np.float64]:
        """Return the terminal spot of every path, shape ``(n_paths,)``."""
        return np.ascontiguousarray(self.spot[:, -1])

    def terminal_log_return(self) -> NDArray[np.float64]:
        """Return terminal log-returns ``log(S_T / S_0)``, shape ``(n_paths,)``."""
        return np.log(self.terminal_spot() / self.initial_spot)


@dataclass(frozen=True, slots=True)
class TerminalDistribution:
    """Monte-Carlo estimate of the terminal log-return distribution with error control.

    This is the "main material" of the engine (SPEC §2.2): the probability distribution of
    the underlying over the trade horizon, here represented by its Monte-Carlo sample plus
    standard-error-aware summary statistics. Downstream code uses it for win-probability,
    expected P&L, and tail (CVaR) metrics.

    Attributes
    ----------
    log_returns:
        Sample of terminal log-returns, shape ``(n_paths,)``.
    horizon:
        Horizon in years.
    initial_spot:
        The initial spot used to generate the sample.
    mean_standard_error:
        Standard error of the estimated mean log-return (for convergence diagnostics).
    """

    log_returns: NDArray[np.float64]
    horizon: float
    initial_spot: float
    mean_standard_error: float

    def __post_init__(self) -> None:
        lr = np.ascontiguousarray(self.log_returns, dtype=np.float64)
        check_array_finite(lr, name="log_returns")
        if lr.ndim != 1 or lr.size == 0:
            raise ValidationError(
                "log_returns must be a non-empty 1-D array", context={"shape": lr.shape}
            )
        lr.setflags(write=False)
        object.__setattr__(self, "log_returns", lr)

    @property
    def n_paths(self) -> int:
        """Number of Monte-Carlo samples."""
        return int(self.log_returns.size)

    def terminal_spot(self) -> NDArray[np.float64]:
        """Return the implied terminal spot sample ``S_0 * exp(log_return)``."""
        return self.initial_spot * np.exp(self.log_returns)

    def quantile(self, q: float | NDArray[np.float64]) -> NDArray[np.float64]:
        """Return empirical quantile(s) of the terminal log-return distribution."""
        q_arr = np.atleast_1d(np.asarray(q, dtype=np.float64))
        if np.any((q_arr < 0.0) | (q_arr > 1.0)):
            raise ValidationError("quantile levels must lie in [0, 1]", context={})
        return np.asarray(np.quantile(self.log_returns, q_arr), dtype=np.float64)

    def probability_below(self, log_return_threshold: float) -> float:
        """Return the empirical probability that the terminal log-return is below a level."""
        return float(np.mean(self.log_returns < log_return_threshold))

    def probability_in_range(self, lower: float, upper: float) -> float:
        """Return the empirical probability the terminal log-return is in ``[lower, upper)``.

        This is the building block for iron-condor win-probability (SPEC §5).
        """
        if lower > upper:
            raise ValidationError(
                "lower must not exceed upper", context={"lower": lower, "upper": upper}
            )
        in_range = (self.log_returns >= lower) & (self.log_returns < upper)
        return float(np.mean(in_range))
