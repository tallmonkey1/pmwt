r"""Monte-Carlo convergence diagnostics and terminal-distribution aggregation.

Institutional-grade Monte Carlo never reports an estimate without an error bar (SPEC §2.3:
"the engine refuses to return a quantity whose MC standard error exceeds tolerance"). This
module turns raw :class:`~options_engine.models.rbergomi.results.SimulationPaths` into a
:class:`~options_engine.models.rbergomi.results.TerminalDistribution` and provides the
standard-error machinery used to gate that conversion.

For a Monte-Carlo mean estimator :math:`\hat\mu = \tfrac1N\sum X_i` the standard error is
:math:`\mathrm{se} = \sigma / \sqrt{N}` with :math:`\sigma` the sample standard deviation.
Antithetic sampling induces *negative* correlation between paired draws, so the naive
i.i.d. standard error is conservative there; we report it as an upper bound, which is the
safe choice for a go/no-go gate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from ...core.errors import ConvergenceError
from ...core.validation import check_positive, check_unit_interval
from .results import SimulationPaths, TerminalDistribution

__all__ = ["MonteCarloSummary", "build_terminal_distribution", "mean_standard_error"]


@dataclass(frozen=True, slots=True)
class MonteCarloSummary:
    """Point estimate with its Monte-Carlo standard error and a confidence interval."""

    estimate: float
    standard_error: float
    n_samples: int

    def confidence_interval(self, level: float = 0.95) -> tuple[float, float]:
        """Return a normal-approximation confidence interval at the given level."""
        check_unit_interval(level, name="level", inclusive=False)
        # Two-sided normal quantile for the requested coverage.
        z = float(norm.ppf(0.5 * (1.0 + level)))
        half_width = z * self.standard_error
        return self.estimate - half_width, self.estimate + half_width

    @property
    def relative_standard_error(self) -> float:
        """Standard error relative to the absolute estimate (``inf`` if estimate is 0)."""
        if self.estimate == 0.0:
            return float("inf")
        return self.standard_error / abs(self.estimate)


def mean_standard_error(samples: NDArray[np.float64]) -> MonteCarloSummary:
    """Return the sample mean with its Monte-Carlo standard error.

    Parameters
    ----------
    samples:
        One-dimensional array of i.i.d. (or antithetically-paired) Monte-Carlo draws.

    Notes
    -----
    Uses the unbiased (``ddof=1``) sample variance. Requires at least two samples to form
    a meaningful error estimate.
    """
    arr = np.ascontiguousarray(samples, dtype=np.float64)
    if arr.ndim != 1 or arr.size < 2:
        raise ConvergenceError(
            "need at least two samples to estimate a standard error",
            context={"n": int(arr.size)},
        )
    n = arr.size
    estimate = float(np.mean(arr))
    std = float(np.std(arr, ddof=1))
    se = std / np.sqrt(n)
    return MonteCarloSummary(estimate=estimate, standard_error=se, n_samples=int(n))


def build_terminal_distribution(
    paths: SimulationPaths,
    *,
    max_rel_standard_error: float | None = None,
) -> TerminalDistribution:
    """Aggregate simulated paths into a terminal-distribution estimate with error control.

    Parameters
    ----------
    paths:
        Simulated spot/variance paths.
    max_rel_standard_error:
        If provided, the relative standard error of the mean terminal log-return must not
        exceed this tolerance, otherwise a :class:`ConvergenceError` is raised. This is the
        gate that prevents under-converged distributions from reaching trading logic. When
        ``None`` the check is skipped (the caller takes responsibility).

    Returns
    -------
    TerminalDistribution
        The terminal log-return sample plus its mean standard error.
    """
    log_returns = paths.terminal_log_return()
    summary = mean_standard_error(log_returns)

    if max_rel_standard_error is not None:
        check_positive(max_rel_standard_error, name="max_rel_standard_error")
        # Guard against the degenerate near-zero-mean case by comparing the *absolute*
        # standard error against the sample's own scale when the mean is ~0.
        scale = max(abs(summary.estimate), float(np.std(log_returns, ddof=1)))
        rel_se = summary.standard_error / scale if scale > 0.0 else float("inf")
        if rel_se > max_rel_standard_error:
            raise ConvergenceError(
                "terminal-distribution Monte-Carlo standard error exceeds tolerance; "
                "increase n_paths or relax the tolerance",
                context={
                    "relative_standard_error": rel_se,
                    "tolerance": max_rel_standard_error,
                    "n_paths": paths.n_paths,
                },
            )

    return TerminalDistribution(
        log_returns=log_returns,
        horizon=paths.horizon,
        initial_spot=paths.initial_spot,
        mean_standard_error=summary.standard_error,
    )
