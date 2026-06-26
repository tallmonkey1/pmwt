r"""Vol-of-vol (:math:`\eta`) and spot-vol correlation (:math:`\rho`) estimation.

Naive moment estimators of :math:`\eta` and :math:`\rho` are badly biased under rough
volatility because the increments of the Volterra driver are strongly dependent (SPEC §0,
§2.4). The robust, standard approach is **simulation-based moment matching** (a form of
quasi-MLE / indirect inference): pick the parameter whose *simulated* moment reproduces the
empirical moment, using the simulator itself as the link function. This is exact in the
limit and avoids closed-form approximations that break for small ``H``.

* :func:`estimate_eta` matches the **level of the log-variance structure function**
  :math:`m_2(\Delta) = \mathbb{E}[(\log v_{t+\Delta} - \log v_t)^2]`. Because
  :math:`\log v` is linear in :math:`\eta` (up to the deterministic drift correction),
  :math:`m_2 \propto \eta^2`, so :math:`\eta` is recovered from the ratio of the empirical
  to a unit-:math:`\eta` simulated structure function -- a single cheap calibration.

* :func:`estimate_rho` inverts the **empirical spot-vol (leverage) correlation** through a
  monotone simulated calibration curve :math:`\text{corr}_{\text{emp}} = f(\rho)` built at
  the already-estimated ``H`` and ``eta``.

Both estimators take an injected :class:`RandomFactory` so they are fully reproducible.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..core.errors import CalibrationError, ValidationError
from ..core.random import RandomFactory
from ..core.timegrid import TimeGrid
from ..core.validation import check_correlation, check_in_range, check_positive
from ..models.rbergomi import ForwardVariance, HybridSimulator, RBergomiParams
from .results import ParameterEstimate

__all__ = ["estimate_eta", "estimate_rho", "structure_function"]


def structure_function(
    log_variance: NDArray[np.float64], lags: NDArray[np.int_]
) -> NDArray[np.float64]:
    r"""Return the second-order structure function :math:`m_2(\Delta)` at each lag.

    Operates per-row for 2-D inputs (each row a path) and pools, so it works for both an
    empirical single series (1-D) and a simulated ensemble (2-D).
    """
    lv = np.asarray(log_variance, dtype=np.float64)
    lag_arr = np.asarray(lags, dtype=np.int_)
    if lag_arr.ndim != 1 or lag_arr.size == 0:
        raise ValidationError("lags must be a non-empty 1-D array", context={})
    if lv.ndim == 1:
        lv = lv[np.newaxis, :]
    elif lv.ndim != 2:
        raise ValidationError("log_variance must be 1-D or 2-D", context={"ndim": lv.ndim})
    n = lv.shape[1]
    if np.any(lag_arr < 1) or np.any(lag_arr >= n):
        raise ValidationError("lags must lie in [1, n-1]", context={"n": int(n)})

    out = np.empty(lag_arr.size, dtype=np.float64)
    for i, lag in enumerate(lag_arr):
        diff = lv[:, lag:] - lv[:, :-lag]
        out[i] = float(np.mean(diff**2))
    return out


def estimate_eta(
    log_variance: NDArray[np.float64],
    *,
    hurst: float,
    xi0_level: float,
    grid: TimeGrid,
    rng_factory: RandomFactory,
    lags: NDArray[np.int_] | None = None,
    n_sim_paths: int = 200,
    rho_for_sim: float = -0.5,
) -> ParameterEstimate:
    r"""Estimate :math:`\eta` by matching the level of the log-variance structure function.

    Parameters
    ----------
    log_variance:
        Empirical log-variance series (1-D), regularly sampled on ``grid``'s step.
    hurst:
        Previously-estimated Hurst exponent.
    xi0_level:
        Forward-variance level used for the simulated reference (the level cancels in the
        ratio, but a sensible value keeps the simulation well-scaled).
    grid:
        Time grid whose ``dt`` matches the sampling interval of ``log_variance``.
    rng_factory:
        Reproducible randomness for the reference simulation.
    lags:
        Lags at which to match the structure function. Defaults to ``1 .. min(n//4, 30)``.
    n_sim_paths:
        Number of reference paths simulated at unit ``eta``.
    rho_for_sim:
        Correlation used in the reference simulation. The structure function of
        ``log v`` is (to leading order) independent of ``rho``, so this has negligible
        effect; it is exposed only for completeness.
    """
    check_in_range(hurst, name="hurst", low=0.0, high=0.5, inclusive=False)
    check_positive(xi0_level, name="xi0_level")
    check_correlation(rho_for_sim, name="rho_for_sim")
    if n_sim_paths < 1:
        raise ValidationError("n_sim_paths must be >= 1", context={"n_sim_paths": n_sim_paths})

    lv = np.asarray(log_variance, dtype=np.float64)
    if lv.ndim != 1 or lv.size < 20:
        raise CalibrationError(
            "log_variance must be a 1-D series of length >= 20", context={"size": int(lv.size)}
        )

    n = lv.size
    if lags is None:
        upper = min(n // 4, 30)
        lag_arr = np.arange(1, max(upper, 2) + 1, dtype=np.int_)
    else:
        lag_arr = np.asarray(lags, dtype=np.int_)

    # Reference simulation at unit eta.
    unit_params = RBergomiParams(
        hurst=hurst, eta=1.0, rho=rho_for_sim, forward_variance=ForwardVariance.flat(xi0_level)
    )
    sim = HybridSimulator(unit_params, rng_factory=rng_factory)
    sim_paths = sim.simulate(grid=grid, n_paths=n_sim_paths, initial_spot=100.0)
    sim_log_var = np.log(sim_paths.variance)

    sim_lags = lag_arr[lag_arr < sim_log_var.shape[1]]
    if sim_lags.size == 0:
        raise CalibrationError(
            "simulation grid too short for the requested lags",
            context={"grid_points": int(sim_log_var.shape[1])},
        )

    m2_data = structure_function(lv, sim_lags)
    m2_unit = structure_function(sim_log_var, sim_lags)
    if np.any(m2_unit <= 0.0):
        raise CalibrationError("degenerate simulated structure function", context={})

    # m2 propto eta^2, so eta = sqrt(mean(m2_data / m2_unit)). Average over lags for
    # robustness; the per-lag dispersion provides the standard error.
    ratio = m2_data / m2_unit
    eta_sq = float(np.mean(ratio))
    if eta_sq <= 0.0:
        raise CalibrationError("non-positive eta^2 estimate", context={"eta_sq": eta_sq})
    eta = float(np.sqrt(eta_sq))

    # Delta-method standard error: Var(eta) ~ Var(eta_sq) / (4 eta^2).
    ratio_se = float(np.std(ratio, ddof=1) / np.sqrt(ratio.size)) if ratio.size > 1 else 0.0
    eta_se = ratio_se / (2.0 * eta) if eta > 0.0 else 0.0

    return ParameterEstimate(name="eta", value=eta, std_error=eta_se, n_observations=int(n))


def estimate_rho(
    log_price: NDArray[np.float64],
    log_variance: NDArray[np.float64],
    *,
    hurst: float,
    eta: float,
    xi0_level: float,
    grid: TimeGrid,
    rng_factory: RandomFactory,
    n_grid: int = 9,
    n_sim_paths: int = 60,
) -> ParameterEstimate:
    r"""Estimate :math:`\rho` by inverting a simulated leverage-correlation curve.

    Builds a monotone map :math:`\text{corr}_{\text{emp}}(\rho)` by simulating at a grid of
    ``rho`` values (holding ``H`` and ``eta`` fixed), then interpolates the empirical
    spot-vol correlation onto that curve. The leverage correlation is monotone increasing
    in ``rho``, which guarantees a unique inverse.

    Parameters
    ----------
    log_price, log_variance:
        Empirical log-price and log-variance series of equal length, sampled on ``grid``.
    hurst, eta, xi0_level:
        Previously-estimated parameters held fixed for the reference simulations.
    n_grid:
        Number of ``rho`` knots in ``[-0.95, -0.05]`` used to build the calibration curve
        (equity leverage is negative; the range is restricted accordingly but the result is
        clamped to a valid correlation).
    n_sim_paths:
        Paths per knot.
    """
    check_in_range(hurst, name="hurst", low=0.0, high=0.5, inclusive=False)
    check_positive(eta, name="eta")
    check_positive(xi0_level, name="xi0_level")
    if n_grid < 3:
        raise ValidationError("n_grid must be >= 3", context={"n_grid": n_grid})

    lp = np.asarray(log_price, dtype=np.float64)
    lv = np.asarray(log_variance, dtype=np.float64)
    if lp.shape != lv.shape or lp.ndim != 1:
        raise ValidationError(
            "log_price and log_variance must be 1-D arrays of equal length",
            context={"lp": lp.shape, "lv": lv.shape},
        )
    if lp.size < 20:
        raise CalibrationError("series too short for rho estimation", context={"n": int(lp.size)})

    emp_corr = _leverage_correlation(lp[np.newaxis, :], lv[np.newaxis, :])

    rho_knots = np.linspace(-0.95, -0.05, n_grid)
    sim_corr = np.empty(n_grid, dtype=np.float64)
    for i, rho in enumerate(rho_knots):
        params = RBergomiParams(
            hurst=hurst, eta=eta, rho=float(rho), forward_variance=ForwardVariance.flat(xi0_level)
        )
        # Distinct sub-stream per knot keeps the curve smooth yet reproducible.
        sim = HybridSimulator(params, rng_factory=rng_factory)
        paths = sim.simulate(grid=grid, n_paths=n_sim_paths, initial_spot=100.0)
        sim_lp = np.log(paths.spot)
        sim_lv = np.log(paths.variance)
        sim_corr[i] = _leverage_correlation(sim_lp, sim_lv)

    # The curve must be monotone increasing in rho for a unique inverse; enforce by sorting
    # on the simulated correlation and guarding against numerical non-monotonicity.
    order = np.argsort(sim_corr)
    sim_corr_sorted = sim_corr[order]
    rho_sorted = rho_knots[order]
    if not np.all(np.diff(sim_corr_sorted) >= -1e-6):
        raise CalibrationError(
            "simulated leverage-correlation curve is not monotone; increase n_sim_paths",
            context={"sim_corr": sim_corr.tolist()},
        )

    rho_est = float(np.interp(emp_corr, sim_corr_sorted, rho_sorted))
    rho_est = float(np.clip(rho_est, -1.0, 1.0))

    # Standard error via the local slope of the calibration curve and the sampling error of
    # the empirical correlation (Fisher approximation: se(corr) ~ 1/sqrt(n - 3)).
    n_eff = max(lp.size - 1, 4)
    corr_se = 1.0 / np.sqrt(n_eff - 3)
    slope = float(np.gradient(rho_sorted, sim_corr_sorted).mean())
    rho_se = abs(slope) * corr_se

    return ParameterEstimate(
        name="rho", value=rho_est, std_error=float(rho_se), n_observations=int(lp.size)
    )


def _leverage_correlation(
    log_price: NDArray[np.float64], log_variance: NDArray[np.float64]
) -> float:
    """Pooled correlation between log-price and log-variance increments across paths."""
    d_logp = np.diff(log_price, axis=1).ravel()
    d_logv = np.diff(log_variance, axis=1).ravel()
    if d_logp.size < 2:
        raise CalibrationError("not enough increments for correlation", context={})
    std_p = float(np.std(d_logp))
    std_v = float(np.std(d_logv))
    if std_p <= 0.0 or std_v <= 0.0:
        raise CalibrationError("degenerate increments (zero variance)", context={})
    return float(np.corrcoef(d_logp, d_logv)[0, 1])
