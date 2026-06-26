r"""End-to-end rBergomi calibration orchestration (SPEC §2.4).

Ties the individual estimators together into the layered procedure described in the spec:

1. Build realized variance / log-variance proxies from observed prices.
2. Estimate the forward-variance level :math:`\xi_0`.
3. Estimate the Hurst exponent :math:`H` from the log-variance scaling.
4. Estimate the vol-of-vol :math:`\eta` by structure-function level matching.
5. Estimate the spot-vol correlation :math:`\rho` by inverting a simulated leverage curve.
6. Run the BNS jump test to decide whether jumps should be enabled.

The orchestrator is deliberately *injectable* and *reproducible*: it takes a
:class:`RandomFactory` and a sampling description, performs no hidden I/O, and returns a
single, fully-validated :class:`CalibrationResult` with provenance metadata.

**Honesty note.** This is a *historical / physical-measure* calibration from underlying
prices. The simulation-based steps (`eta`, `rho`) use Monte-Carlo references, so their
estimates carry simulation noise that shrinks with ``n_sim_paths``. When option data is
available, :func:`options_engine.calibration.forward_variance.estimate_xi0_curve` should be
used to pin the forward-variance term structure under the pricing measure instead of the
flat historical level used here.
"""

from __future__ import annotations

import datetime as _dt

import numpy as np
from numpy.typing import NDArray

from ..core.errors import CalibrationError
from ..core.random import RandomFactory
from ..core.timegrid import TRADING_DAYS_PER_YEAR, TimeGrid
from .hurst import estimate_hurst
from .jumps import bns_jump_test
from .realized import daily_realized_variance, log_returns, log_variance_proxy
from .results import CalibrationResult, ParameterEstimate
from .vol_of_vol import estimate_eta, estimate_rho

__all__ = ["CalibrationConfig", "calibrate_rbergomi"]


class CalibrationConfig:
    """Configuration for the end-to-end calibration.

    Grouping the knobs here keeps the calibration function signature small and makes the
    settings serializable for run manifests.
    """

    __slots__ = (
        "jump_significance",
        "min_days",
        "n_sim_paths_eta",
        "n_sim_paths_rho",
        "rho_grid_size",
        "steps_per_day",
        "trading_days_per_year",
    )

    def __init__(
        self,
        *,
        steps_per_day: int = 26,
        trading_days_per_year: int = TRADING_DAYS_PER_YEAR,
        n_sim_paths_eta: int = 200,
        n_sim_paths_rho: int = 60,
        rho_grid_size: int = 9,
        jump_significance: float = 0.01,
        min_days: int = 60,
    ) -> None:
        if steps_per_day < 2:
            raise CalibrationError(
                "steps_per_day must be >= 2 (need intraday returns)",
                context={"steps_per_day": steps_per_day},
            )
        if min_days < 30:
            raise CalibrationError(
                "min_days must be >= 30 for a meaningful calibration",
                context={"min_days": min_days},
            )
        self.steps_per_day = steps_per_day
        self.trading_days_per_year = trading_days_per_year
        self.n_sim_paths_eta = n_sim_paths_eta
        self.n_sim_paths_rho = n_sim_paths_rho
        self.rho_grid_size = rho_grid_size
        self.jump_significance = jump_significance
        self.min_days = min_days


def calibrate_rbergomi(
    intraday_prices: NDArray[np.float64],
    *,
    rng_factory: RandomFactory,
    config: CalibrationConfig | None = None,
    data_start: _dt.datetime | None = None,
    data_end: _dt.datetime | None = None,
    now: _dt.datetime | None = None,
) -> CalibrationResult:
    r"""Calibrate the rBergomi model from an intraday price series.

    Parameters
    ----------
    intraday_prices:
        One-dimensional series of intraday prices whose length is a multiple of
        ``config.steps_per_day`` (a whole number of trading days).
    rng_factory:
        Reproducible randomness for the simulation-based estimators.
    config:
        Calibration settings; defaults are used if omitted.
    data_start, data_end, now:
        Optional provenance timestamps. Default to the Unix epoch / current UTC time so the
        result is always well-formed and auditable.

    Returns
    -------
    CalibrationResult
        The validated, fully-specified parameter set with provenance and diagnostics.
    """
    cfg = config or CalibrationConfig()
    prices = np.asarray(intraday_prices, dtype=np.float64)
    if prices.ndim != 1:
        raise CalibrationError("intraday_prices must be 1-D", context={"ndim": prices.ndim})
    if prices.size % cfg.steps_per_day != 0:
        raise CalibrationError(
            "price series length must be a whole number of trading days",
            context={"size": int(prices.size), "steps_per_day": cfg.steps_per_day},
        )
    n_days = prices.size // cfg.steps_per_day
    if n_days < cfg.min_days:
        raise CalibrationError(
            "insufficient history for calibration",
            context={"days": n_days, "min_days": cfg.min_days},
        )

    # 1. Realized-variance proxies.
    intraday_ret = log_returns(prices)
    # Drop the first overnight return so each day has exactly steps_per_day-1 intraday
    # returns plus alignment; we instead recompute per-day RV directly from intraday
    # returns truncated to whole days.
    usable = (intraday_ret.size // cfg.steps_per_day) * cfg.steps_per_day
    intraday_ret = intraday_ret[:usable]
    daily_rv = daily_realized_variance(
        intraday_ret,
        steps_per_day=cfg.steps_per_day,
        annualize=True,
        trading_days_per_year=cfg.trading_days_per_year,
    )
    log_var = log_variance_proxy(daily_rv)

    # 2. Forward-variance level (flat, historical).
    xi0_level = float(np.mean(daily_rv))
    xi0_est = ParameterEstimate(
        name="xi0_level",
        value=xi0_level,
        std_error=float(np.std(daily_rv, ddof=1) / np.sqrt(daily_rv.size)),
        n_observations=int(daily_rv.size),
    )

    # 3. Hurst exponent.
    hurst_est = estimate_hurst(log_var)

    # Daily grid for the simulation-based estimators (one step per trading day).
    daily_grid = TimeGrid.from_calendar_days(
        calendar_days=daily_rv.size,
        steps_per_day=1,
        trading_days_per_year=cfg.trading_days_per_year,
    )

    # 4. Vol-of-vol.
    eta_est = estimate_eta(
        log_var,
        hurst=hurst_est.value,
        xi0_level=xi0_level,
        grid=daily_grid,
        rng_factory=rng_factory,
        n_sim_paths=cfg.n_sim_paths_eta,
    )

    # 5. Spot-vol correlation. Use daily-close log-prices aligned with the daily RV.
    daily_close = prices[cfg.steps_per_day - 1 :: cfg.steps_per_day][: daily_rv.size]
    log_price = np.log(daily_close)
    # Align lengths: log_var has one entry per day; use as many days as both share.
    m = min(log_price.size, log_var.size)
    rho_est = estimate_rho(
        log_price[:m],
        log_var[:m],
        hurst=hurst_est.value,
        eta=eta_est.value,
        xi0_level=xi0_level,
        grid=TimeGrid.from_calendar_days(
            calendar_days=m, steps_per_day=1, trading_days_per_year=cfg.trading_days_per_year
        ),
        rng_factory=rng_factory,
        n_grid=cfg.rho_grid_size,
        n_sim_paths=cfg.n_sim_paths_rho,
    )

    # 6. Jump test on the pooled intraday returns.
    jump_result = bns_jump_test(intraday_ret, significance=cfg.jump_significance)

    epoch = _dt.datetime(1970, 1, 1, tzinfo=_dt.UTC)
    return CalibrationResult(
        hurst=hurst_est,
        eta=eta_est,
        rho=rho_est,
        xi0_level=xi0_est,
        as_of=now or _dt.datetime.now(_dt.UTC),
        data_start=data_start or epoch,
        data_end=data_end or (now or _dt.datetime.now(_dt.UTC)),
        jumps_detected=jump_result.jumps_detected,
        diagnostics={
            "hurst_r_squared": hurst_est.r_squared or 0.0,
            "eta_std_error": eta_est.std_error,
            "rho_std_error": rho_est.std_error,
            "jump_statistic": jump_result.statistic,
            "jump_p_value": jump_result.p_value,
            "relative_jump": jump_result.relative_jump,
            "n_days": float(n_days),
        },
    )
