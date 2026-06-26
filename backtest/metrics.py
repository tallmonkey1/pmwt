r"""Performance and risk metrics for backtest evaluation (SPEC §9).

These are the standard, mathematically-defined statistics a promotion gate keys off. Each is
implemented from its definition and unit-tested against hand-computed values, because a subtly
wrong Sharpe or drawdown silently corrupts every go/no-go decision downstream.

Included:

* **Total / annualized return**, **annualized volatility**.
* **Sharpe** and **Sortino** ratios (annualized).
* **Maximum drawdown** and **Calmar** ratio.
* **CVaR / VaR** of the return distribution (tail risk a short-gamma book lives or dies by).
* **Win rate** and **profit factor** over trades.
* **Deflated Sharpe Ratio (DSR)** (Bailey & López de Prado, 2014) -- the Sharpe corrected for
  the number of trials, non-normal returns (skew/kurtosis), and sample length. This is the
  spec's "multiple-testing / deflated-Sharpe correction": a strategy that looks good only
  because many variants were tried will have a low DSR, and the gate rejects it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from ..core.errors import ValidationError
from ..core.validation import check_array_finite, check_positive

__all__ = [
    "PerformanceMetrics",
    "compute_performance_metrics",
    "conditional_value_at_risk",
    "deflated_sharpe_ratio",
    "max_drawdown",
    "sharpe_ratio",
    "sortino_ratio",
]

# Trading periods per year for annualization, by step cadence. Daily is the default.
TRADING_PERIODS_PER_YEAR: int = 252


def _validate_returns(returns: NDArray[np.float64]) -> NDArray[np.float64]:
    r = np.asarray(returns, dtype=np.float64)
    check_array_finite(r, name="returns")
    if r.ndim != 1 or r.size < 2:
        raise ValidationError(
            "returns must be a 1-D series of length >= 2", context={"size": int(r.size)}
        )
    return r


def sharpe_ratio(
    returns: NDArray[np.float64], *, periods_per_year: int = TRADING_PERIODS_PER_YEAR
) -> float:
    r"""Return the annualized Sharpe ratio of a per-period return series.

    :math:`\mathrm{Sharpe} = \sqrt{P}\,\bar{r} / \sigma_r`, with ``P`` periods per year. A
    zero-volatility series has an undefined Sharpe and returns ``0.0`` (no risk-adjusted edge
    can be claimed without variability).
    """
    r = _validate_returns(returns)
    std = float(np.std(r, ddof=1))
    # Treat a negligible std as zero volatility: no risk-adjusted edge is definable, and the
    # tiny floating-point residual of a "constant" series must not produce a giant Sharpe.
    if std <= 1e-12:
        return 0.0
    return float(np.sqrt(periods_per_year) * np.mean(r) / std)


def sortino_ratio(
    returns: NDArray[np.float64],
    *,
    periods_per_year: int = TRADING_PERIODS_PER_YEAR,
    target: float = 0.0,
) -> float:
    r"""Return the annualized Sortino ratio (downside-deviation-adjusted return)."""
    r = _validate_returns(returns)
    downside = np.minimum(r - target, 0.0)
    downside_dev = float(np.sqrt(np.mean(downside**2)))
    if downside_dev <= 0.0:
        return 0.0
    return float(np.sqrt(periods_per_year) * (np.mean(r) - target) / downside_dev)


def max_drawdown(equity_curve: NDArray[np.float64]) -> float:
    r"""Return the maximum drawdown of an equity curve as a positive fraction in ``[0, 1]``.

    Drawdown at time ``t`` is ``1 - equity_t / running_peak_t``; the maximum over the series is
    returned. Requires a strictly-positive equity curve.
    """
    eq = np.asarray(equity_curve, dtype=np.float64)
    check_array_finite(eq, name="equity_curve")
    if eq.ndim != 1 or eq.size < 1:
        raise ValidationError("equity_curve must be a non-empty 1-D series", context={})
    if np.any(eq <= 0.0):
        raise ValidationError("equity_curve must be strictly positive", context={})
    running_peak = np.maximum.accumulate(eq)
    drawdowns = 1.0 - eq / running_peak
    return float(np.max(drawdowns))


def conditional_value_at_risk(returns: NDArray[np.float64], *, alpha: float = 0.05) -> float:
    r"""Return the CVaR (expected shortfall) of a return series at level ``alpha``.

    The mean of the worst ``alpha`` fraction of returns, reported as a positive loss
    magnitude (a CVaR of 0.03 means the average of the worst outcomes is a 3% loss).
    """
    r = _validate_returns(returns)
    if not 0.0 < alpha < 1.0:
        raise ValidationError("alpha must lie in (0, 1)", context={"alpha": alpha})
    var_threshold = float(np.quantile(r, alpha))
    tail = r[r <= var_threshold]
    if tail.size == 0:  # pragma: no cover - quantile guarantees at least one element
        return float(-var_threshold)
    return float(-np.mean(tail))


def deflated_sharpe_ratio(
    returns: NDArray[np.float64],
    *,
    n_trials: int,
) -> float:
    r"""Return the Deflated Sharpe Ratio probability (Bailey & López de Prado, 2014).

    The DSR is the probability that the *true* Sharpe exceeds a benchmark that accounts for
    the number of independent strategy trials, the sample length, and the return
    distribution's skewness and kurtosis. It answers "is this Sharpe real, or an artefact of
    trying many variants?" -- exactly the multiple-testing correction the promotion gate needs.

    Returns a probability in ``[0, 1]``; a common acceptance threshold is ``> 0.95``.

    Parameters
    ----------
    returns:
        Per-period return series.
    n_trials:
        Number of strategy configurations tried (the multiple-testing count). Must be >= 1.

    Notes
    -----
    The DSR is computed entirely on the *per-period* Sharpe (no annualization), which is the
    correct, scale-invariant basis for the multiple-testing correction.
    """
    r = _validate_returns(returns)
    if n_trials < 1:
        raise ValidationError("n_trials must be >= 1", context={"n_trials": n_trials})

    n = r.size
    sr = float(np.mean(r) / np.std(r, ddof=1)) if np.std(r, ddof=1) > 0 else 0.0  # per-period
    skew = float(_sample_skew(r))
    kurt = float(_sample_kurtosis(r))  # non-excess (Pearson) kurtosis

    # Expected maximum Sharpe from N independent trials of zero true Sharpe (the benchmark).
    euler_mascheroni = 0.5772156649015329
    e_max_z = (1.0 - euler_mascheroni) * norm.ppf(
        1.0 - 1.0 / n_trials
    ) + euler_mascheroni * norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    sr_benchmark = e_max_z / np.sqrt(n - 1) if n > 1 else 0.0

    # Standard error of the Sharpe estimator under non-normality.
    denom = 1.0 - skew * sr + 0.25 * (kurt - 1.0) * sr**2
    if denom <= 0.0:
        return 0.0
    sr_std = np.sqrt(max(denom, 1e-12) / (n - 1))
    if sr_std <= 0.0:  # pragma: no cover - guarded above
        return 0.0
    dsr = norm.cdf((sr - sr_benchmark) / sr_std)
    return float(dsr)


def _sample_skew(r: NDArray[np.float64]) -> float:
    mean = np.mean(r)
    std = np.std(r, ddof=0)
    if std <= 0.0:
        return 0.0
    return float(np.mean(((r - mean) / std) ** 3))


def _sample_kurtosis(r: NDArray[np.float64]) -> float:
    mean = np.mean(r)
    std = np.std(r, ddof=0)
    if std <= 0.0:
        return 0.0
    return float(np.mean(((r - mean) / std) ** 4))


@dataclass(frozen=True, slots=True)
class PerformanceMetrics:
    """A bundle of backtest performance and risk statistics."""

    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    cvar_95: float
    win_rate: float
    profit_factor: float
    n_periods: int


def compute_performance_metrics(
    *,
    returns: NDArray[np.float64],
    equity_curve: NDArray[np.float64],
    trade_pnls: NDArray[np.float64] | None = None,
    periods_per_year: int = TRADING_PERIODS_PER_YEAR,
) -> PerformanceMetrics:
    """Compute the full performance-metrics bundle from a return series and equity curve.

    Parameters
    ----------
    returns:
        Per-period returns (fractional).
    equity_curve:
        Equity values over time (strictly positive).
    trade_pnls:
        Optional per-trade P&L for win-rate / profit-factor. If omitted, those are computed
        from the sign of the per-period returns.
    periods_per_year:
        Annualization factor.
    """
    r = _validate_returns(returns)
    check_positive(periods_per_year, name="periods_per_year")
    eq = np.asarray(equity_curve, dtype=np.float64)
    check_array_finite(eq, name="equity_curve")
    if eq.size < 2 or np.any(eq <= 0.0):
        raise ValidationError("equity_curve must be positive with length >= 2", context={})

    total_return = float(eq[-1] / eq[0] - 1.0)
    mean_r = float(np.mean(r))
    annualized_return = float((1.0 + mean_r) ** periods_per_year - 1.0)
    annualized_vol = float(np.std(r, ddof=1) * np.sqrt(periods_per_year))
    mdd = max_drawdown(eq)
    calmar = float(annualized_return / mdd) if mdd > 0.0 else 0.0

    pnls = r if trade_pnls is None else np.asarray(trade_pnls, dtype=np.float64)
    wins = pnls[pnls > 0.0]
    losses = pnls[pnls < 0.0]
    win_rate = float(wins.size / pnls.size) if pnls.size > 0 else 0.0
    gross_profit = float(np.sum(wins))
    gross_loss = float(-np.sum(losses))
    profit_factor = float(gross_profit / gross_loss) if gross_loss > 0.0 else float("inf")

    return PerformanceMetrics(
        total_return=total_return,
        annualized_return=annualized_return,
        annualized_volatility=annualized_vol,
        sharpe=sharpe_ratio(r, periods_per_year=periods_per_year),
        sortino=sortino_ratio(r, periods_per_year=periods_per_year),
        max_drawdown=mdd,
        calmar=calmar,
        cvar_95=conditional_value_at_risk(r, alpha=0.05),
        win_rate=win_rate,
        profit_factor=profit_factor,
        n_periods=int(r.size),
    )
