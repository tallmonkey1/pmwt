r"""Hurst-exponent estimation via the log-moment scaling method.

This implements the estimator of Gatheral, Jaisson & Rosenbaum (2018, *Volatility is
rough*). For a rough-volatility log-variance process the ``q``-th absolute-moment of
log-variance increments scales as a power of the lag:

.. math::

    m(q, \Delta) = \mathbb{E}\big[\,|\,\log v_{t+\Delta} - \log v_t\,|^{q}\,\big]
                 \;\propto\; \Delta^{\,q H}.

Hence for each moment order ``q`` the slope of :math:`\log m(q, \Delta)` against
:math:`\log \Delta` equals :math:`q H`, and regressing those per-``q`` slopes against
``q`` (through the origin) yields a single, robust estimate of ``H``. The smallness of the
estimated ``H`` (empirically ~0.1) is the signature of rough volatility.

**Honesty note (SPEC §0).** When the input log-variance is computed from *noisy* realized
variance, microstructure/measurement noise biases ``H`` upward (the noise injects spurious
high-frequency roughness that the estimator misreads as a less-rough, higher-``H`` process).
This estimator is exact on a clean log-variance proxy and *upward-biased* on a noisy one;
the bias is documented, surfaced via the fit ``r_squared``, and must be mitigated upstream
(longer aggregation windows, pre-averaging) rather than hidden here.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..core.errors import CalibrationError, ValidationError
from .results import ParameterEstimate

__all__ = ["DEFAULT_MOMENT_ORDERS", "estimate_hurst"]

#: Default absolute-moment orders. A spread of orders makes the estimate robust to the
#: tail behaviour of the increments.
DEFAULT_MOMENT_ORDERS: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 3.0)


def _resolve_lags(
    lags: NDArray[np.int_] | None, *, n: int, max_lag: int | None
) -> NDArray[np.int_]:
    """Validate user-supplied lags or build the default lag range for a series of length n."""
    if lags is None:
        upper = max_lag if max_lag is not None else min(n // 4, 50)
        if upper < 2:
            raise CalibrationError(
                "series too short to form a lag range", context={"n": int(n), "upper": upper}
            )
        return np.arange(1, upper + 1, dtype=np.int_)

    lag_array = np.asarray(lags, dtype=np.int_)
    if lag_array.ndim != 1 or lag_array.size < 2:
        raise ValidationError("lags must be a 1-D array of length >= 2", context={})
    if np.any(lag_array < 1) or np.any(lag_array >= n):
        raise ValidationError("lags must lie in [1, n-1]", context={"n": int(n)})
    return lag_array


def estimate_hurst(
    log_variance: NDArray[np.float64],
    *,
    lags: NDArray[np.int_] | None = None,
    moment_orders: tuple[float, ...] = DEFAULT_MOMENT_ORDERS,
    max_lag: int | None = None,
) -> ParameterEstimate:
    r"""Estimate the Hurst exponent from a log-variance series.

    Parameters
    ----------
    log_variance:
        One-dimensional series of :math:`\log v_t` (e.g. from
        :func:`options_engine.calibration.realized.log_variance_proxy`). Must be regularly
        sampled.
    lags:
        Integer lags :math:`\Delta` (in sampling steps) at which to compute the moments. If
        ``None``, a default geometric-ish range ``1 .. max_lag`` is used.
    moment_orders:
        The absolute-moment orders ``q``. Must be positive.
    max_lag:
        Upper bound for the default lag range. Defaults to ``min(len // 4, 50)`` so that
        each moment is estimated from a large number of overlapping increments.

    Returns
    -------
    ParameterEstimate
        The estimate of ``H`` with a standard error (from the slope-vs-``q`` regression)
        and the regression ``r_squared`` as a fit-quality diagnostic.

    Raises
    ------
    CalibrationError
        If the estimated ``H`` falls outside the model's valid open interval ``(0, 0.5)``
        or the regression is degenerate.
    """
    lv = np.asarray(log_variance, dtype=np.float64)
    if lv.ndim != 1:
        raise ValidationError("log_variance must be 1-D", context={"ndim": lv.ndim})
    if not np.all(np.isfinite(lv)):
        raise ValidationError("log_variance contains non-finite values", context={})
    n = lv.size
    if n < 20:
        raise CalibrationError(
            "log_variance series is too short for a reliable Hurst estimate",
            context={"n": int(n)},
        )
    if any(q <= 0.0 for q in moment_orders):
        raise ValidationError(
            "moment orders must be positive", context={"orders": list(moment_orders)}
        )

    lag_array = _resolve_lags(lags, n=n, max_lag=max_lag)
    log_lags = np.log(lag_array.astype(np.float64))

    # For each moment order, slope of log m(q, lag) vs log lag estimates q * H.
    slopes = np.empty(len(moment_orders), dtype=np.float64)
    for i, q in enumerate(moment_orders):
        log_m = np.empty(lag_array.size, dtype=np.float64)
        for j, lag in enumerate(lag_array):
            increments = np.abs(lv[lag:] - lv[:-lag])
            moment = float(np.mean(increments**q))
            if moment <= 0.0:
                raise CalibrationError(
                    "degenerate (zero) moment encountered; series may be constant",
                    context={"q": q, "lag": int(lag)},
                )
            log_m[j] = np.log(moment)
        slope = float(np.polyfit(log_lags, log_m, 1)[0])
        slopes[i] = slope

    # Regress slopes on q through the origin: slope_i ~ H * q_i. The least-squares solution
    # is H = sum(q*slope) / sum(q^2). Its standard error comes from the residuals.
    q_arr = np.asarray(moment_orders, dtype=np.float64)
    denom = float(np.sum(q_arr**2))
    if denom <= 0.0:  # pragma: no cover - guarded by positive-q validation
        raise CalibrationError("degenerate moment-order design", context={})
    hurst = float(np.sum(q_arr * slopes) / denom)

    residuals = slopes - hurst * q_arr
    dof = max(len(moment_orders) - 1, 1)
    residual_var = float(np.sum(residuals**2) / dof)
    std_error = float(np.sqrt(residual_var / denom))

    # Coefficient of determination of the through-origin fit.
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum(slopes**2))
    r_squared = max(0.0, 1.0 - ss_res / ss_tot) if ss_tot > 0.0 else 0.0

    if not (0.0 < hurst < 0.5):
        raise CalibrationError(
            "estimated Hurst exponent outside the valid open interval (0, 0.5)",
            context={"hurst": hurst, "r_squared": r_squared},
        )

    return ParameterEstimate(
        name="hurst",
        value=hurst,
        std_error=std_error,
        n_observations=int(n),
        r_squared=r_squared,
    )
