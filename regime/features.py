r"""Point-in-time feature construction for regime detection (SPEC §2.6).

The features must be **leakage-free**: each row at time ``t`` may use only information
available up to and including ``t``. We therefore build features from *trailing* windows of
realized volatility and returns -- never centred or forward-looking windows. This is a hard
correctness requirement; a regime model trained on look-ahead features produces
spectacular, fraudulent backtests.

Features (per day ``t``):

0. ``log_rv``            -- log of trailing realized variance (the dominant vol-level axis).
1. ``rv_change``         -- change in log realized variance vs. the previous step
                            (vol acceleration; distinguishes calm vs. stressing markets).
2. ``abs_return``        -- absolute daily return (instantaneous shock magnitude).
3. ``downside_ratio``    -- share of the trailing window with negative returns
                            (captures the asymmetry typical of stressed regimes).

The first feature is placed first deliberately: :class:`GaussianHMM` seeds its EM by
quantile-binning the first feature, so leading with the vol level gives well-ordered states.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..core.errors import ValidationError
from ..core.timegrid import TRADING_DAYS_PER_YEAR
from ..core.validation import check_array_finite

__all__ = ["N_REGIME_FEATURES", "build_regime_features"]

#: Number of regime features produced by :func:`build_regime_features`.
N_REGIME_FEATURES: int = 4


def build_regime_features(
    daily_returns: NDArray[np.float64],
    *,
    window: int = 21,
    trading_days_per_year: int = TRADING_DAYS_PER_YEAR,
) -> NDArray[np.float64]:
    r"""Return a leakage-free regime feature matrix from a daily-return series.

    Parameters
    ----------
    daily_returns:
        One-dimensional series of daily log-returns.
    window:
        Trailing window length (in days) for the realized-volatility features. Must be
        ``>= 2`` and shorter than the series.
    trading_days_per_year:
        Annualization factor for realized variance.

    Returns
    -------
    numpy.ndarray
        Feature matrix of shape ``(T - window, N_REGIME_FEATURES)``. The first ``window``
        observations are consumed to form the initial trailing window, so the output is
        shorter than the input -- this alignment is what guarantees no look-ahead.
    """
    r = np.asarray(daily_returns, dtype=np.float64)
    check_array_finite(r, name="daily_returns")
    if r.ndim != 1:
        raise ValidationError("daily_returns must be 1-D", context={"ndim": r.ndim})
    if window < 2:
        raise ValidationError("window must be >= 2", context={"window": window})
    if r.size <= window:
        raise ValidationError(
            "series must be longer than the window",
            context={"size": int(r.size), "window": window},
        )

    n_out = r.size - window
    features = np.empty((n_out, N_REGIME_FEATURES), dtype=np.float64)
    prev_log_rv = np.nan
    for i in range(n_out):
        # Trailing window ending at index (window + i - 1), inclusive of the current day.
        win = r[i : i + window]
        realized_var = float(np.mean(win**2)) * trading_days_per_year
        realized_var = max(realized_var, 1e-12)
        log_rv = float(np.log(realized_var))
        rv_change = 0.0 if np.isnan(prev_log_rv) else log_rv - prev_log_rv
        abs_return = float(abs(r[i + window - 1]))
        downside_ratio = float(np.mean(win < 0.0))

        features[i, 0] = log_rv
        features[i, 1] = rv_change
        features[i, 2] = abs_return
        features[i, 3] = downside_ratio
        prev_log_rv = log_rv

    return features
