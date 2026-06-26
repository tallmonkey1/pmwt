r"""Walk-forward calibration windows (SPEC §2.4: "all calibration is walk-forward").

Re-fitting parameters on a rolling window and only ever using them on *subsequent*,
out-of-sample data is the discipline that prevents look-ahead bias from contaminating the
whole engine. This module produces the (train-window, as-of-date) schedule and runs the
calibrator over each window, returning a time-ordered list of results that downstream code
consumes strictly causally (each result is valid only after its ``as_of`` timestamp).
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..core.errors import ValidationError
from ..core.random import RandomFactory
from .calibrator import CalibrationConfig, calibrate_rbergomi
from .results import CalibrationResult

__all__ = ["WalkForwardWindow", "generate_windows", "run_walk_forward"]


@dataclass(frozen=True, slots=True)
class WalkForwardWindow:
    """A single train window expressed in whole trading days (index-based)."""

    train_start_day: int
    train_end_day: int  # exclusive

    def __post_init__(self) -> None:
        if self.train_start_day < 0:
            raise ValidationError(
                "train_start_day must be non-negative", context={"start": self.train_start_day}
            )
        if self.train_end_day <= self.train_start_day:
            raise ValidationError(
                "train_end_day must exceed train_start_day",
                context={"start": self.train_start_day, "end": self.train_end_day},
            )

    @property
    def n_days(self) -> int:
        """Number of trading days in the window."""
        return self.train_end_day - self.train_start_day


def generate_windows(
    *, total_days: int, train_days: int, step_days: int, anchored: bool = False
) -> list[WalkForwardWindow]:
    """Generate rolling (or anchored/expanding) walk-forward windows.

    Parameters
    ----------
    total_days:
        Total number of trading days available.
    train_days:
        Length of each training window (the rolling-window case).
    step_days:
        How far to advance the window each step.
    anchored:
        If True, windows expand from day 0 (anchored/expanding) instead of rolling; each
        window still ends ``step_days`` later than the previous one.

    Returns
    -------
    list[WalkForwardWindow]
        Time-ordered windows; the caller calibrates on each and applies the result only to
        data after ``train_end_day``.
    """
    if total_days <= 0 or train_days <= 0 or step_days <= 0:
        raise ValidationError(
            "total_days, train_days and step_days must be positive",
            context={"total": total_days, "train": train_days, "step": step_days},
        )
    if train_days > total_days:
        raise ValidationError(
            "train_days cannot exceed total_days",
            context={"train": train_days, "total": total_days},
        )

    windows: list[WalkForwardWindow] = []
    end = train_days
    while end <= total_days:
        start = 0 if anchored else end - train_days
        windows.append(WalkForwardWindow(train_start_day=start, train_end_day=end))
        end += step_days
    return windows


def run_walk_forward(
    intraday_prices: NDArray[np.float64],
    *,
    rng_factory: RandomFactory,
    windows: Sequence[WalkForwardWindow],
    config: CalibrationConfig | None = None,
    session_dates: Sequence[_dt.datetime] | None = None,
) -> list[CalibrationResult]:
    """Calibrate the rBergomi model over each walk-forward window.

    Parameters
    ----------
    intraday_prices:
        Full intraday price history whose length is a whole number of trading days.
    rng_factory:
        Reproducible randomness; each window draws from a distinct sub-stream so windows are
        independent yet collectively reproducible.
    windows:
        The walk-forward schedule (see :func:`generate_windows`).
    config:
        Calibration settings.
    session_dates:
        Optional per-day timestamps (length = total days) used to stamp each result's
        provenance. If omitted, results are stamped with the current UTC time.

    Returns
    -------
    list[CalibrationResult]
        One result per window, in window order.
    """
    cfg = config or CalibrationConfig()
    prices = np.asarray(intraday_prices, dtype=np.float64)
    if prices.size % cfg.steps_per_day != 0:
        raise ValidationError(
            "price series length must be a whole number of trading days",
            context={"size": int(prices.size), "steps_per_day": cfg.steps_per_day},
        )
    total_days = prices.size // cfg.steps_per_day
    if session_dates is not None and len(session_dates) != total_days:
        raise ValidationError(
            "session_dates length must equal the number of trading days",
            context={"dates": len(session_dates), "days": total_days},
        )

    results: list[CalibrationResult] = []
    for i, window in enumerate(windows):
        if window.train_end_day > total_days:
            raise ValidationError(
                "window exceeds available days",
                context={"end": window.train_end_day, "total": total_days},
            )
        start_idx = window.train_start_day * cfg.steps_per_day
        end_idx = window.train_end_day * cfg.steps_per_day
        segment = prices[start_idx:end_idx]

        data_start = data_end = as_of = None
        if session_dates is not None:
            data_start = session_dates[window.train_start_day]
            data_end = session_dates[window.train_end_day - 1]
            as_of = data_end

        # Distinct, reproducible RNG sub-stream per window.
        window_factory = RandomFactory(rng_factory.seed + 1 + i)
        results.append(
            calibrate_rbergomi(
                segment,
                rng_factory=window_factory,
                config=cfg,
                data_start=data_start,
                data_end=data_end,
                now=as_of,
            )
        )
    return results
