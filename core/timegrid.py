"""Deterministic time grids for simulation and decisioning.

The price simulator, calibration, and RL environment all need a consistent notion of
*time measured in years* on a discrete grid. Getting this right is a correctness issue:
option pricing and rough-volatility scaling are extremely sensitive to the day-count and
to off-by-one errors in the number of steps. This module provides a single, validated,
immutable representation used everywhere.

Conventions
-----------
* Time is expressed in **years** using a configurable annualization factor (default
  ``252`` trading days). Intraday grids subdivide a trading day.
* A grid of ``n_steps`` steps over horizon ``T`` has ``n_steps + 1`` time points
  ``t_0 = 0, ..., t_n = T``; ``times`` returns all points and ``dt`` the step size.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import assert_never

import numpy as np
from numpy.typing import NDArray

from .enums import TradingMode
from .errors import ValidationError
from .validation import check_positive

__all__ = ["TRADING_DAYS_PER_YEAR", "TimeGrid", "trading_seconds_per_day"]

#: Standard annualization factor for equity/index markets.
TRADING_DAYS_PER_YEAR: int = 252

#: Regular-trading-hours seconds in a US equity/index session (6.5 hours).
_RTH_SECONDS_PER_DAY: int = int(6.5 * 3600)


def trading_seconds_per_day() -> int:
    """Return the number of seconds in a regular-trading-hours session."""
    return _RTH_SECONDS_PER_DAY


@dataclass(frozen=True, slots=True)
class TimeGrid:
    """An immutable, uniformly-spaced time grid expressed in years.

    Parameters
    ----------
    horizon_years:
        Total horizon ``T`` in years. Must be strictly positive.
    n_steps:
        Number of intervals. The grid has ``n_steps + 1`` points. Must be >= 1.
    """

    horizon_years: float
    n_steps: int

    def __post_init__(self) -> None:
        check_positive(self.horizon_years, name="horizon_years")
        if not isinstance(self.n_steps, int) or isinstance(self.n_steps, bool):
            raise TypeError("n_steps must be an int")
        if self.n_steps < 1:
            raise ValidationError("n_steps must be >= 1", context={"n_steps": self.n_steps})

    @property
    def dt(self) -> float:
        """Uniform step size in years."""
        return self.horizon_years / self.n_steps

    @property
    def n_points(self) -> int:
        """Number of grid points (steps + 1)."""
        return self.n_steps + 1

    def times(self) -> NDArray[np.float64]:
        """Return all grid time points ``[0, ..., T]`` of length ``n_points``.

        Uses :func:`numpy.linspace` to guarantee the endpoint is exactly ``T`` (avoiding
        floating-point drift from cumulative ``dt`` addition).
        """
        return np.linspace(0.0, self.horizon_years, self.n_points, dtype=np.float64)

    def step_times(self) -> NDArray[np.float64]:
        """Return the time points excluding ``t_0 = 0`` (length ``n_steps``)."""
        return self.times()[1:]

    @classmethod
    def from_calendar_days(
        cls,
        *,
        calendar_days: float,
        steps_per_day: int,
        trading_days_per_year: int = TRADING_DAYS_PER_YEAR,
    ) -> TimeGrid:
        """Build a grid spanning ``calendar_days`` trading days.

        Parameters
        ----------
        calendar_days:
            Number of trading days to span (may be fractional).
        steps_per_day:
            Number of simulation steps per trading day.
        trading_days_per_year:
            Annualization factor.
        """
        check_positive(calendar_days, name="calendar_days")
        if steps_per_day < 1:
            raise ValidationError(
                "steps_per_day must be >= 1", context={"steps_per_day": steps_per_day}
            )
        horizon_years = calendar_days / trading_days_per_year
        n_steps = max(1, round(calendar_days * steps_per_day))
        return cls(horizon_years=horizon_years, n_steps=n_steps)

    @classmethod
    def for_mode(
        cls,
        mode: TradingMode,
        *,
        horizon_days: float,
        trading_days_per_year: int = TRADING_DAYS_PER_YEAR,
    ) -> TimeGrid:
        """Build a grid whose resolution matches the trading cadence.

        NORMAL mode (1h-1d holding) uses an hourly step; MFD mode (1m-1h holding) uses a
        one-minute step. The step counts are derived from the regular-trading-hours
        session length so intraday grids align with real market sessions.
        """
        seconds_per_day = trading_seconds_per_day()
        if mode is TradingMode.NORMAL:
            step_seconds = 3600  # hourly
        elif mode is TradingMode.MFD:
            step_seconds = 60  # one minute
        else:  # pragma: no cover - exhaustiveness guard for future enum members
            assert_never(mode)
        steps_per_day = max(1, seconds_per_day // step_seconds)
        return cls.from_calendar_days(
            calendar_days=horizon_days,
            steps_per_day=steps_per_day,
            trading_days_per_year=trading_days_per_year,
        )
