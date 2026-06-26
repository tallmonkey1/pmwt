"""Tests for the time-grid abstraction."""

from __future__ import annotations

import pytest

from options_engine.core.enums import TradingMode
from options_engine.core.errors import ValidationError
from options_engine.core.timegrid import (
    TRADING_DAYS_PER_YEAR,
    TimeGrid,
    trading_seconds_per_day,
)


def test_basic_grid_properties() -> None:
    grid = TimeGrid(horizon_years=1.0, n_steps=4)
    assert grid.n_points == 5
    assert grid.dt == pytest.approx(0.25)
    times = grid.times()
    assert times[0] == 0.0
    assert times[-1] == pytest.approx(1.0)
    assert len(times) == 5


def test_endpoint_is_exact() -> None:
    # linspace guarantees the endpoint equals the horizon exactly (no drift).
    grid = TimeGrid(horizon_years=0.3, n_steps=1000)
    assert grid.times()[-1] == 0.3


def test_step_times_excludes_zero() -> None:
    grid = TimeGrid(horizon_years=1.0, n_steps=3)
    step_times = grid.step_times()
    assert len(step_times) == 3
    assert step_times[0] > 0.0


def test_invalid_horizon_rejected() -> None:
    with pytest.raises(ValidationError):
        TimeGrid(horizon_years=0.0, n_steps=10)


def test_invalid_n_steps_rejected() -> None:
    with pytest.raises(ValidationError):
        TimeGrid(horizon_years=1.0, n_steps=0)


def test_n_steps_type_checked() -> None:
    with pytest.raises(TypeError):
        TimeGrid(horizon_years=1.0, n_steps=2.5)  # type: ignore[arg-type]


def test_from_calendar_days() -> None:
    grid = TimeGrid.from_calendar_days(calendar_days=5, steps_per_day=1)
    assert grid.n_steps == 5
    assert grid.horizon_years == pytest.approx(5 / TRADING_DAYS_PER_YEAR)


def test_from_calendar_days_validates() -> None:
    with pytest.raises(ValidationError):
        TimeGrid.from_calendar_days(calendar_days=0, steps_per_day=1)
    with pytest.raises(ValidationError):
        TimeGrid.from_calendar_days(calendar_days=1, steps_per_day=0)


def test_for_mode_normal_uses_hourly_steps() -> None:
    grid = TimeGrid.for_mode(TradingMode.NORMAL, horizon_days=1.0)
    # 6.5h session at hourly resolution -> 6 steps per day.
    expected_steps_per_day = trading_seconds_per_day() // 3600
    assert grid.n_steps == expected_steps_per_day


def test_for_mode_mfd_uses_minute_steps() -> None:
    grid = TimeGrid.for_mode(TradingMode.MFD, horizon_days=1.0)
    expected_steps_per_day = trading_seconds_per_day() // 60
    assert grid.n_steps == expected_steps_per_day


def test_grid_is_immutable() -> None:
    grid = TimeGrid(horizon_years=1.0, n_steps=4)
    with pytest.raises((AttributeError, TypeError)):
        grid.n_steps = 5  # type: ignore[misc]
