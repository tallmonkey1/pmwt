"""Integration tests for the end-to-end calibrator and walk-forward driver.

These assert that the full pipeline runs deterministically and yields a valid, in-domain,
fully-provenanced result. They deliberately do NOT assert tight parameter recovery from
noisy realized variance: as documented in SPEC §0 and the estimator docstrings, estimating
H/rho from noisy RV is biased, and pretending otherwise would be dishonest. Tight recovery
is validated against *clean latent* inputs in the per-estimator tests.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from options_engine.calibration import (
    CalibrationConfig,
    calibrate_rbergomi,
    generate_windows,
    run_walk_forward,
)
from options_engine.calibration.results import CalibrationResult
from options_engine.core.errors import CalibrationError, ValidationError
from options_engine.core.random import RandomFactory
from options_engine.core.timegrid import TimeGrid
from options_engine.models.rbergomi import (
    ForwardVariance,
    HybridSimulator,
    RBergomiParams,
)


def _synthetic_intraday_prices(*, days: int, steps_per_day: int, seed: int = 11) -> np.ndarray:
    params = RBergomiParams(
        hurst=0.11, eta=1.6, rho=-0.7, forward_variance=ForwardVariance.flat(0.04)
    )
    grid = TimeGrid(horizon_years=days / 252, n_steps=days * steps_per_day)
    paths = HybridSimulator(params, rng_factory=RandomFactory(seed)).simulate(
        grid=grid, n_paths=1, initial_spot=100.0
    )
    return paths.spot[0][1:]  # exactly days * steps_per_day points


@pytest.mark.slow
class TestCalibrateRBergomi:
    def test_pipeline_produces_valid_result(self) -> None:
        prices = _synthetic_intraday_prices(days=200, steps_per_day=26)
        cfg = CalibrationConfig(steps_per_day=26, n_sim_paths_eta=120, n_sim_paths_rho=40)
        result = calibrate_rbergomi(prices, rng_factory=RandomFactory(99), config=cfg)
        assert isinstance(result, CalibrationResult)
        # All parameters in their valid domains (enforced by CalibrationResult, asserted here).
        assert 0.0 < result.hurst.value < 0.5
        assert result.eta.value > 0.0
        assert -1.0 <= result.rho.value <= 1.0
        assert result.xi0_level.value > 0.0
        assert "hurst_r_squared" in result.diagnostics

    def test_reproducible(self) -> None:
        prices = _synthetic_intraday_prices(days=120, steps_per_day=26)
        cfg = CalibrationConfig(steps_per_day=26, n_sim_paths_eta=80, n_sim_paths_rho=30)
        a = calibrate_rbergomi(prices, rng_factory=RandomFactory(7), config=cfg)
        b = calibrate_rbergomi(prices, rng_factory=RandomFactory(7), config=cfg)
        assert a.hurst.value == b.hurst.value
        assert a.eta.value == b.eta.value
        assert a.rho.value == b.rho.value

    def test_provenance_timestamps(self) -> None:
        prices = _synthetic_intraday_prices(days=120, steps_per_day=26)
        cfg = CalibrationConfig(steps_per_day=26, n_sim_paths_eta=60, n_sim_paths_rho=20)
        start = dt.datetime(2023, 1, 1, tzinfo=dt.UTC)
        end = dt.datetime(2023, 6, 1, tzinfo=dt.UTC)
        result = calibrate_rbergomi(
            prices,
            rng_factory=RandomFactory(1),
            config=cfg,
            data_start=start,
            data_end=end,
            now=end,
        )
        assert result.data_start == start
        assert result.data_end == end
        assert result.as_of == end

    def test_rejects_short_history(self) -> None:
        prices = _synthetic_intraday_prices(days=40, steps_per_day=26)
        cfg = CalibrationConfig(steps_per_day=26, min_days=60)
        with pytest.raises(CalibrationError):
            calibrate_rbergomi(prices, rng_factory=RandomFactory(1), config=cfg)

    def test_rejects_ragged_length(self) -> None:
        prices = np.full(26 * 100 + 3, 100.0)
        cfg = CalibrationConfig(steps_per_day=26)
        with pytest.raises(CalibrationError):
            calibrate_rbergomi(prices, rng_factory=RandomFactory(1), config=cfg)


class TestConfigValidation:
    def test_rejects_low_steps_per_day(self) -> None:
        with pytest.raises(CalibrationError):
            CalibrationConfig(steps_per_day=1)

    def test_rejects_low_min_days(self) -> None:
        with pytest.raises(CalibrationError):
            CalibrationConfig(min_days=10)


class TestWalkForward:
    def test_generate_rolling_windows(self) -> None:
        windows = generate_windows(total_days=300, train_days=100, step_days=50)
        assert windows[0].train_start_day == 0
        assert windows[0].train_end_day == 100
        assert windows[1].train_start_day == 50
        assert all(w.n_days == 100 for w in windows)

    def test_generate_anchored_windows(self) -> None:
        windows = generate_windows(total_days=300, train_days=100, step_days=50, anchored=True)
        # Anchored windows all start at 0 and grow.
        assert all(w.train_start_day == 0 for w in windows)
        assert windows[-1].train_end_day > windows[0].train_end_day

    def test_rejects_train_exceeding_total(self) -> None:
        with pytest.raises(ValidationError):
            generate_windows(total_days=50, train_days=100, step_days=10)

    @pytest.mark.slow
    def test_run_walk_forward(self) -> None:
        prices = _synthetic_intraday_prices(days=240, steps_per_day=26)
        cfg = CalibrationConfig(steps_per_day=26, n_sim_paths_eta=60, n_sim_paths_rho=20)
        windows = generate_windows(total_days=240, train_days=120, step_days=60)
        results = run_walk_forward(
            prices, rng_factory=RandomFactory(3), windows=windows, config=cfg
        )
        assert len(results) == len(windows)
        for r in results:
            assert 0.0 < r.hurst.value < 0.5
            assert r.eta.value > 0.0
