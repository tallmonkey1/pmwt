"""Tests for surrogate distributional metrics."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from options_engine.core.errors import ValidationError
from options_engine.surrogate.metrics import (
    calibration_error,
    crps_from_quantiles,
    pit_values,
    quantile_loss_numpy,
    wasserstein1_from_quantiles,
)

LEVELS = np.linspace(0.01, 0.99, 99)


def _normal_q(mu: float = 0.0, sigma: float = 0.1) -> np.ndarray:
    return norm.ppf(LEVELS, loc=mu, scale=sigma)


class TestWasserstein:
    def test_zero_for_identical(self) -> None:
        q = _normal_q()
        assert wasserstein1_from_quantiles(q, q, LEVELS) == pytest.approx(0.0, abs=1e-12)

    def test_location_shift_equals_shift(self) -> None:
        # W1 between two distributions differing by a location shift c equals |c|.
        q1 = _normal_q(mu=0.0)
        q2 = _normal_q(mu=0.05)
        assert wasserstein1_from_quantiles(q1, q2, LEVELS) == pytest.approx(0.05, abs=2e-3)

    def test_positive_for_different_scale(self) -> None:
        q1 = _normal_q(sigma=0.1)
        q2 = _normal_q(sigma=0.2)
        assert wasserstein1_from_quantiles(q1, q2, LEVELS) > 0.0


class TestCalibration:
    def test_well_calibrated_low_error(self) -> None:
        sigma = 0.1
        q = _normal_q(sigma=sigma)
        rng = np.random.default_rng(0)
        samples = rng.normal(0.0, sigma, size=50_000)
        err = calibration_error(samples, q, LEVELS)
        assert err < 0.02

    def test_miscalibrated_high_error(self) -> None:
        q = _normal_q(sigma=0.1)
        rng = np.random.default_rng(1)
        # Samples are much wider than the forecast -> poor calibration.
        samples = rng.normal(0.0, 0.3, size=50_000)
        assert calibration_error(samples, q, LEVELS) > 0.1

    def test_pit_uniform_when_calibrated(self) -> None:
        sigma = 0.1
        q = _normal_q(sigma=sigma)
        rng = np.random.default_rng(2)
        samples = rng.normal(0.0, sigma, size=50_000)
        pit = pit_values(samples, q, LEVELS)
        # Uniform PIT has mean ~0.5 and std ~1/sqrt(12) ~ 0.289.
        assert float(np.mean(pit)) == pytest.approx(0.5, abs=0.02)
        assert float(np.std(pit)) == pytest.approx(1 / np.sqrt(12), abs=0.02)


class TestCRPS:
    def test_lower_for_better_forecast(self) -> None:
        rng = np.random.default_rng(3)
        samples = rng.normal(0.0, 0.1, size=20_000)
        good = _normal_q(sigma=0.1)
        bad = _normal_q(sigma=0.3)
        assert crps_from_quantiles(good, samples, LEVELS) < crps_from_quantiles(
            bad, samples, LEVELS
        )


class TestQuantileLossNumpy:
    def test_zero_for_identical(self) -> None:
        q = _normal_q()
        assert quantile_loss_numpy(q, q, LEVELS) == pytest.approx(0.0, abs=1e-12)


class TestValidation:
    def test_mismatched_lengths(self) -> None:
        with pytest.raises(ValidationError):
            wasserstein1_from_quantiles(_normal_q(), _normal_q()[:-1], LEVELS)

    def test_levels_must_be_increasing(self) -> None:
        with pytest.raises(ValidationError):
            wasserstein1_from_quantiles(
                np.array([0.0, 0.1]), np.array([0.0, 0.1]), np.array([0.5, 0.3])
            )
