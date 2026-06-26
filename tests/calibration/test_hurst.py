"""Tests for the Hurst-exponent estimator.

The decisive test is parameter recovery: on a clean latent log-variance path with a known
Hurst exponent, the estimator must recover it accurately.
"""

from __future__ import annotations

import numpy as np
import pytest

from options_engine.calibration.hurst import estimate_hurst
from options_engine.core.errors import CalibrationError, ValidationError

from .conftest import SyntheticPath, make_path


class TestEstimateHurst:
    @pytest.mark.slow
    @pytest.mark.parametrize("true_h", [0.07, 0.1, 0.15, 0.25])
    def test_recovers_known_hurst(self, true_h: float) -> None:
        path = make_path(hurst=true_h, seed=3)
        est = estimate_hurst(path.log_variance)
        # Recovery within 0.03 absolute on a clean latent path.
        assert est.value == pytest.approx(true_h, abs=0.03)
        assert est.r_squared is not None and est.r_squared > 0.95

    def test_returns_named_estimate(self, clean_path: SyntheticPath) -> None:
        est = estimate_hurst(clean_path.log_variance)
        assert est.name == "hurst"
        assert 0.0 < est.value < 0.5
        assert est.std_error >= 0.0
        assert est.n_observations == clean_path.log_variance.size

    def test_confidence_interval(self, clean_path: SyntheticPath) -> None:
        est = estimate_hurst(clean_path.log_variance)
        lo, hi = est.confidence_interval(2.0)
        assert lo <= est.value <= hi

    def test_rejects_short_series(self) -> None:
        with pytest.raises(CalibrationError):
            estimate_hurst(np.linspace(-3.0, -2.0, 10))

    def test_rejects_non_finite(self) -> None:
        bad = np.full(50, -3.0)
        bad[10] = np.nan
        with pytest.raises(ValidationError):
            estimate_hurst(bad)

    def test_rejects_constant_series(self) -> None:
        # A constant log-variance has zero increments -> degenerate moments.
        with pytest.raises(CalibrationError):
            estimate_hurst(np.full(100, -3.2))

    def test_custom_moment_orders_validated(self, clean_path: SyntheticPath) -> None:
        with pytest.raises(ValidationError):
            estimate_hurst(clean_path.log_variance, moment_orders=(1.0, -1.0))
