"""Tests for forward-variance (xi0) estimation."""

from __future__ import annotations

import numpy as np
import pytest

from options_engine.calibration.forward_variance import (
    estimate_xi0_curve,
    estimate_xi0_level,
)
from options_engine.core.errors import CalibrationError, ValidationError


class TestEstimateXi0Level:
    def test_mean_of_series(self) -> None:
        rv = np.array([0.03, 0.04, 0.05])
        est = estimate_xi0_level(rv)
        assert est.value == pytest.approx(0.04)
        assert est.name == "xi0_level"
        assert est.std_error > 0.0

    def test_rejects_nonpositive(self) -> None:
        with pytest.raises(ValidationError):
            estimate_xi0_level(np.array([0.03, -0.01]))

    def test_rejects_too_short(self) -> None:
        with pytest.raises(CalibrationError):
            estimate_xi0_level(np.array([0.04]))


class TestEstimateXi0Curve:
    def test_flat_iv_gives_flat_forward_variance(self) -> None:
        # Constant implied vol => constant forward variance equal to sigma^2.
        maturities = np.array([0.25, 0.5, 1.0])
        vols = np.array([0.2, 0.2, 0.2])
        curve = estimate_xi0_curve(maturities, vols)
        np.testing.assert_allclose(curve(np.array([0.1, 0.5, 1.0])), 0.04, rtol=1e-9)

    def test_upward_term_structure(self) -> None:
        # Total variance must be increasing; forward variances stay positive.
        maturities = np.array([0.25, 0.5, 1.0])
        vols = np.array([0.18, 0.20, 0.22])
        curve = estimate_xi0_curve(maturities, vols)
        # First-interval forward variance equals sigma_0^2.
        assert curve.at(0.1) == pytest.approx(0.18**2, rel=1e-9)
        # Later forward variance is higher than the first.
        assert curve.at(0.9) > curve.at(0.1)

    def test_detects_calendar_arbitrage(self) -> None:
        # Decreasing total variance => negative forward variance => arbitrage.
        maturities = np.array([0.25, 0.5])
        vols = np.array([0.40, 0.10])  # total var 0.04 -> 0.005 (decreasing)
        with pytest.raises(CalibrationError):
            estimate_xi0_curve(maturities, vols)

    def test_rejects_unsorted_maturities(self) -> None:
        with pytest.raises(ValidationError):
            estimate_xi0_curve(np.array([0.5, 0.25]), np.array([0.2, 0.2]))

    def test_rejects_nonpositive_vol(self) -> None:
        with pytest.raises(ValidationError):
            estimate_xi0_curve(np.array([0.5]), np.array([0.0]))
