"""Tests for realized-variance and jump-robust volatility estimators."""

from __future__ import annotations

import numpy as np
import pytest

from options_engine.calibration.realized import (
    bipower_variation,
    daily_realized_variance,
    log_returns,
    log_variance_proxy,
    realized_variance,
)
from options_engine.core.errors import ValidationError


class TestLogReturns:
    def test_basic(self) -> None:
        prices = np.array([100.0, 110.0, 99.0])
        r = log_returns(prices)
        np.testing.assert_allclose(r, np.log([110 / 100, 99 / 110]))

    def test_rejects_short(self) -> None:
        with pytest.raises(ValidationError):
            log_returns(np.array([100.0]))

    def test_rejects_nonpositive(self) -> None:
        with pytest.raises(ValidationError):
            log_returns(np.array([100.0, -1.0]))


class TestRealizedVariance:
    def test_sum_of_squares(self) -> None:
        r = np.array([0.01, -0.02, 0.005])
        assert realized_variance(r) == pytest.approx(np.sum(r**2))

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValidationError):
            realized_variance(np.array([]))


class TestBipowerVariation:
    def test_robust_to_single_jump(self) -> None:
        # BV should be far less affected by one huge return than RV is.
        rng = np.random.default_rng(0)
        r = rng.normal(0.0, 0.01, size=500)
        rv_clean = realized_variance(r)
        bv_clean = bipower_variation(r)
        # Inject a large jump.
        r_jump = r.copy()
        r_jump[250] = 0.20
        rv_jump = realized_variance(r_jump)
        bv_jump = bipower_variation(r_jump)
        # RV jumps a lot; BV barely moves.
        assert rv_jump - rv_clean > 0.03
        assert abs(bv_jump - bv_clean) < 0.01

    def test_approximates_rv_without_jumps(self) -> None:
        rng = np.random.default_rng(1)
        r = rng.normal(0.0, 0.01, size=5000)
        rv = realized_variance(r)
        bv = bipower_variation(r)
        # Without jumps, BV and RV both estimate integrated variance.
        assert bv == pytest.approx(rv, rel=0.1)

    def test_requires_two(self) -> None:
        with pytest.raises(ValidationError):
            bipower_variation(np.array([0.01]))


class TestDailyRealizedVariance:
    def test_shape_and_annualization(self) -> None:
        r = np.full(26 * 3, 0.001)
        daily = daily_realized_variance(r, steps_per_day=26, annualize=True)
        assert daily.shape == (3,)
        expected = 26 * (0.001**2) * 252
        np.testing.assert_allclose(daily, expected)

    def test_no_annualization(self) -> None:
        r = np.full(10, 0.001)
        daily = daily_realized_variance(r, steps_per_day=10, annualize=False)
        assert daily[0] == pytest.approx(10 * 0.001**2)

    def test_rejects_ragged(self) -> None:
        with pytest.raises(ValidationError):
            daily_realized_variance(np.ones(25), steps_per_day=26)


class TestLogVarianceProxy:
    def test_log(self) -> None:
        v = np.array([0.04, 0.09])
        np.testing.assert_allclose(log_variance_proxy(v), np.log(v))

    def test_rejects_nonpositive(self) -> None:
        with pytest.raises(ValidationError):
            log_variance_proxy(np.array([0.04, 0.0]))
