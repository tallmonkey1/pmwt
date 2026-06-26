"""Tests for calibration result containers and their validation/staleness logic."""

from __future__ import annotations

import datetime as dt

import pytest

from options_engine.calibration.results import CalibrationResult, ParameterEstimate
from options_engine.core.errors import ValidationError


def _estimate(name: str, value: float) -> ParameterEstimate:
    return ParameterEstimate(name=name, value=value, std_error=0.01, n_observations=100)


class TestParameterEstimate:
    def test_valid(self) -> None:
        est = _estimate("hurst", 0.1)
        assert est.value == 0.1

    def test_confidence_interval(self) -> None:
        est = ParameterEstimate(name="x", value=1.0, std_error=0.1, n_observations=10)
        lo, hi = est.confidence_interval(2.0)
        assert lo == pytest.approx(0.8)
        assert hi == pytest.approx(1.2)

    def test_rejects_empty_name(self) -> None:
        with pytest.raises(ValidationError):
            ParameterEstimate(name="", value=1.0, std_error=0.1, n_observations=10)

    def test_rejects_negative_std_error(self) -> None:
        with pytest.raises(ValidationError):
            ParameterEstimate(name="x", value=1.0, std_error=-0.1, n_observations=10)

    def test_rejects_nonpositive_obs(self) -> None:
        with pytest.raises(ValidationError):
            ParameterEstimate(name="x", value=1.0, std_error=0.1, n_observations=0)

    def test_rejects_bad_r_squared(self) -> None:
        with pytest.raises(ValidationError):
            ParameterEstimate(name="x", value=1.0, std_error=0.1, n_observations=10, r_squared=1.5)


class TestCalibrationResult:
    def _result(self, **overrides: object) -> CalibrationResult:
        now = dt.datetime(2024, 1, 10, tzinfo=dt.UTC)
        defaults: dict[str, object] = {
            "hurst": _estimate("hurst", 0.1),
            "eta": _estimate("eta", 1.5),
            "rho": _estimate("rho", -0.7),
            "xi0_level": _estimate("xi0_level", 0.04),
            "as_of": now,
            "data_start": now - dt.timedelta(days=30),
            "data_end": now,
        }
        defaults.update(overrides)
        return CalibrationResult(**defaults)  # type: ignore[arg-type]

    def test_valid_construction(self) -> None:
        res = self._result()
        assert res.hurst.value == 0.1
        assert not res.jumps_detected

    def test_rejects_out_of_domain_hurst(self) -> None:
        with pytest.raises(ValidationError):
            self._result(hurst=_estimate("hurst", 0.6))

    def test_rejects_nonpositive_eta(self) -> None:
        with pytest.raises(ValidationError):
            self._result(eta=_estimate("eta", -1.0))

    def test_rejects_out_of_range_rho(self) -> None:
        with pytest.raises(ValidationError):
            self._result(rho=_estimate("rho", -1.5))

    def test_rejects_inverted_dates(self) -> None:
        now = dt.datetime(2024, 1, 10, tzinfo=dt.UTC)
        with pytest.raises(ValidationError):
            self._result(data_start=now, data_end=now - dt.timedelta(days=1))

    def test_staleness(self) -> None:
        res = self._result()
        future = res.as_of + dt.timedelta(days=10)
        assert res.is_stale(now=future, max_age=dt.timedelta(days=5))
        assert not res.is_stale(now=future, max_age=dt.timedelta(days=30))
