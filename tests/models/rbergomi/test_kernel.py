"""Tests for the Volterra-kernel mathematics."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from options_engine.core.errors import ValidationError
from options_engine.models.rbergomi.kernel import (
    cross_covariance,
    hybrid_discrete_covariance,
    hybrid_weights,
    volterra_autocovariance,
)


class TestVolterraAutocovariance:
    @pytest.mark.parametrize("hurst", [0.05, 0.1, 0.3, 0.45])
    def test_diagonal_equals_t_to_2h(self, hurst: float) -> None:
        # The defining property: E[Y_t^2] = t^{2H}.
        t = np.array([0.25, 0.5, 1.0, 2.0, 5.0])
        diag = volterra_autocovariance(t, t, hurst)
        np.testing.assert_allclose(diag, t ** (2.0 * hurst), rtol=1e-10)

    def test_symmetry(self) -> None:
        a = volterra_autocovariance(np.array(0.3), np.array(0.8), 0.15)
        b = volterra_autocovariance(np.array(0.8), np.array(0.3), 0.15)
        np.testing.assert_allclose(a, b, rtol=1e-12)

    def test_zero_time_gives_zero(self) -> None:
        assert float(volterra_autocovariance(np.array(0.0), np.array(1.0), 0.1)) == 0.0

    def test_positive_semidefinite_matrix(self) -> None:
        # The covariance matrix on a grid must be PSD.
        t = np.linspace(0.05, 2.0, 40)
        cov = volterra_autocovariance(t[:, None], t[None, :], 0.1)
        eigenvalues = np.linalg.eigvalsh(cov)
        assert eigenvalues.min() > -1e-10

    def test_invalid_hurst(self) -> None:
        with pytest.raises(ValidationError):
            volterra_autocovariance(np.array(1.0), np.array(1.0), 0.5)

    def test_negative_time_rejected(self) -> None:
        with pytest.raises(ValidationError):
            volterra_autocovariance(np.array(-1.0), np.array(1.0), 0.1)

    @settings(max_examples=50, deadline=None)
    @given(
        s=st.floats(0.01, 5.0),
        t=st.floats(0.01, 5.0),
        hurst=st.floats(0.05, 0.45),
    )
    def test_cauchy_schwarz(self, s: float, t: float, hurst: float) -> None:
        # |Cov(Y_s, Y_t)| <= sqrt(Var(Y_s) Var(Y_t)).
        cov = float(volterra_autocovariance(np.array(s), np.array(t), hurst))
        var_s = s ** (2.0 * hurst)
        var_t = t ** (2.0 * hurst)
        assert abs(cov) <= np.sqrt(var_s * var_t) * (1.0 + 1e-9)


class TestCrossCovariance:
    def test_saturates_for_large_s(self) -> None:
        # For s >= t, E[Y_t Z_s] no longer depends on s (the kernel support ends at t).
        h = 0.1
        a = cross_covariance(np.array(1.0), np.array(1.0), h)
        b = cross_covariance(np.array(1.0), np.array(3.0), h)
        np.testing.assert_allclose(a, b, rtol=1e-12)

    def test_zero_at_zero(self) -> None:
        assert float(cross_covariance(np.array(0.0), np.array(1.0), 0.1)) == 0.0

    def test_monotone_in_s(self) -> None:
        h = 0.2
        s = np.linspace(0.0, 1.0, 50)
        vals = cross_covariance(np.full_like(s, 1.0), s, h)
        assert np.all(np.diff(vals) >= -1e-12)


class TestHybridCoefficients:
    @pytest.mark.parametrize("hurst", [0.05, 0.1, 0.3])
    def test_discrete_covariance_is_pd(self, hurst: float) -> None:
        cov = hybrid_discrete_covariance(0.01, hurst)
        assert cov.shape == (2, 2)
        assert np.allclose(cov, cov.T)
        assert np.all(np.linalg.eigvalsh(cov) > 0.0)

    def test_discrete_covariance_var_w1_is_dt(self) -> None:
        cov = hybrid_discrete_covariance(0.02, 0.1)
        assert cov[0, 0] == pytest.approx(0.02)

    def test_weights_first_two_are_zero(self) -> None:
        w = hybrid_weights(10, 0.01, 0.1)
        assert w[0] == 0.0
        assert w[1] == 0.0
        assert w.size == 11

    def test_weights_positive_and_decreasing(self) -> None:
        w = hybrid_weights(50, 0.01, 0.1)
        tail = w[2:]
        assert np.all(tail > 0.0)
        # For H < 1/2 the kernel decays, so weights are decreasing in lag.
        assert np.all(np.diff(tail) < 0.0)

    def test_invalid_inputs(self) -> None:
        with pytest.raises(ValidationError):
            hybrid_discrete_covariance(0.0, 0.1)
        with pytest.raises(ValidationError):
            hybrid_weights(0, 0.01, 0.1)
