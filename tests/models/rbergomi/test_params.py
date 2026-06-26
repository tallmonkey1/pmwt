"""Tests for rBergomi parameters and the forward-variance curve."""

from __future__ import annotations

import numpy as np
import pytest

from options_engine.core.errors import ValidationError
from options_engine.models.rbergomi.params import ForwardVariance, RBergomiParams


class TestForwardVariance:
    def test_flat_curve(self) -> None:
        fv = ForwardVariance.flat(0.04)
        assert fv.is_flat
        assert fv.at(0.0) == pytest.approx(0.04)
        assert fv.at(10.0) == pytest.approx(0.04)
        np.testing.assert_allclose(fv(np.array([0.0, 1.0, 5.0])), 0.04)

    def test_flat_rejects_nonpositive(self) -> None:
        with pytest.raises(ValidationError):
            ForwardVariance.flat(0.0)
        with pytest.raises(ValidationError):
            ForwardVariance.flat(-1.0)

    def test_piecewise_linear_interpolation(self) -> None:
        fv = ForwardVariance(
            knot_times=np.array([0.0, 1.0, 2.0]),
            knot_values=np.array([0.04, 0.06, 0.05]),
        )
        assert not fv.is_flat
        assert fv.at(0.5) == pytest.approx(0.05)  # midpoint of 0.04 and 0.06
        assert fv.at(1.5) == pytest.approx(0.055)

    def test_flat_extrapolation(self) -> None:
        fv = ForwardVariance(knot_times=np.array([0.0, 1.0]), knot_values=np.array([0.04, 0.06]))
        assert fv.at(5.0) == pytest.approx(0.06)  # beyond last knot

    def test_must_start_at_zero(self) -> None:
        with pytest.raises(ValidationError):
            ForwardVariance(knot_times=np.array([0.5, 1.0]), knot_values=np.array([0.04, 0.06]))

    def test_strictly_increasing_times(self) -> None:
        with pytest.raises(ValidationError):
            ForwardVariance(
                knot_times=np.array([0.0, 1.0, 1.0]),
                knot_values=np.array([0.04, 0.05, 0.06]),
            )

    def test_positive_values(self) -> None:
        with pytest.raises(ValidationError):
            ForwardVariance(knot_times=np.array([0.0, 1.0]), knot_values=np.array([0.04, 0.0]))

    def test_length_mismatch(self) -> None:
        with pytest.raises(ValidationError):
            ForwardVariance(knot_times=np.array([0.0, 1.0]), knot_values=np.array([0.04]))

    def test_negative_time_rejected(self) -> None:
        fv = ForwardVariance.flat(0.04)
        with pytest.raises(ValidationError):
            fv(np.array([-0.1]))

    def test_immutable_knots(self) -> None:
        fv = ForwardVariance.flat(0.04)
        with pytest.raises(ValueError):
            fv.knot_values[0] = 99.0


class TestRBergomiParams:
    def _make(self, **kwargs: object) -> RBergomiParams:
        defaults: dict[str, object] = {
            "hurst": 0.1,
            "eta": 1.5,
            "rho": -0.7,
            "forward_variance": ForwardVariance.flat(0.04),
        }
        defaults.update(kwargs)
        return RBergomiParams(**defaults)  # type: ignore[arg-type]

    def test_valid_construction(self) -> None:
        p = self._make()
        assert p.hurst == 0.1
        assert p.alpha == pytest.approx(-0.4)
        assert p.gamma == pytest.approx(0.4)
        assert p.rate == 0.0

    def test_hurst_open_interval(self) -> None:
        with pytest.raises(ValidationError):
            self._make(hurst=0.0)
        with pytest.raises(ValidationError):
            self._make(hurst=0.5)

    def test_eta_positive(self) -> None:
        with pytest.raises(ValidationError):
            self._make(eta=0.0)

    def test_rho_bounds(self) -> None:
        with pytest.raises(ValidationError):
            self._make(rho=-1.5)
        assert self._make(rho=-1.0).rho == -1.0

    def test_forward_variance_type_checked(self) -> None:
        with pytest.raises(ValidationError):
            self._make(forward_variance=0.04)

    def test_xi0_delegates(self) -> None:
        p = self._make()
        np.testing.assert_allclose(p.xi0(np.array([0.0, 1.0])), 0.04)

    def test_with_eta(self) -> None:
        p = self._make()
        p2 = p.with_eta(2.0)
        assert p2.eta == 2.0
        assert p.eta == 1.5  # original unchanged
        assert p2.hurst == p.hurst
