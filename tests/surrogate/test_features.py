"""Tests for surrogate feature construction and scaling."""

from __future__ import annotations

import numpy as np
import pytest

from options_engine.core.errors import ModelStateError, ValidationError
from options_engine.surrogate.features import (
    N_FEATURES,
    FeatureScaler,
    RawInputs,
    build_feature_matrix,
)


def _raw(n: int = 10) -> RawInputs:
    rng = np.random.default_rng(0)
    return RawInputs(
        hurst=rng.uniform(0.05, 0.45, n),
        eta=rng.uniform(0.5, 3.0, n),
        rho=rng.uniform(-0.95, -0.05, n),
        xi0=rng.uniform(0.01, 0.16, n),
        horizon=rng.uniform(1, 30, n) / 252,
    )


class TestRawInputs:
    def test_valid(self) -> None:
        raw = _raw(5)
        assert raw.n_samples == 5

    def test_length_mismatch_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RawInputs(
                hurst=np.array([0.1, 0.2]),
                eta=np.array([1.0]),
                rho=np.array([-0.5, -0.5]),
                xi0=np.array([0.04, 0.04]),
                horizon=np.array([0.04, 0.04]),
            )

    def test_non_finite_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RawInputs(
                hurst=np.array([np.nan]),
                eta=np.array([1.0]),
                rho=np.array([-0.5]),
                xi0=np.array([0.04]),
                horizon=np.array([0.04]),
            )


class TestBuildFeatureMatrix:
    def test_shape_and_columns(self) -> None:
        raw = _raw(7)
        feats = build_feature_matrix(raw)
        assert feats.shape == (7, N_FEATURES)
        # Column 3 is log(xi0).
        np.testing.assert_allclose(feats[:, 3], np.log(raw.xi0))
        # Column 5 is sqrt(xi0 * horizon).
        np.testing.assert_allclose(feats[:, 5], np.sqrt(raw.xi0 * raw.horizon))

    def test_rejects_nonpositive_xi0(self) -> None:
        raw = RawInputs(
            hurst=np.array([0.1]),
            eta=np.array([1.0]),
            rho=np.array([-0.5]),
            xi0=np.array([-0.01]),
            horizon=np.array([0.04]),
        )
        with pytest.raises(ValidationError):
            build_feature_matrix(raw)


class TestFeatureScaler:
    def test_fit_transform_standardizes(self) -> None:
        feats = build_feature_matrix(_raw(200))
        scaler = FeatureScaler()
        scaled = scaler.fit_transform(feats)
        np.testing.assert_allclose(scaled.mean(axis=0), 0.0, atol=1e-9)
        np.testing.assert_allclose(scaled.std(axis=0), 1.0, atol=1e-6)

    def test_transform_before_fit_raises(self) -> None:
        with pytest.raises(ModelStateError):
            FeatureScaler().transform(np.zeros((2, N_FEATURES)))

    def test_constant_column_safe(self) -> None:
        feats = np.ones((5, 3))
        feats[:, 1] = np.arange(5)
        scaler = FeatureScaler()
        scaled = scaler.fit_transform(feats)
        assert np.all(np.isfinite(scaled))

    def test_width_mismatch_rejected(self) -> None:
        scaler = FeatureScaler().fit(np.random.default_rng(0).normal(size=(10, N_FEATURES)))
        with pytest.raises(ValidationError):
            scaler.transform(np.zeros((2, N_FEATURES + 1)))

    def test_state_dict_round_trip(self) -> None:
        feats = build_feature_matrix(_raw(50))
        scaler = FeatureScaler().fit(feats)
        restored = FeatureScaler.from_state_dict(scaler.state_dict())
        np.testing.assert_allclose(scaler.transform(feats), restored.transform(feats))
