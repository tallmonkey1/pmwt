"""Tests for leakage-free regime feature construction."""

from __future__ import annotations

import numpy as np
import pytest

from options_engine.core.errors import ValidationError
from options_engine.regime.features import N_REGIME_FEATURES, build_regime_features


class TestBuildRegimeFeatures:
    def test_shape(self) -> None:
        rng = np.random.default_rng(0)
        rets = rng.normal(0, 0.01, size=500)
        feats = build_regime_features(rets, window=21)
        assert feats.shape == (500 - 21, N_REGIME_FEATURES)

    def test_log_rv_increases_with_volatility(self) -> None:
        rng = np.random.default_rng(1)
        calm = rng.normal(0, 0.005, size=200)
        wild = rng.normal(0, 0.03, size=200)
        rets = np.concatenate([calm, wild])
        feats = build_regime_features(rets, window=21)
        # log_rv (col 0) should be substantially higher in the wild second half.
        first_half = feats[:150, 0].mean()
        second_half = feats[-150:, 0].mean()
        assert second_half > first_half + 1.0

    def test_no_lookahead_prefix_invariance(self) -> None:
        # Feature row i depends only on returns up to index (i + window - 1). Appending
        # future data must not change earlier feature rows -- the leakage-free guarantee.
        rng = np.random.default_rng(2)
        rets = rng.normal(0, 0.01, size=300)
        extra = rng.normal(0, 0.05, size=100)
        feats_short = build_regime_features(rets, window=21)
        feats_long = build_regime_features(np.concatenate([rets, extra]), window=21)
        np.testing.assert_allclose(feats_short, feats_long[: feats_short.shape[0]])

    def test_downside_ratio_in_unit_interval(self) -> None:
        rng = np.random.default_rng(3)
        feats = build_regime_features(rng.normal(0, 0.01, size=400), window=21)
        assert np.all(feats[:, 3] >= 0.0) and np.all(feats[:, 3] <= 1.0)

    def test_rejects_window_too_large(self) -> None:
        with pytest.raises(ValidationError):
            build_regime_features(np.zeros(10), window=21)

    def test_rejects_small_window(self) -> None:
        with pytest.raises(ValidationError):
            build_regime_features(np.zeros(100), window=1)

    def test_rejects_non_finite(self) -> None:
        rets = np.zeros(100)
        rets[5] = np.nan
        with pytest.raises(ValidationError):
            build_regime_features(rets, window=21)
