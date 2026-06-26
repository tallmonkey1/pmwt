"""Tests for the MarketAlpha framework and its alpha-to-model mappings."""

from __future__ import annotations

import pytest

from options_engine.core.errors import ValidationError
from options_engine.core.market_alpha import (
    DEFAULT_ALPHA_DIM,
    MarketAlpha,
    alpha_components,
    alpha_to_drift_noise,
    alpha_to_eta,
    alpha_to_hurst,
    alpha_to_jump_intensity,
    alpha_to_jump_size,
    alpha_to_shock_intensity,
    alpha_to_stoikov_noise,
)


class TestMarketAlphaConstruction:
    def test_scalar_factory_validates_range(self) -> None:
        with pytest.raises(ValidationError):
            MarketAlpha.scalar(-0.1)
        with pytest.raises(ValidationError):
            MarketAlpha.scalar(1.1)

    def test_scalar_factory_returns_length_one(self) -> None:
        a = MarketAlpha.scalar(0.5)
        assert len(a) == 1
        assert a.is_scalar
        assert a.scalar_value == 0.5

    def test_full_default_dim_validates_each_component(self) -> None:
        MarketAlpha.ones()
        MarketAlpha.zeros()
        MarketAlpha.from_components()

    def test_out_of_range_components_rejected(self) -> None:
        with pytest.raises(ValidationError):
            MarketAlpha(features=(0.5, 1.5, 0.5, 0.5, 0.5))

    def test_too_many_components_rejected(self) -> None:
        with pytest.raises(ValidationError):
            MarketAlpha(features=tuple([0.5] * (DEFAULT_ALPHA_DIM + 1)))

    def test_empty_features_rejected(self) -> None:
        with pytest.raises(ValidationError):
            MarketAlpha(features=())


class TestMarketAlphaMappings:
    """The alpha-to-model-parameter mappings should be monotonic in alpha."""

    def test_hurst_monotonic_in_calmness(self) -> None:
        # alpha=1 (calm) -> larger Hurst; alpha=0 (rough) -> smaller Hurst.
        h_calm = alpha_to_hurst(MarketAlpha.ones())
        h_rough = alpha_to_hurst(MarketAlpha.zeros())
        assert h_calm > h_rough
        assert h_calm == pytest.approx(0.49, abs=1e-9)
        assert h_rough == pytest.approx(0.05, abs=1e-9)

    def test_eta_decreases_with_calmness(self) -> None:
        e_calm = alpha_to_eta(MarketAlpha.ones())
        e_rough = alpha_to_eta(MarketAlpha.zeros())
        assert e_calm < e_rough

    def test_stoikov_noise_zero_at_ones(self) -> None:
        assert alpha_to_stoikov_noise(MarketAlpha.ones()) == 0.0
        assert alpha_to_stoikov_noise(MarketAlpha.zeros()) > 0.0

    def test_drift_noise_zero_at_ones(self) -> None:
        assert alpha_to_drift_noise(MarketAlpha.ones()) == 0.0

    def test_jumps_off_at_full_suppression(self) -> None:
        assert alpha_to_jump_intensity(MarketAlpha.ones()) == 0.0
        assert alpha_to_jump_size(MarketAlpha.ones()) == 0.0

    def test_jumps_grow_as_suppression_falls(self) -> None:
        lam_high = alpha_to_jump_intensity(MarketAlpha.from_components(jump_suppression=0.9))
        lam_low = alpha_to_jump_intensity(MarketAlpha.from_components(jump_suppression=0.1))
        assert lam_low > lam_high

    def test_shock_intensity_zero_at_ones(self) -> None:
        assert alpha_to_shock_intensity(MarketAlpha.ones()) == 0.0


class TestMarketAlphaAccessors:
    def test_padded_right_aligns_short_alphas(self) -> None:
        short = MarketAlpha.scalar(0.3)
        padded = short.padded()
        assert len(padded) == DEFAULT_ALPHA_DIM
        assert padded[0] == 0.3
        for i in range(1, DEFAULT_ALPHA_DIM):
            assert padded[i] == 1.0

    def test_indexing_missing_dims_returns_one(self) -> None:
        short = MarketAlpha.scalar(0.3)
        assert short[0] == 0.3
        assert short[4] == 1.0  # missing = fully calm
        with pytest.raises(IndexError):
            _ = short[10]

    def test_components_returns_named_dict(self) -> None:
        comps = alpha_components(MarketAlpha.ones())
        assert comps["overall_calmness"] == 1.0
        assert comps["stoikov_noise_suppression"] == 1.0

    def test_clipped_clamps_out_of_range(self) -> None:
        # Use a *valid* alpha and explicitly construct a clipped version with custom
        # bounds; the public API validates alpha on construction, so an out-of-range
        # alpha cannot reach ``clipped()`` -- the clipping is for callers who want to
        # project a proposed alpha (e.g. from a gradient step) back into the valid
        # domain before construction.
        a = MarketAlpha(features=(0.3, 0.5, 0.5, 0.5, 0.5))
        clipped = a.clipped(lo=0.4, hi=0.6)
        for f in clipped.features:
            assert 0.4 <= f <= 0.6
