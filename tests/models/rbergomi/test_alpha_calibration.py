"""Tests for the alpha-to-rBergomi calibration helper."""

from __future__ import annotations

import pytest

from options_engine.core.market_alpha import (
    MarketAlpha,
    alpha_to_hurst,
    alpha_to_eta,
)
from options_engine.models.rbergomi.alpha_calibration import (
    alpha_diagnostics,
    build_rbergomi_params_from_alpha,
)


class TestBuildRbergomiParams:
    def test_ones_alpha_gives_smooth_params(self) -> None:
        params = build_rbergomi_params_from_alpha(MarketAlpha.ones())
        assert alpha_to_hurst(MarketAlpha.ones()) == pytest.approx(params.hurst)
        assert alpha_to_eta(MarketAlpha.ones()) == pytest.approx(params.eta)
        assert 0.4 < params.hurst < 0.5  # alpha=1 -> H close to (but strictly inside) 0.5
        assert params.eta < 0.5

    def test_zeros_alpha_gives_rough_params(self) -> None:
        params = build_rbergomi_params_from_alpha(MarketAlpha.zeros())
        assert params.hurst < 0.1
        assert params.eta > 1.5

    def test_rejects_non_alpha(self) -> None:
        with pytest.raises(TypeError):
            build_rbergomi_params_from_alpha("not an alpha")


class TestAlphaDiagnostics:
    def test_returns_all_mappings(self) -> None:
        d = alpha_diagnostics(MarketAlpha.ones())
        for key in (
            "model.hurst",
            "model.eta",
            "model.stoikov_noise",
            "model.drift_noise",
            "model.jump_intensity",
            "model.jump_size",
            "model.shock_intensity",
        ):
            assert key in d
        # At alpha=ones, the noise / jump / shock knobs are all 0.
        assert d["model.stoikov_noise"] == 0.0
        assert d["model.jump_intensity"] == 0.0
        assert d["model.shock_intensity"] == 0.0
