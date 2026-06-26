"""Tests for vol-of-vol (eta) and correlation (rho) estimators.

Both are validated by recovering known parameters from clean latent synthetic paths.
"""

from __future__ import annotations

import numpy as np
import pytest

from options_engine.calibration.vol_of_vol import (
    estimate_eta,
    estimate_rho,
    structure_function,
)
from options_engine.core.errors import CalibrationError, ValidationError
from options_engine.core.random import RandomFactory

from .conftest import make_path


class TestStructureFunction:
    def test_increases_with_lag(self) -> None:
        path = make_path(seed=4)
        lags = np.array([1, 5, 20, 50])
        m2 = structure_function(path.log_variance, lags)
        # Structure function is increasing in lag for a rough process.
        assert np.all(np.diff(m2) > 0.0)

    def test_handles_2d(self) -> None:
        path = make_path(seed=4)
        stacked = np.vstack([path.log_variance, path.log_variance])
        m2 = structure_function(stacked, np.array([1, 10]))
        assert m2.shape == (2,)

    def test_rejects_bad_lags(self) -> None:
        path = make_path(seed=4)
        with pytest.raises(ValidationError):
            structure_function(path.log_variance, np.array([]))


class TestEstimateEta:
    @pytest.mark.slow
    @pytest.mark.parametrize("true_eta", [1.0, 1.6, 2.4])
    def test_recovers_known_eta(self, true_eta: float) -> None:
        path = make_path(eta=true_eta, seed=6)
        est = estimate_eta(
            path.log_variance,
            hurst=path.hurst,
            xi0_level=path.xi0,
            grid=path.grid,
            rng_factory=RandomFactory(77),
            n_sim_paths=120,
        )
        # Simulation-based recovery within ~7% on a clean path.
        assert est.value == pytest.approx(true_eta, rel=0.07)
        assert est.name == "eta"
        assert est.std_error >= 0.0

    def test_rejects_short_series(self) -> None:
        path = make_path(seed=6)
        with pytest.raises(CalibrationError):
            estimate_eta(
                path.log_variance[:10],
                hurst=path.hurst,
                xi0_level=path.xi0,
                grid=path.grid,
                rng_factory=RandomFactory(1),
            )


class TestEstimateRho:
    @pytest.mark.slow
    @pytest.mark.parametrize("true_rho", [-0.3, -0.5, -0.7])
    def test_recovers_known_rho(self, true_rho: float) -> None:
        path = make_path(rho=true_rho, seed=8)
        est = estimate_rho(
            path.log_price,
            path.log_variance,
            hurst=path.hurst,
            eta=path.eta,
            xi0_level=path.xi0,
            grid=path.grid,
            rng_factory=RandomFactory(78),
            n_sim_paths=50,
        )
        # Leverage-curve inversion recovers rho within ~0.1 absolute on a clean path.
        assert est.value == pytest.approx(true_rho, abs=0.1)
        assert est.name == "rho"

    def test_rejects_mismatched_lengths(self) -> None:
        path = make_path(seed=8)
        with pytest.raises(ValidationError):
            estimate_rho(
                path.log_price[:-1],
                path.log_variance,
                hurst=path.hurst,
                eta=path.eta,
                xi0_level=path.xi0,
                grid=path.grid,
                rng_factory=RandomFactory(2),
            )
