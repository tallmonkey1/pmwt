"""Tests for the DistributionSurrogate train/predict/serialize lifecycle."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from options_engine.core.errors import ModelStateError, ValidationError
from options_engine.core.random import RandomFactory
from options_engine.core.timegrid import TimeGrid
from options_engine.models.rbergomi import (
    ForwardVariance,
    HybridSimulator,
    RBergomiParams,
    build_terminal_distribution,
)
from options_engine.surrogate.dataset import generate_training_data
from options_engine.surrogate.metrics import wasserstein1_from_quantiles
from options_engine.surrogate.surrogate import DistributionSurrogate, TrainingConfig


@pytest.fixture(scope="module")
def small_training_data():
    # Small, fast dataset for lifecycle tests (accuracy is checked separately, slow).
    return generate_training_data(
        n_scenarios=40, rng_factory=RandomFactory(1), n_paths=3000, steps_per_day=2
    )


def _quick_config() -> TrainingConfig:
    return TrainingConfig(hidden_sizes=(16, 16), max_epochs=15, patience=5, seed=0)


class TestLifecycle:
    def test_predict_before_fit_raises(self) -> None:
        with pytest.raises(ModelStateError):
            DistributionSurrogate().predict(hurst=0.1, eta=1.5, rho=-0.7, xi0=0.04, horizon=0.04)

    def test_fit_then_predict(self, small_training_data) -> None:
        s = DistributionSurrogate()
        report = s.fit(small_training_data, config=_quick_config())
        assert s.is_trained
        assert report.epochs_run >= 1
        assert len(report.train_losses) == report.epochs_run

        dist = s.predict(hurst=0.1, eta=1.5, rho=-0.7, xi0=0.04, horizon=10 / 252)
        # Structural guarantee: monotone quantiles.
        assert np.all(np.diff(dist.quantile_values) >= 0.0)
        assert dist.n_quantiles == small_training_data.quantile_levels.size

    def test_predict_batch_shape(self, small_training_data) -> None:
        s = DistributionSurrogate()
        s.fit(small_training_data, config=_quick_config())
        preds = s.predict_quantiles(small_training_data.features[:5])
        assert preds.shape == (5, small_training_data.quantile_levels.size)
        # Each row monotone.
        assert np.all(np.diff(preds, axis=1) >= 0.0)

    def test_reproducible_training(self, small_training_data) -> None:
        a = DistributionSurrogate()
        a.fit(small_training_data, config=_quick_config())
        b = DistributionSurrogate()
        b.fit(small_training_data, config=_quick_config())
        pa = a.predict_quantiles(small_training_data.features[:3])
        pb = b.predict_quantiles(small_training_data.features[:3])
        np.testing.assert_allclose(pa, pb, rtol=1e-6, atol=1e-7)

    def test_save_load_round_trip(self, small_training_data, tmp_path) -> None:
        s = DistributionSurrogate()
        s.fit(small_training_data, config=_quick_config())
        path = str(tmp_path / "surrogate.pt")
        s.save(path)

        loaded = DistributionSurrogate().load(path)
        assert loaded.is_trained
        original = s.predict_quantiles(small_training_data.features[:4])
        restored = loaded.predict_quantiles(small_training_data.features[:4])
        np.testing.assert_allclose(original, restored, rtol=1e-6, atol=1e-7)


class TestTrainingConfig:
    def test_invalid_validation_fraction(self) -> None:
        with pytest.raises(ValidationError):
            TrainingConfig(validation_fraction=1.5)

    def test_invalid_learning_rate(self) -> None:
        with pytest.raises(ValidationError):
            TrainingConfig(learning_rate=0.0)


@pytest.mark.slow
class TestAccuracy:
    def test_surrogate_approximates_monte_carlo(self) -> None:
        data = generate_training_data(
            n_scenarios=300, rng_factory=RandomFactory(2), n_paths=10_000, steps_per_day=4
        )
        s = DistributionSurrogate()
        s.fit(
            data,
            config=TrainingConfig(
                hidden_sizes=(128, 128), max_epochs=250, patience=30, learning_rate=2e-3, seed=0
            ),
        )

        # Held-out scenario inside the training ranges.
        h, eta, rho, xi, horizon_days = 0.11, 1.6, -0.65, 0.045, 12.0
        horizon = horizon_days / 252
        dist = s.predict(hurst=h, eta=eta, rho=rho, xi0=xi, horizon=horizon)

        grid = TimeGrid.from_calendar_days(calendar_days=horizon_days, steps_per_day=4)
        mc = build_terminal_distribution(
            HybridSimulator(
                RBergomiParams(
                    hurst=h, eta=eta, rho=rho, forward_variance=ForwardVariance.flat(xi)
                ),
                rng_factory=RandomFactory(9),
                antithetic=True,
            ).simulate(grid=grid, n_paths=40_000, initial_spot=100.0)
        )

        levels = s.quantile_levels
        w1 = wasserstein1_from_quantiles(dist.quantile(levels), mc.quantile(levels), levels)
        # Surrogate should be within a small Wasserstein distance of MC ground truth.
        assert w1 < 0.02

    def test_deterministic_torch_after_seed(self) -> None:
        # Sanity: the test environment's torch seeding is deterministic.
        torch.manual_seed(0)
        a = torch.randn(5)
        torch.manual_seed(0)
        b = torch.randn(5)
        assert torch.allclose(a, b)
