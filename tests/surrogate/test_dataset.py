"""Tests for surrogate training-data generation."""

from __future__ import annotations

import numpy as np
import pytest

from options_engine.core.errors import ValidationError
from options_engine.core.random import RandomFactory
from options_engine.surrogate.dataset import (
    ScenarioRanges,
    default_quantile_levels,
    generate_training_data,
)
from options_engine.surrogate.features import N_FEATURES


class TestDefaultQuantileLevels:
    def test_in_open_interval_and_increasing(self) -> None:
        levels = default_quantile_levels(99)
        assert levels.size == 99
        assert levels[0] > 0.0 and levels[-1] < 1.0
        assert np.all(np.diff(levels) > 0.0)

    def test_rejects_too_few(self) -> None:
        with pytest.raises(ValidationError):
            default_quantile_levels(1)


class TestScenarioRanges:
    def test_defaults_valid(self) -> None:
        r = ScenarioRanges()
        assert r.hurst[0] < r.hurst[1]

    def test_rejects_inverted_range(self) -> None:
        with pytest.raises(ValidationError):
            ScenarioRanges(eta=(2.0, 1.0))

    def test_rejects_hurst_out_of_domain(self) -> None:
        with pytest.raises(ValidationError):
            ScenarioRanges(hurst=(0.1, 0.6))


class TestGenerateTrainingData:
    def test_shapes(self) -> None:
        data = generate_training_data(
            n_scenarios=12, rng_factory=RandomFactory(0), n_paths=2000, steps_per_day=2
        )
        assert data.n_samples == 12
        assert data.features.shape == (12, N_FEATURES)
        assert data.quantiles.shape == (12, data.quantile_levels.size)

    def test_quantiles_are_monotone(self) -> None:
        data = generate_training_data(
            n_scenarios=8, rng_factory=RandomFactory(0), n_paths=3000, steps_per_day=2
        )
        # Empirical quantiles are sorted by construction.
        assert np.all(np.diff(data.quantiles, axis=1) >= 0.0)

    def test_reproducible(self) -> None:
        a = generate_training_data(
            n_scenarios=6, rng_factory=RandomFactory(3), n_paths=2000, steps_per_day=2
        )
        b = generate_training_data(
            n_scenarios=6, rng_factory=RandomFactory(3), n_paths=2000, steps_per_day=2
        )
        np.testing.assert_array_equal(a.quantiles, b.quantiles)
        np.testing.assert_array_equal(a.features, b.features)

    def test_rejects_bad_args(self) -> None:
        with pytest.raises(ValidationError):
            generate_training_data(n_scenarios=0, rng_factory=RandomFactory(0))
