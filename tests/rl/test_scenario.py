"""Tests for episode scenario generation."""

from __future__ import annotations

import numpy as np
import pytest

from options_engine.core.enums import VolRegime
from options_engine.core.errors import ValidationError
from options_engine.core.random import RandomFactory
from options_engine.rl.scenario import (
    ScenarioConfig,
    generate_episode,
)


class TestScenarioConfig:
    def test_rejects_bad_vrp(self) -> None:
        with pytest.raises(ValidationError):
            ScenarioConfig(vrp_multiplier_range=(0.8, 1.5))  # below 1.0 = no premium

    def test_rejects_inverted_variance_range(self) -> None:
        with pytest.raises(ValidationError):
            ScenarioConfig(realized_variance_range=(0.1, 0.05))


class TestGenerateEpisode:
    def test_structure(self) -> None:
        ep = generate_episode(
            rng_factory=RandomFactory(0),
            config=ScenarioConfig(n_steps=5, n_paths=3000),
            episode_index=1,
        )
        assert len(ep) == 5
        for step in ep.steps:
            assert step.spot > 0.0
            assert step.realized_terminal_sample.ndim == 1
            assert step.chain.strikes.size >= 4
            assert 0.0 <= step.atm_relative_spread <= 5.0
            probs = step.regime.current_probabilities
            assert sum(probs.values()) == pytest.approx(1.0, abs=1e-6)

    def test_vrp_embedded(self) -> None:
        ep = generate_episode(
            rng_factory=RandomFactory(0),
            config=ScenarioConfig(n_steps=3, n_paths=3000),
            episode_index=1,
        )
        assert ep.vrp_multiplier >= 1.0

    def test_reproducible(self) -> None:
        cfg = ScenarioConfig(n_steps=3, n_paths=3000)
        a = generate_episode(rng_factory=RandomFactory(7), config=cfg, episode_index=2)
        b = generate_episode(rng_factory=RandomFactory(7), config=cfg, episode_index=2)
        assert a.realized_variance == b.realized_variance
        np.testing.assert_array_equal(
            a.steps[0].realized_terminal_sample, b.steps[0].realized_terminal_sample
        )

    def test_calm_episode_has_higher_low_prob(self) -> None:
        # A low realized-variance episode should carry a higher low-vol regime probability
        # than a high-variance one (monotone regime surrogate).
        calm_cfg = ScenarioConfig(n_steps=2, n_paths=3000, realized_variance_range=(0.02, 0.021))
        wild_cfg = ScenarioConfig(n_steps=2, n_paths=3000, realized_variance_range=(0.079, 0.08))
        calm = generate_episode(rng_factory=RandomFactory(1), config=calm_cfg, episode_index=1)
        wild = generate_episode(rng_factory=RandomFactory(1), config=wild_cfg, episode_index=1)
        calm_low = calm.steps[0].regime.current_prob(VolRegime.LOW)
        wild_low = wild.steps[0].regime.current_prob(VolRegime.LOW)
        assert calm_low > wild_low
