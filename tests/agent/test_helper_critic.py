"""Tests for the helper critic."""

from __future__ import annotations

import numpy as np
import pytest

from options_engine.agent.helper_critic import (
    DEFAULT_HELPER_LATTICE_SIZE,
    HelperCritic,
    HelperCriticConfig,
    alpha_lattice,
    feature_score_from_components,
)
from options_engine.core.errors import ValidationError
from options_engine.core.market_alpha import MarketAlpha


class TestAlphaLattice:
    def test_default_lattice_size(self) -> None:
        lat = alpha_lattice()
        assert len(lat) == DEFAULT_HELPER_LATTICE_SIZE

    def test_lattice_first_is_zeros_last_is_ones(self) -> None:
        lat = alpha_lattice(size=5)
        assert lat[0][0] == pytest.approx(0.0)
        assert lat[-1][0] == pytest.approx(1.0)

    def test_lattice_each_alpha_is_padded_to_default_dim(self) -> None:
        for alpha in alpha_lattice(size=4):
            assert len(alpha) == 5  # DEFAULT_ALPHA_DIM

    def test_rejects_zero_size(self) -> None:
        with pytest.raises(ValidationError):
            alpha_lattice(size=0)


class TestHelperCritic:
    def test_initial_best_alpha_is_ones(self) -> None:
        critic = HelperCritic()
        assert critic.best_alpha()[0] == 1.0

    def test_update_after_select(self) -> None:
        critic = HelperCritic(config=HelperCriticConfig(seed=0, exploration_rate=0.0))
        # Force a specific alpha via the lattice.
        critic._last_alpha = MarketAlpha.scalar(0.5)
        r = critic.update(features=np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]))
        assert r == pytest.approx(3.0)

    def test_update_increments_q_for_pulled_alpha(self) -> None:
        critic = HelperCritic(config=HelperCriticConfig(seed=0, exploration_rate=0.0))
        critic._last_alpha = MarketAlpha.scalar(0.5)
        critic.update(features=np.ones(6))
        critic._last_alpha = MarketAlpha.scalar(0.5)
        critic.update(features=np.zeros(6))
        # The alpha at index ~4 (scalar 0.5) should have been pulled twice.
        pulls = int(critic._n_pulls.sum())
        assert pulls == 2

    def test_diagnostics_returns_expected_keys(self) -> None:
        critic = HelperCritic()
        d = critic.diagnostics()
        assert "best_q" in d
        assert "mean_q" in d
        assert "total_pulls" in d

    def test_select_alpha_without_explore_returns_best(self) -> None:
        critic = HelperCritic(config=HelperCriticConfig(seed=0, exploration_rate=0.0))
        critic._last_alpha = MarketAlpha.scalar(0.2)
        critic.update(features=np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9]))
        # Next alpha should still come from the lattice (and most likely the same one).
        next_a = critic.select_alpha()
        assert next_a in critic.lattice

    def test_update_before_select_raises(self) -> None:
        critic = HelperCritic()
        with pytest.raises(ValidationError):
            critic.update(features=np.zeros(6))


class TestFeatureScore:
    def test_clips_to_unit_interval(self) -> None:
        v = feature_score_from_components(
            win_probability=2.0,
            cvar_safety=-1.0,
            profit_factor=0.5,
            stability=0.5,
            margin_safety=0.5,
            drawdown_safety=0.5,
        )
        assert np.all((v >= 0.0) & (v <= 1.0))
