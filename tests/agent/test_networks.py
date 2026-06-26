"""Tests for the actor and distributional-critic networks."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from options_engine.agent.networks import (
    DistributionalCritic,
    GaussianActor,
    quantile_fractions,
)
from options_engine.core.errors import ValidationError


class TestQuantileFractions:
    def test_midpoints(self) -> None:
        taus = quantile_fractions(4)
        np.testing.assert_allclose(taus.numpy(), [0.125, 0.375, 0.625, 0.875])

    def test_in_open_interval(self) -> None:
        taus = quantile_fractions(50)
        assert torch.all(taus > 0.0) and torch.all(taus < 1.0)

    def test_rejects_zero(self) -> None:
        with pytest.raises(ValidationError):
            quantile_fractions(0)


class TestGaussianActor:
    def test_distribution_shape(self) -> None:
        actor = GaussianActor(obs_dim=5, action_dim=3, hidden_sizes=(16,))
        dist = actor.distribution(torch.zeros(7, 5))
        assert dist.mean.shape == (7, 3)
        assert dist.stddev.shape == (7, 3)

    def test_evaluate_actions_shapes(self) -> None:
        actor = GaussianActor(obs_dim=5, action_dim=3, hidden_sizes=(16,))
        obs = torch.zeros(4, 5)
        actions = torch.zeros(4, 3)
        log_prob, entropy = actor.evaluate_actions(obs, actions)
        assert log_prob.shape == (4,)
        assert entropy.shape == (4,)

    def test_log_prob_matches_manual_gaussian(self) -> None:
        # With near-zero mean-head output, log-prob of action 0 should match a unit Gaussian.
        actor = GaussianActor(obs_dim=2, action_dim=1, hidden_sizes=(8,))
        with torch.no_grad():
            obs = torch.zeros(1, 2)
            dist = actor.distribution(obs)
            manual = dist.log_prob(torch.zeros(1, 1)).sum(dim=-1)
            log_prob, _ = actor.evaluate_actions(obs, torch.zeros(1, 1))
        assert torch.allclose(manual, log_prob)

    def test_rejects_wrong_obs_shape(self) -> None:
        actor = GaussianActor(obs_dim=5, action_dim=3, hidden_sizes=(16,))
        with pytest.raises(ValidationError):
            actor.distribution(torch.zeros(4, 6))


class TestDistributionalCritic:
    def test_quantiles_sorted(self) -> None:
        critic = DistributionalCritic(obs_dim=5, n_quantiles=16, hidden_sizes=(16,))
        with torch.no_grad():
            q = critic.quantiles(torch.randn(8, 5))
        assert q.shape == (8, 16)
        # Quantiles are sorted ascending (valid inverse-CDF).
        assert torch.all(q[:, 1:] - q[:, :-1] >= -1e-6)

    def test_value_is_mean_of_quantiles(self) -> None:
        critic = DistributionalCritic(obs_dim=3, n_quantiles=10, hidden_sizes=(8,))
        with torch.no_grad():
            obs = torch.randn(5, 3)
            value = critic.value(obs)
            mean_q = critic.quantiles(obs).mean(dim=-1)
        assert torch.allclose(value, mean_q)

    def test_cvar_is_lower_than_value(self) -> None:
        # CVaR (mean of lowest quantiles) must not exceed the overall mean value.
        critic = DistributionalCritic(
            obs_dim=3, n_quantiles=20, cvar_alpha=0.25, hidden_sizes=(8,)
        )
        with torch.no_grad():
            obs = torch.randn(10, 3)
            value = critic.value(obs)
            cvar = critic.cvar(obs)
        assert torch.all(cvar <= value + 1e-6)

    def test_rejects_bad_cvar_alpha(self) -> None:
        with pytest.raises(ValidationError):
            DistributionalCritic(obs_dim=3, n_quantiles=10, cvar_alpha=1.5)
