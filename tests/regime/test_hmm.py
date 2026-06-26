"""Tests for the Gaussian HMM, validated against synthetic data with known states."""

from __future__ import annotations

import numpy as np
import pytest

from options_engine.core.errors import ModelStateError, ValidationError
from options_engine.core.random import RandomFactory
from options_engine.regime.hmm import GaussianHMM

from .conftest import make_regimes


class TestConstruction:
    def test_rejects_too_few_states(self) -> None:
        with pytest.raises(ValidationError):
            GaussianHMM(n_states=1, n_features=1)

    def test_rejects_zero_features(self) -> None:
        with pytest.raises(ValidationError):
            GaussianHMM(n_states=2, n_features=0)

    def test_use_before_fit_raises(self) -> None:
        hmm = GaussianHMM(n_states=2, n_features=1)
        with pytest.raises(ModelStateError):
            hmm.predict_proba(np.zeros((5, 1)))


class TestParameterValidation:
    def test_set_parameters_round_trip(self) -> None:
        hmm = GaussianHMM(n_states=2, n_features=1)
        pi = np.array([0.5, 0.5])
        a = np.array([[0.9, 0.1], [0.2, 0.8]])
        mu = np.array([[0.0], [1.0]])
        var = np.array([[1.0], [1.0]])
        hmm.set_parameters(start_prob=pi, transition_matrix=a, means=mu, variances=var)
        np.testing.assert_allclose(hmm.transition_matrix, a, atol=1e-9)
        np.testing.assert_allclose(hmm.means, mu)

    def test_rejects_non_stochastic_transition(self) -> None:
        hmm = GaussianHMM(n_states=2, n_features=1)
        with pytest.raises(ValidationError):
            hmm.set_parameters(
                start_prob=np.array([0.5, 0.5]),
                transition_matrix=np.array([[0.9, 0.2], [0.2, 0.8]]),
                means=np.array([[0.0], [1.0]]),
                variances=np.array([[1.0], [1.0]]),
            )

    def test_rejects_nonpositive_variance(self) -> None:
        hmm = GaussianHMM(n_states=2, n_features=1)
        with pytest.raises(ValidationError):
            hmm.set_parameters(
                start_prob=np.array([0.5, 0.5]),
                transition_matrix=np.array([[0.9, 0.1], [0.2, 0.8]]),
                means=np.array([[0.0], [1.0]]),
                variances=np.array([[1.0], [0.0]]),
            )


@pytest.fixture(scope="module")
def fitted_hmm() -> GaussianHMM:
    """A single HMM fit once and shared across consistency tests (keeps the suite fast)."""
    data = make_regimes(n=1500, seed=1)
    hmm = GaussianHMM(n_states=3, n_features=1)
    hmm.fit(data.observations, rng_factory=RandomFactory(2), n_restarts=2, max_iter=60)
    return hmm


class TestInferenceConsistency:
    def test_posteriors_sum_to_one(self, fitted_hmm: GaussianHMM) -> None:
        data = make_regimes(n=500, seed=3)
        gamma = fitted_hmm.predict_proba(data.observations)
        np.testing.assert_allclose(gamma.sum(axis=1), 1.0, atol=1e-9)

    def test_filtered_posteriors_sum_to_one(self, fitted_hmm: GaussianHMM) -> None:
        data = make_regimes(n=500, seed=4)
        filtered = fitted_hmm.filter_proba(data.observations)
        np.testing.assert_allclose(filtered.sum(axis=1), 1.0, atol=1e-9)

    def test_loglik_finite_for_long_sequence(self, fitted_hmm: GaussianHMM) -> None:
        # The whole point of log-space: long sequences must not underflow.
        data = make_regimes(n=5000, seed=5)
        ll = fitted_hmm.score(data.observations)
        assert np.isfinite(ll)

    def test_viterbi_path_shape_and_range(self, fitted_hmm: GaussianHMM) -> None:
        data = make_regimes(n=400, seed=6)
        path = fitted_hmm.viterbi(data.observations)
        assert path.shape == (400,)
        assert path.min() >= 0 and path.max() < 3

    def test_rejects_wrong_feature_width(self, fitted_hmm: GaussianHMM) -> None:
        with pytest.raises(ValidationError):
            fitted_hmm.predict_proba(np.zeros((5, 2)))


@pytest.mark.slow
class TestParameterRecovery:
    def test_recovers_means(self) -> None:
        data = make_regimes(n=4000, means=(-6.0, -4.5, -3.0), seed=10)
        hmm = GaussianHMM(n_states=3, n_features=1)
        report = hmm.fit(
            data.observations, rng_factory=RandomFactory(7), n_restarts=5, max_iter=150
        )
        assert report.converged
        recovered = np.sort(hmm.means[:, 0])
        np.testing.assert_allclose(recovered, [-6.0, -4.5, -3.0], atol=0.15)

    def test_viterbi_accuracy_high(self) -> None:
        data = make_regimes(n=4000, seed=11)
        hmm = GaussianHMM(n_states=3, n_features=1)
        hmm.fit(data.observations, rng_factory=RandomFactory(8), n_restarts=5, max_iter=150)
        path = hmm.viterbi(data.observations)
        # Map learned states to truth by mean ordering.
        order = np.argsort(hmm.means[:, 0])
        remap = {int(order[i]): i for i in range(3)}
        mapped = np.array([remap[int(p)] for p in path])
        accuracy = float(np.mean(mapped == data.states))
        assert accuracy > 0.9

    def test_recovers_persistence(self) -> None:
        data = make_regimes(n=4000, persistence=0.95, seed=12)
        hmm = GaussianHMM(n_states=3, n_features=1)
        hmm.fit(data.observations, rng_factory=RandomFactory(9), n_restarts=5, max_iter=150)
        # Diagonal of the learned transition should be high (sticky regimes).
        assert np.all(np.diag(hmm.transition_matrix) > 0.85)


class TestFitValidation:
    def test_rejects_too_short_sequence(self) -> None:
        hmm = GaussianHMM(n_states=3, n_features=1)
        with pytest.raises(ValidationError):
            hmm.fit(np.zeros((2, 1)), rng_factory=RandomFactory(0))

    def test_reproducible(self) -> None:
        data = make_regimes(n=800, seed=20)
        a = GaussianHMM(n_states=3, n_features=1)
        a.fit(data.observations, rng_factory=RandomFactory(1), n_restarts=2, max_iter=50)
        b = GaussianHMM(n_states=3, n_features=1)
        b.fit(data.observations, rng_factory=RandomFactory(1), n_restarts=2, max_iter=50)
        np.testing.assert_allclose(a.means, b.means)
        np.testing.assert_allclose(a.transition_matrix, b.transition_matrix)
