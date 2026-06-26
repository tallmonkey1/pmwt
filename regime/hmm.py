r"""Gaussian Hidden Markov Model with numerically-stable EM (Baum-Welch).

The regime layer (SPEC §2.6) is built on a Markov-switching / hidden-Markov backbone: an
interpretable, well-understood model with a strong prior that volatility moves between a
small number of persistent states. This module implements a Gaussian HMM from first
principles, entirely in **log space**, so that long sequences and small probabilities never
underflow -- a correctness requirement for institutional use (a model that silently returns
``nan`` on a long history is worse than useless).

Model
-----
* ``K`` hidden states; observations ``x_t`` are ``D``-dimensional real vectors.
* Initial distribution ``pi`` (length ``K``).
* Row-stochastic transition matrix ``A`` (``K x K``); ``A[i, j] = P(s_t = j | s_{t-1} = i)``.
* Per-state Gaussian emissions with **diagonal** covariance: mean ``mu_k`` and variances
  ``var_k`` (length ``D``). Diagonal covariance is the standard, robust choice for regime
  features and keeps EM closed-form and stable.

Algorithms
----------
* :meth:`forward_backward` -- log-space alpha/beta recursions returning the log-likelihood,
  the smoothed state posteriors ``gamma``, and the pairwise posteriors ``xi`` needed by EM.
* :meth:`fit` -- Baum-Welch EM with multiple random restarts (best log-likelihood kept),
  variance floors, and convergence on the log-likelihood.
* :meth:`viterbi` -- the maximum-a-posteriori state path, in log space.

All randomness flows through an injected :class:`RandomFactory` for reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp

from ..core.errors import ModelStateError, NumericalError, ValidationError
from ..core.logging import get_logger
from ..core.random import RandomFactory
from ..core.validation import check_positive

__all__ = ["GaussianHMM", "HMMFitReport"]

_logger = get_logger(__name__)

# Numerical floor for variances and probabilities to keep logs finite.
_VAR_FLOOR = 1e-8
_LOG_FLOOR = -700.0  # log of the smallest positive double is ~ -745; stay safely above.
_LOG_2PI = float(np.log(2.0 * np.pi))


@dataclass
class HMMFitReport:
    """Diagnostics returned by :meth:`GaussianHMM.fit`."""

    log_likelihood: float
    n_iterations: int
    converged: bool
    n_restarts: int


class GaussianHMM:
    """A Gaussian hidden Markov model with diagonal-covariance emissions.

    Parameters
    ----------
    n_states:
        Number of hidden states ``K`` (``>= 2``).
    n_features:
        Observation dimension ``D`` (``>= 1``).
    """

    def __init__(self, *, n_states: int, n_features: int) -> None:
        if n_states < 2:
            raise ValidationError("n_states must be >= 2", context={"n_states": n_states})
        if n_features < 1:
            raise ValidationError("n_features must be >= 1", context={"n_features": n_features})
        self.n_states = n_states
        self.n_features = n_features
        self._start_log_prob: NDArray[np.float64] | None = None
        self._log_transition: NDArray[np.float64] | None = None
        self._means: NDArray[np.float64] | None = None
        self._variances: NDArray[np.float64] | None = None

    # -- lifecycle -------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        """True once the model has parameters (fit or set manually)."""
        return self._means is not None

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise ModelStateError("HMM must be fitted before use", context={})

    # -- parameter accessors ---------------------------------------------------------

    @property
    def start_prob(self) -> NDArray[np.float64]:
        """Initial state distribution ``pi`` (length K)."""
        self._require_fitted()
        assert self._start_log_prob is not None
        return np.exp(self._start_log_prob)

    @property
    def transition_matrix(self) -> NDArray[np.float64]:
        """Row-stochastic transition matrix ``A`` (K x K)."""
        self._require_fitted()
        assert self._log_transition is not None
        return np.exp(self._log_transition)

    @property
    def means(self) -> NDArray[np.float64]:
        """Per-state emission means (K x D)."""
        self._require_fitted()
        assert self._means is not None
        return self._means

    @property
    def variances(self) -> NDArray[np.float64]:
        """Per-state emission variances (K x D)."""
        self._require_fitted()
        assert self._variances is not None
        return self._variances

    def set_parameters(
        self,
        *,
        start_prob: NDArray[np.float64],
        transition_matrix: NDArray[np.float64],
        means: NDArray[np.float64],
        variances: NDArray[np.float64],
    ) -> GaussianHMM:
        """Set model parameters explicitly (used for testing and warm starts)."""
        pi = np.asarray(start_prob, dtype=np.float64)
        a = np.asarray(transition_matrix, dtype=np.float64)
        mu = np.asarray(means, dtype=np.float64)
        var = np.asarray(variances, dtype=np.float64)
        self._validate_parameter_shapes(pi, a, mu, var)
        if not np.isclose(pi.sum(), 1.0, atol=1e-6):
            raise ValidationError("start_prob must sum to 1", context={"sum": float(pi.sum())})
        if not np.allclose(a.sum(axis=1), 1.0, atol=1e-6):
            raise ValidationError("transition rows must sum to 1", context={})
        if np.any(var <= 0.0):
            raise ValidationError("variances must be strictly positive", context={})
        self._start_log_prob = np.log(np.clip(pi, 1e-300, None))
        self._log_transition = np.log(np.clip(a, 1e-300, None))
        self._means = mu
        self._variances = np.maximum(var, _VAR_FLOOR)
        return self

    def _validate_parameter_shapes(
        self,
        pi: NDArray[np.float64],
        a: NDArray[np.float64],
        mu: NDArray[np.float64],
        var: NDArray[np.float64],
    ) -> None:
        if pi.shape != (self.n_states,):
            raise ValidationError("start_prob shape mismatch", context={"shape": pi.shape})
        if a.shape != (self.n_states, self.n_states):
            raise ValidationError("transition shape mismatch", context={"shape": a.shape})
        if mu.shape != (self.n_states, self.n_features):
            raise ValidationError("means shape mismatch", context={"shape": mu.shape})
        if var.shape != (self.n_states, self.n_features):
            raise ValidationError("variances shape mismatch", context={"shape": var.shape})

    # -- emissions -------------------------------------------------------------------

    def _log_emission(self, observations: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return the ``(T, K)`` matrix of log emission densities ``log p(x_t | s_t = k)``."""
        assert self._means is not None and self._variances is not None
        x = observations[:, np.newaxis, :]  # (T, 1, D)
        mu = self._means[np.newaxis, :, :]  # (1, K, D)
        var = self._variances[np.newaxis, :, :]  # (1, K, D)
        # Diagonal Gaussian log-density, summed over feature dimensions.
        log_det = np.sum(np.log(var), axis=2)  # (1, K)
        quad = np.sum((x - mu) ** 2 / var, axis=2)  # (T, K)
        result = -0.5 * (self.n_features * _LOG_2PI + log_det + quad)
        return np.asarray(result, dtype=np.float64)

    # -- inference -------------------------------------------------------------------

    def _validate_observations(self, observations: NDArray[np.float64]) -> NDArray[np.float64]:
        obs = np.asarray(observations, dtype=np.float64)
        if obs.ndim != 2 or obs.shape[1] != self.n_features:
            raise ValidationError(
                "observations must have shape (T, n_features)",
                context={"shape": obs.shape, "n_features": self.n_features},
            )
        if obs.shape[0] < 1:
            raise ValidationError("observations must be non-empty", context={})
        if not np.all(np.isfinite(obs)):
            raise ValidationError("observations contain non-finite values", context={})
        return obs

    def forward_backward(
        self, observations: NDArray[np.float64]
    ) -> tuple[float, NDArray[np.float64], NDArray[np.float64]]:
        r"""Run the log-space forward-backward algorithm.

        Returns
        -------
        log_likelihood:
            The total sequence log-likelihood ``log p(x_{1:T})``.
        gamma:
            Smoothed state posteriors ``P(s_t = k | x_{1:T})``, shape ``(T, K)``.
        log_xi_sum:
            Log of the summed pairwise posteriors ``sum_t P(s_t=i, s_{t+1}=j | x)``, shape
            ``(K, K)`` -- the sufficient statistic EM needs for the transition update.
        """
        self._require_fitted()
        assert self._start_log_prob is not None and self._log_transition is not None
        obs = self._validate_observations(observations)
        n = obs.shape[0]
        log_b = self._log_emission(obs)  # (T, K)
        log_a = self._log_transition  # (K, K)

        # Forward pass.
        log_alpha = np.empty((n, self.n_states), dtype=np.float64)
        log_alpha[0] = self._start_log_prob + log_b[0]
        for t in range(1, n):
            # log_alpha[t, j] = logsumexp_i(log_alpha[t-1, i] + log_a[i, j]) + log_b[t, j]
            log_alpha[t] = logsumexp(log_alpha[t - 1][:, np.newaxis] + log_a, axis=0) + log_b[t]
        log_likelihood = float(logsumexp(log_alpha[-1]))
        if not np.isfinite(log_likelihood):
            raise NumericalError("non-finite log-likelihood in forward pass", context={})

        # Backward pass.
        log_beta = np.zeros((n, self.n_states), dtype=np.float64)
        for t in range(n - 2, -1, -1):
            log_beta[t] = logsumexp(
                log_a + (log_b[t + 1] + log_beta[t + 1])[np.newaxis, :], axis=1
            )

        # Posteriors.
        log_gamma = log_alpha + log_beta
        log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
        gamma = np.exp(log_gamma)

        # Pairwise posteriors, accumulated in log space across time.
        if n > 1:
            # log_xi[t, i, j] = alpha[t,i] + a[i,j] + b[t+1,j] + beta[t+1,j] - ll
            log_xi = (
                log_alpha[:-1, :, np.newaxis]
                + log_a[np.newaxis, :, :]
                + (log_b[1:] + log_beta[1:])[:, np.newaxis, :]
                - log_likelihood
            )
            log_xi_sum = logsumexp(log_xi, axis=0)
        else:
            log_xi_sum = np.full((self.n_states, self.n_states), _LOG_FLOOR)

        return log_likelihood, gamma, log_xi_sum

    def predict_proba(self, observations: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return smoothed state posteriors ``gamma`` (T x K)."""
        _, gamma, _ = self.forward_backward(observations)
        return gamma

    def filter_proba(self, observations: NDArray[np.float64]) -> NDArray[np.float64]:
        r"""Return *filtered* (causal) state posteriors ``P(s_t = k | x_{1:t})``.

        Unlike the smoothed :meth:`predict_proba`, the filtered posterior uses only
        information up to time ``t`` -- the correct quantity for live, leakage-free
        nowcasting (SPEC §2.6: "point-in-time, leakage-free").
        """
        self._require_fitted()
        assert self._start_log_prob is not None and self._log_transition is not None
        obs = self._validate_observations(observations)
        n = obs.shape[0]
        log_b = self._log_emission(obs)
        log_a = self._log_transition

        log_alpha = np.empty((n, self.n_states), dtype=np.float64)
        log_alpha[0] = self._start_log_prob + log_b[0]
        for t in range(1, n):
            log_alpha[t] = logsumexp(log_alpha[t - 1][:, np.newaxis] + log_a, axis=0) + log_b[t]
        # Normalize each time step to get the filtered posterior.
        log_filtered = log_alpha - logsumexp(log_alpha, axis=1, keepdims=True)
        return np.asarray(np.exp(log_filtered), dtype=np.float64)

    def viterbi(self, observations: NDArray[np.float64]) -> NDArray[np.int_]:
        """Return the most likely hidden-state path (Viterbi), shape ``(T,)``."""
        self._require_fitted()
        assert self._start_log_prob is not None and self._log_transition is not None
        obs = self._validate_observations(observations)
        n = obs.shape[0]
        log_b = self._log_emission(obs)
        log_a = self._log_transition

        log_delta = np.empty((n, self.n_states), dtype=np.float64)
        backpointer = np.zeros((n, self.n_states), dtype=np.int_)
        log_delta[0] = self._start_log_prob + log_b[0]
        for t in range(1, n):
            scores = log_delta[t - 1][:, np.newaxis] + log_a  # (K, K)
            backpointer[t] = np.argmax(scores, axis=0)
            log_delta[t] = np.max(scores, axis=0) + log_b[t]

        path = np.empty(n, dtype=np.int_)
        path[-1] = int(np.argmax(log_delta[-1]))
        for t in range(n - 2, -1, -1):
            path[t] = backpointer[t + 1, path[t + 1]]
        return path

    def score(self, observations: NDArray[np.float64]) -> float:
        """Return the log-likelihood of a sequence under the fitted model."""
        ll, _, _ = self.forward_backward(observations)
        return ll

    # -- training (Baum-Welch EM) ----------------------------------------------------

    def fit(
        self,
        observations: NDArray[np.float64],
        *,
        rng_factory: RandomFactory,
        n_restarts: int = 5,
        max_iter: int = 200,
        tol: float = 1e-4,
        var_floor: float = _VAR_FLOOR,
    ) -> HMMFitReport:
        """Fit the HMM by Baum-Welch EM with random restarts; keep the best fit.

        Parameters
        ----------
        observations:
            Training sequence, shape ``(T, n_features)``.
        rng_factory:
            Reproducible randomness for the restart initializations.
        n_restarts:
            Number of random initializations; the highest-likelihood fit is retained.
        max_iter, tol:
            EM iteration cap and relative log-likelihood convergence tolerance.
        var_floor:
            Lower bound on emission variances (prevents degenerate collapse onto a point).
        """
        obs = self._validate_observations(observations)
        check_positive(var_floor, name="var_floor")
        if obs.shape[0] < self.n_states:
            raise ValidationError(
                "need at least n_states observations to fit", context={"T": int(obs.shape[0])}
            )
        if n_restarts < 1:
            raise ValidationError("n_restarts must be >= 1", context={})

        best_ll = -np.inf
        best_params: tuple[NDArray[np.float64], ...] | None = None
        best_report = HMMFitReport(
            log_likelihood=best_ll, n_iterations=0, converged=False, n_restarts=n_restarts
        )

        for restart in range(n_restarts):
            rng = rng_factory.generator(f"hmm.restart.{restart}")
            self._initialize_parameters(obs, rng=rng, var_floor=var_floor)
            ll, n_iter, converged = self._run_em(
                obs, max_iter=max_iter, tol=tol, var_floor=var_floor
            )
            if ll > best_ll:
                best_ll = ll
                best_params = (
                    self._start_log_prob.copy(),  # type: ignore[union-attr]
                    self._log_transition.copy(),  # type: ignore[union-attr]
                    self._means.copy(),  # type: ignore[union-attr]
                    self._variances.copy(),  # type: ignore[union-attr]
                )
                best_report = HMMFitReport(
                    log_likelihood=ll,
                    n_iterations=n_iter,
                    converged=converged,
                    n_restarts=n_restarts,
                )

        if best_params is None:  # pragma: no cover - defensive; loop runs >= once
            raise NumericalError("EM failed on all restarts", context={})
        self._start_log_prob, self._log_transition, self._means, self._variances = best_params
        _logger.info(
            "hmm_fitted",
            extra={
                "log_likelihood": best_report.log_likelihood,
                "iterations": best_report.n_iterations,
                "converged": best_report.converged,
                "n_states": self.n_states,
            },
        )
        return best_report

    def _initialize_parameters(
        self, obs: NDArray[np.float64], *, rng: np.random.Generator, var_floor: float
    ) -> None:
        """Initialize EM by assigning observations to states via quantile binning + noise.

        Quantile-based seeding (rather than pure random means) gives EM a sensible, ordered
        starting point and dramatically improves restart reliability for regime data, where
        states differ primarily in the *level* of the (log-)variance feature.
        """
        n = obs.shape[0]
        # Order states along the first feature's quantiles (the dominant vol-level axis).
        order_feature = obs[:, 0]
        quantile_edges = np.quantile(order_feature, np.linspace(0, 1, self.n_states + 1))
        means = np.empty((self.n_states, self.n_features), dtype=np.float64)
        variances = np.empty((self.n_states, self.n_features), dtype=np.float64)
        global_var = np.maximum(obs.var(axis=0), var_floor)
        for k in range(self.n_states):
            lo, hi = quantile_edges[k], quantile_edges[k + 1]
            mask = (order_feature >= lo) & (order_feature <= hi)
            if not np.any(mask):
                mask = np.ones(n, dtype=bool)
            block = obs[mask]
            means[k] = block.mean(axis=0) + rng.normal(0, 1e-3, size=self.n_features)
            variances[k] = np.maximum(block.var(axis=0), var_floor)
        variances = np.maximum(variances, global_var * 0.1)

        # Persistent transitions (regimes are sticky): strong diagonal prior.
        a = np.full((self.n_states, self.n_states), 0.1 / (self.n_states - 1))
        np.fill_diagonal(a, 0.9)
        pi = np.full(self.n_states, 1.0 / self.n_states)
        self.set_parameters(start_prob=pi, transition_matrix=a, means=means, variances=variances)

    def _run_em(
        self, obs: NDArray[np.float64], *, max_iter: int, tol: float, var_floor: float
    ) -> tuple[float, int, bool]:
        """Run EM to convergence from the current parameters; returns (ll, n_iter, converged)."""
        prev_ll = -np.inf
        converged = False
        n_iter = 0
        for iteration in range(1, max_iter + 1):
            n_iter = iteration
            log_likelihood, gamma, log_xi_sum = self.forward_backward(obs)
            self._m_step(obs, gamma, log_xi_sum, var_floor=var_floor)
            if np.isfinite(prev_ll):
                rel_improve = (log_likelihood - prev_ll) / (abs(prev_ll) + 1e-12)
                if abs(rel_improve) < tol:
                    converged = True
                    prev_ll = log_likelihood
                    break
            prev_ll = log_likelihood
        return prev_ll, n_iter, converged

    def _m_step(
        self,
        obs: NDArray[np.float64],
        gamma: NDArray[np.float64],
        log_xi_sum: NDArray[np.float64],
        *,
        var_floor: float,
    ) -> None:
        """Maximization step: closed-form parameter updates from the posteriors."""
        # Initial distribution.
        start = np.clip(gamma[0], 1e-300, None)
        start /= start.sum()

        # Transition matrix from summed pairwise posteriors.
        xi_sum = np.exp(log_xi_sum - log_xi_sum.max())  # stabilized; row-normalized next
        row_sums = xi_sum.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums <= 0.0, 1.0, row_sums)
        a = xi_sum / row_sums

        # Emission means and variances, weighted by gamma.
        weights = gamma.sum(axis=0)  # (K,)
        weights_safe = np.where(weights <= 0.0, 1.0, weights)
        means = (gamma.T @ obs) / weights_safe[:, np.newaxis]
        variances = np.empty_like(means)
        for k in range(self.n_states):
            diff = obs - means[k]
            variances[k] = (gamma[:, k] @ (diff**2)) / weights_safe[k]
        variances = np.maximum(variances, var_floor)

        self.set_parameters(
            start_prob=start, transition_matrix=a, means=means, variances=variances
        )
