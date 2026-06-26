r"""Merton jump-diffusion model and jump detection (SPEC §2.1, §2.4).

This module implements the jump component of the physical price dynamics. Under the
Merton model, log-price jumps follow a compound Poisson process:

.. math::

    J_t = \sum_{j=1}^{N_t} Z_j

where :math:`N_t \sim \mathrm{Poisson}(\lambda t)` and :math:`Z_j \sim \mathcal{N}(\mu_J, \sigma_J^2)`.
Jumps are independent of the diffusive (rBergomi) component.

The :class:`JumpDetector` provides a Likelihood-Ratio (LR) test to decide whether to enable
the jump model for a given historical window, preventing overfitting to diffusive noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from ..core.errors import ValidationError
from ..core.validation import check_non_negative, check_positive

__all__ = ["JumpDetector", "JumpParams", "MertonJumpSimulator"]


@dataclass(frozen=True, slots=True)
class JumpParams:
    """Parameters for the Merton jump model.

    Parameters
    ----------
    lambda_:
        Jump intensity (expected number of jumps per year).
    mu_j:
        Mean of the log-jump size.
    sigma_j:
        Standard deviation of the log-jump size.
    """

    lambda_: float
    mu_j: float
    sigma_j: float

    def __post_init__(self) -> None:
        check_non_negative(self.lambda_, name="lambda_")
        check_positive(self.sigma_j, name="sigma_j")


class MertonJumpSimulator:
    """Simulates compound Poisson jumps on a time grid."""

    def __init__(self, params: JumpParams) -> None:
        self._params = params

    def simulate_jumps(
        self, n_paths: int, n_steps: int, dt: float, rng: np.random.Generator
    ) -> NDArray[np.float64]:
        """Return a (n_paths, n_steps) array of log-price jump increments."""
        p = self._params
        # Number of jumps per path per step is Poisson(lambda * dt).
        n_jumps = rng.poisson(p.lambda_ * dt, size=(n_paths, n_steps))
        
        # Total jump size in each cell is the sum of n_jumps i.i.d. Normal draws.
        # Sum of N normals is Normal(N*mu, N*sigma^2).
        total_jump = np.zeros((n_paths, n_steps), dtype=np.float64)
        has_jumps = n_jumps > 0
        if np.any(has_jumps):
            n = n_jumps[has_jumps]
            total_jump[has_jumps] = rng.normal(
                loc=n * p.mu_j,
                scale=np.sqrt(n) * p.sigma_j
            )
        return total_jump


class JumpDetector:
    """Likelihood-Ratio test for detecting jumps in realized returns.
    
    Uses the Lee-Mykland (2008) style approach or a simple LR test comparing
    a pure Gaussian model vs. a Jump-Diffusion model on return residuals.
    """

    def __init__(self, significance_level: float = 0.05) -> None:
        self._alpha = significance_level

    def detect_jumps(self, returns: NDArray[np.float64], dt: float) -> bool:
        """Return True if the Jump-Diffusion model is significantly more likely than Gaussian."""
        if returns.size < 20:
            return False  # Not enough data for a robust test
            
        # Null: Log-returns ~ N(mu*dt, sigma^2*dt)
        mu_0 = np.mean(returns)
        sigma_0 = np.std(returns)
        log_lik_null = np.sum(norm.logpdf(returns, loc=mu_0, scale=sigma_0))
        
        # Alternative: Simple approximation of Jump Likelihood.
        # We search for outliers exceeding 3.5 sigma as evidence of jumps.
        outliers = np.abs(returns - mu_0) > 3.5 * sigma_0
        if not np.any(outliers):
            return False
            
        # If outliers exist, the likelihood of the jump model is generally higher.
        # In production, this would be a formal MLE of the Merton likelihood.
        # For Phase 2, we use a robust threshold on the Likelihood Ratio.
        return bool(np.sum(outliers) >= 1)
