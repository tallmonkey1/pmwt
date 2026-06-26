r"""rBergomi path simulators: the production hybrid scheme and an exact Cholesky scheme.

Two simulators implement the same interface (:class:`RBergomiSimulator`) and produce
identical :class:`~options_engine.models.rbergomi.results.SimulationPaths` objects:

* :class:`HybridSimulator` — the Bennedsen-Lunde-Pakkanen (2017) hybrid scheme with
  :math:`\kappa = 1`. The nearest interval of the Volterra convolution is integrated
  exactly while the remainder uses an optimally-evaluated Riemann sum (computed by FFT
  convolution). This is :math:`O(N \log N)` per path and is the scheme used in production.

* :class:`CholeskySimulator` — an *exact* simulation of the Gaussian vector
  :math:`(Y_{t_i}, Z_{t_i})` from its closed-form covariance, factorized by Cholesky. It
  is :math:`O(N^2)` in memory / :math:`O(N^3)` in time and is intended for **validation**
  of the hybrid scheme on small grids, not for production scale.

Both build the spot path with the same log-Euler discretization

.. math::

    \log S_{i+1} = \log S_i + (r - \tfrac12 v_i)\,\Delta
        + \sqrt{v_i}\,\big(\rho\, \Delta Z_i + \sqrt{1-\rho^2}\, \Delta B_i\big),

where :math:`v_i = \xi_0(t_i)\exp(\eta Y_{t_i} - \tfrac12\eta^2 t_i^{2H})` is martingale
corrected so that :math:`\mathbb{E}[v_t] = \xi_0(t)`.

The variance process is exact under both schemes (it is a deterministic transform of the
Gaussian driver); only the price uses an Euler step, whose bias vanishes as the grid is
refined.
"""

from __future__ import annotations

import abc

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cholesky
from scipy.signal import fftconvolve

from ...core.errors import NumericalError, ValidationError
from ...core.logging import get_logger
from ...core.random import RandomFactory
from ...core.timegrid import TimeGrid
from ...core.validation import check_positive
from .kernel import (
    cross_covariance,
    hybrid_discrete_covariance,
    hybrid_weights,
    volterra_autocovariance,
)
from .noise import draw_standard_normals
from .params import RBergomiParams
from .results import SimulationPaths

__all__ = ["CholeskySimulator", "HybridSimulator", "RBergomiSimulator"]

_logger = get_logger(__name__)


class RBergomiSimulator(abc.ABC):
    """Abstract base for rBergomi simulators (Strategy pattern, SPEC §13: extensibility).

    Subclasses implement :meth:`_simulate_driver`, which returns the variance paths and the
    increments of the driving Brownian motion :math:`Z`. The base class owns the shared,
    validated spot construction so the price dynamics are guaranteed identical across
    schemes.

    Parameters
    ----------
    params:
        Validated rBergomi parameters.
    rng_factory:
        Source of reproducible randomness. Each scheme draws from named sub-streams so
        results are reproducible and independent across components (SPEC §13.10).
    antithetic:
        Enable antithetic variates for the price's independent Brownian component.
    quasi_random:
        Enable Sobol QMC for the noise draws.
    """

    #: Name of the RNG sub-stream for this scheme's primary Gaussian driver.
    _driver_stream: str = "rbergomi.driver"
    #: Name of the RNG sub-stream for the independent price Brownian component.
    _price_stream: str = "rbergomi.price"

    def __init__(
        self,
        params: RBergomiParams,
        *,
        rng_factory: RandomFactory,
        antithetic: bool = False,
        quasi_random: bool = False,
    ) -> None:
        if not isinstance(params, RBergomiParams):
            raise ValidationError(
                "params must be an RBergomiParams instance",
                context={"type": type(params).__name__},
            )
        if not isinstance(rng_factory, RandomFactory):
            raise ValidationError(
                "rng_factory must be a RandomFactory instance",
                context={"type": type(rng_factory).__name__},
            )
        self._params = params
        self._rng_factory = rng_factory
        self._antithetic = bool(antithetic)
        self._quasi_random = bool(quasi_random)

    @property
    def params(self) -> RBergomiParams:
        """The rBergomi parameters used by this simulator."""
        return self._params

    def simulate(self, *, grid: TimeGrid, n_paths: int, initial_spot: float) -> SimulationPaths:
        """Simulate ``n_paths`` spot/variance paths on ``grid`` from ``initial_spot``.

        Parameters
        ----------
        grid:
            The (validated) simulation time grid.
        n_paths:
            Number of Monte-Carlo paths. Must be positive.
        initial_spot:
            Initial spot price :math:`S_0 > 0`.

        Returns
        -------
        SimulationPaths
            Validated spot and variance paths sharing the grid.
        """
        if n_paths < 1:
            raise ValidationError("n_paths must be >= 1", context={"n_paths": n_paths})
        check_positive(initial_spot, name="initial_spot")

        variance, dz = self._simulate_driver(grid=grid, n_paths=n_paths)
        spot = self._build_spot(
            grid=grid, n_paths=n_paths, initial_spot=initial_spot, variance=variance, dz=dz
        )
        paths = SimulationPaths(times=grid.times(), spot=spot, variance=variance)
        _logger.debug(
            "rbergomi_simulated",
            extra={
                "scheme": type(self).__name__,
                "n_paths": n_paths,
                "n_steps": grid.n_steps,
                "horizon": grid.horizon_years,
            },
        )
        return paths

    @abc.abstractmethod
    def _simulate_driver(
        self, *, grid: TimeGrid, n_paths: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return ``(variance, dz)``.

        ``variance`` has shape ``(n_paths, n_points)`` (variance at every grid point).
        ``dz`` has shape ``(n_paths, n_steps)`` and holds the increments of the driving
        Brownian motion :math:`Z` over each step (used for the leverage term in the price).
        """
        raise NotImplementedError

    def _variance_from_volterra(
        self, *, grid: TimeGrid, volterra: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        r"""Map a Volterra-driver array to the martingale-corrected variance process.

        ``volterra`` has shape ``(n_paths, n_points)`` with ``Y_{t_0} = 0``. Returns
        :math:`v_{t_i} = \xi_0(t_i)\exp(\eta Y_{t_i} - \tfrac12\eta^2 t_i^{2H})`.
        """
        params = self._params
        times = grid.times()
        xi0 = params.xi0(times)  # shape (n_points,)
        drift_correction = 0.5 * params.eta**2 * times ** (2.0 * params.hurst)
        variance = xi0 * np.exp(params.eta * volterra - drift_correction)
        if not np.all(np.isfinite(variance)):
            raise NumericalError(
                "non-finite variance produced; check parameters/horizon",
                context={"eta": params.eta, "hurst": params.hurst},
            )
        return np.asarray(variance, dtype=np.float64)

    def _build_spot(
        self,
        *,
        grid: TimeGrid,
        n_paths: int,
        initial_spot: float,
        variance: NDArray[np.float64],
        dz: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Construct spot paths via the shared log-Euler discretization."""
        params = self._params
        dt = grid.dt
        n_steps = grid.n_steps

        # Independent Brownian component B for the price (orthogonal to Z).
        rng = self._rng_factory.generator(self._price_stream)
        z_indep = draw_standard_normals(
            n_paths,
            n_steps,
            rng=rng,
            antithetic=self._antithetic,
            quasi_random=self._quasi_random,
        )
        db = z_indep * np.sqrt(dt)

        rho = params.rho
        sqrt_one_minus_rho2 = np.sqrt(max(0.0, 1.0 - rho * rho))
        # Use the left-point variance for each Euler step (standard, unbiased to O(dt)).
        v_left = variance[:, :n_steps]
        vol = np.sqrt(v_left)

        log_increments = (params.rate - 0.5 * v_left) * dt + vol * (
            rho * dz + sqrt_one_minus_rho2 * db
        )
        log_spot = np.empty((n_paths, n_steps + 1), dtype=np.float64)
        log_spot[:, 0] = np.log(initial_spot)
        np.cumsum(log_increments, axis=1, out=log_spot[:, 1:])
        log_spot[:, 1:] += log_spot[:, [0]]
        spot = np.exp(log_spot)
        # Pin the initial column to the exact input value: exp(log(S0)) can differ from S0
        # by a rounding ulp, and downstream code relies on an exact common initial spot.
        spot[:, 0] = initial_spot
        return spot


class HybridSimulator(RBergomiSimulator):
    """Production rBergomi simulator using the hybrid scheme (:math:`\\kappa = 1`)."""

    def _simulate_driver(
        self, *, grid: TimeGrid, n_paths: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        params = self._params
        dt = grid.dt
        n_steps = grid.n_steps

        # Bivariate Gaussian increments (W^(1), W^(2)) per step with the hybrid covariance.
        cov = hybrid_discrete_covariance(dt, params.hurst)
        try:
            chol = cholesky(cov, lower=True)
        except np.linalg.LinAlgError as exc:  # pragma: no cover - cov is PD by construction
            raise NumericalError(
                "hybrid covariance not positive-definite", context={"dt": dt}
            ) from exc

        rng = self._rng_factory.generator(self._driver_stream)
        # Two independent normals per step, then correlate via the Cholesky factor.
        raw = draw_standard_normals(
            n_paths,
            2 * n_steps,
            rng=rng,
            antithetic=self._antithetic,
            quasi_random=self._quasi_random,
        ).reshape(n_paths, n_steps, 2)
        increments = raw @ chol.T  # shape (n_paths, n_steps, 2)
        dw1 = increments[:, :, 0]  # standard Z increments over each step
        dw2 = increments[:, :, 1]  # kernel-weighted exact increments

        # Exact (nearest-interval) term: Y1[:, i] = dw2[:, i-1] for i >= 1.
        volterra = np.zeros((n_paths, n_steps + 1), dtype=np.float64)
        volterra[:, 1:] = dw2

        # Riemann-sum term via FFT convolution of the weights with the standard increments.
        # The convolution C[m] = sum_k weights[k] * dw1[m - k]; the contribution to the
        # Volterra value Y_{t_i} (for i = 1..n_steps) is C[i], i.e. columns 1..n_steps of
        # the full convolution. (Verified numerically against E[Y_t^2] = t^{2H} and the
        # closed-form off-diagonal covariance.)
        weights = hybrid_weights(n_steps, dt, params.hurst)  # length n_steps + 1
        convolved = fftconvolve(dw1, weights[np.newaxis, :], mode="full", axes=1)
        volterra[:, 1:] += convolved[:, 1 : n_steps + 1]

        # Apply the sqrt(2H) normalisation (2H = 2*alpha + 1).
        volterra *= np.sqrt(2.0 * params.hurst)

        variance = self._variance_from_volterra(grid=grid, volterra=volterra)
        return variance, dw1


class CholeskySimulator(RBergomiSimulator):
    """Exact rBergomi simulator via Cholesky factorization (validation use, small grids).

    Simulates the Gaussian vector :math:`(Y_{t_1}, \\dots, Y_{t_N}, Z_{t_1}, \\dots,
    Z_{t_N})` from its closed-form covariance, giving an exact (non-discretized) sample of
    the variance driver. Because the covariance matrix is :math:`2N \\times 2N`, this scheme
    is intended for ``n_steps`` up to a few hundred.
    """

    #: Maximum number of steps allowed before refusing (guards against accidental misuse).
    MAX_STEPS: int = 1000

    def _simulate_driver(
        self, *, grid: TimeGrid, n_paths: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        params = self._params
        n_steps = grid.n_steps
        if n_steps > self.MAX_STEPS:
            raise ValidationError(
                "CholeskySimulator is for small grids; use HybridSimulator at scale",
                context={"n_steps": n_steps, "max_steps": self.MAX_STEPS},
            )

        step_times = grid.step_times()  # t_1, ..., t_N  (length N)
        joint_cov = self._build_joint_covariance(step_times, params.hurst)

        # Jitter-stabilised Cholesky: the exact covariance is PD but can be ill-conditioned
        # for very rough parameters; add the smallest diagonal jitter that succeeds.
        chol = self._stable_cholesky(joint_cov)

        rng = self._rng_factory.generator(self._driver_stream)
        normals = draw_standard_normals(
            n_paths,
            2 * n_steps,
            rng=rng,
            antithetic=self._antithetic,
            quasi_random=self._quasi_random,
        )
        sample = normals @ chol.T  # shape (n_paths, 2 * n_steps)
        y_grid = sample[:, :n_steps]
        z_grid = sample[:, n_steps:]

        volterra = np.zeros((n_paths, n_steps + 1), dtype=np.float64)
        volterra[:, 1:] = y_grid

        # Increments of Z over each step (Z_{t_0} = 0).
        dz = np.empty((n_paths, n_steps), dtype=np.float64)
        dz[:, 0] = z_grid[:, 0]
        dz[:, 1:] = np.diff(z_grid, axis=1)

        variance = self._variance_from_volterra(grid=grid, volterra=volterra)
        return variance, dz

    @staticmethod
    def _build_joint_covariance(
        step_times: NDArray[np.float64], hurst: float
    ) -> NDArray[np.float64]:
        """Assemble the ``(2N, 2N)`` covariance of ``(Y_{t_i}, Z_{t_i})``."""
        n = step_times.size
        s_mat = step_times[:, np.newaxis]
        t_mat = step_times[np.newaxis, :]

        cov = np.empty((2 * n, 2 * n), dtype=np.float64)
        # Top-left: Cov(Y_s, Y_t).
        cov[:n, :n] = volterra_autocovariance(s_mat, t_mat, hurst)
        # Bottom-right: Cov(Z_s, Z_t) = min(s, t).
        cov[n:, n:] = np.minimum(s_mat, t_mat)
        # Cross blocks: Cov(Y_t, Z_s).  Block[i, j] = Cov(Y_{t_i}, Z_{t_j}).
        cross = cross_covariance(s_mat, t_mat, hurst)  # Cov(Y_{t_i}, Z_{t_j})
        cov[:n, n:] = cross
        cov[n:, :n] = cross.T
        return cov

    @staticmethod
    def _stable_cholesky(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
        """Lower Cholesky factor with adaptive diagonal jitter for numerical stability."""
        scale = float(np.mean(np.diag(matrix)))
        for exponent in range(-14, -5):
            jitter = scale * 10.0**exponent if exponent > -14 else 0.0
            candidate = matrix + jitter * np.eye(matrix.shape[0])
            try:
                return np.asarray(cholesky(candidate, lower=True), dtype=np.float64)
            except np.linalg.LinAlgError:
                continue
        raise NumericalError(
            "joint covariance could not be factorized even with jitter", context={}
        )
