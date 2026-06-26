"""Validation tests for the rBergomi simulators.

These are the most important correctness tests in the phase: they check the simulators
against *analytic* properties of the model (not against each other only), namely

* the martingale property of variance (E[v_t] = xi_0(t)),
* the exact second moment of the Volterra driver (E[Y_t^2] = t^{2H}),
* the discounted-spot martingale property (E[S_T] = S_0 e^{rT}),
* the Black-Scholes limit as eta -> 0 (variance becomes deterministic),
* and agreement between the production hybrid scheme and the exact Cholesky scheme.
"""

from __future__ import annotations

import numpy as np
import pytest

from options_engine.core.errors import ValidationError
from options_engine.core.random import RandomFactory
from options_engine.core.timegrid import TimeGrid
from options_engine.models.rbergomi.params import ForwardVariance, RBergomiParams
from options_engine.models.rbergomi.simulator import (
    CholeskySimulator,
    HybridSimulator,
)


def _params(
    hurst: float = 0.1, eta: float = 1.5, rho: float = -0.7, xi: float = 0.04, rate: float = 0.0
) -> RBergomiParams:
    return RBergomiParams(
        hurst=hurst, eta=eta, rho=rho, forward_variance=ForwardVariance.flat(xi), rate=rate
    )


def _factory() -> RandomFactory:
    return RandomFactory(2024)


class TestHybridSimulator:
    def test_output_shapes(self) -> None:
        sim = HybridSimulator(_params(), rng_factory=_factory())
        grid = TimeGrid(horizon_years=1.0, n_steps=50)
        paths = sim.simulate(grid=grid, n_paths=1000, initial_spot=100.0)
        assert paths.spot.shape == (1000, 51)
        assert paths.variance.shape == (1000, 51)
        assert np.all(paths.spot[:, 0] == 100.0)

    @pytest.mark.slow
    def test_variance_martingale_property(self) -> None:
        # E[v_t] = xi_0(t) for all t (the martingale correction guarantees this exactly).
        # The variance process is heavy-tailed (log-normal), so its sample mean converges
        # slowly for large vol-of-vol; we use a moderate eta plus antithetic variates and a
        # tolerance comfortably above the measured Monte-Carlo standard error.
        xi = 0.04
        sim = HybridSimulator(_params(eta=1.0, xi=xi), rng_factory=_factory(), antithetic=True)
        grid = TimeGrid(horizon_years=1.0, n_steps=100)
        paths = sim.simulate(grid=grid, n_paths=80_000, initial_spot=100.0)
        mean_var = paths.variance.mean(axis=0)
        np.testing.assert_allclose(mean_var, xi, rtol=0.03)

    @pytest.mark.slow
    def test_volterra_second_moment(self) -> None:
        # Recover E[Y_t^2] = t^{2H} from the simulated variance by inverting the transform.
        hurst, eta, xi = 0.1, 1.0, 0.04
        sim = HybridSimulator(_params(hurst=hurst, eta=eta, xi=xi), rng_factory=_factory())
        grid = TimeGrid(horizon_years=1.0, n_steps=100)
        paths = sim.simulate(grid=grid, n_paths=80_000, initial_spot=100.0)
        times = paths.times
        # log v = log xi + eta Y - 0.5 eta^2 t^{2H}  =>  Y = (log v - log xi)/eta + 0.5 eta t^{2H}
        log_v = np.log(paths.variance[:, 1:])
        y = (log_v - np.log(xi)) / eta + 0.5 * eta * times[1:] ** (2 * hurst)
        emp_var = y.var(axis=0)
        theo = times[1:] ** (2 * hurst)
        # Check at a few representative maturities.
        for idx in (10, 50, 99):
            assert emp_var[idx] == pytest.approx(theo[idx], rel=0.05)

    @pytest.mark.slow
    def test_discounted_spot_is_martingale(self) -> None:
        # Under the pricing measure E[S_T] = S_0 e^{rT}.
        rate = 0.03
        sim = HybridSimulator(_params(eta=1.0, rate=rate), rng_factory=_factory())
        grid = TimeGrid(horizon_years=0.5, n_steps=80)
        paths = sim.simulate(grid=grid, n_paths=80_000, initial_spot=100.0)
        expected = 100.0 * np.exp(rate * 0.5)
        assert float(paths.terminal_spot().mean()) == pytest.approx(expected, rel=0.01)

    @pytest.mark.slow
    def test_black_scholes_limit_eta_zero(self) -> None:
        # As eta -> 0 the variance is deterministic and equal to xi_0, so terminal
        # log-returns are Normal((r - xi/2)T, xi T).
        xi, rate, horizon = 0.04, 0.0, 1.0
        sim = HybridSimulator(_params(eta=1e-8, rho=0.0, xi=xi, rate=rate), rng_factory=_factory())
        grid = TimeGrid(horizon_years=horizon, n_steps=100)
        paths = sim.simulate(grid=grid, n_paths=80_000, initial_spot=100.0)
        lr = paths.terminal_log_return()
        assert float(lr.mean()) == pytest.approx((rate - 0.5 * xi) * horizon, abs=0.002)
        assert float(lr.std()) == pytest.approx(np.sqrt(xi * horizon), rel=0.02)

    def test_variance_reduction_runs(self) -> None:
        sim = HybridSimulator(
            _params(), rng_factory=_factory(), antithetic=True, quasi_random=True
        )
        grid = TimeGrid(horizon_years=0.25, n_steps=20)
        paths = sim.simulate(grid=grid, n_paths=512, initial_spot=100.0)
        assert np.all(np.isfinite(paths.spot))

    def test_reproducible(self) -> None:
        grid = TimeGrid(horizon_years=0.5, n_steps=30)
        a = HybridSimulator(_params(), rng_factory=RandomFactory(99)).simulate(
            grid=grid, n_paths=500, initial_spot=100.0
        )
        b = HybridSimulator(_params(), rng_factory=RandomFactory(99)).simulate(
            grid=grid, n_paths=500, initial_spot=100.0
        )
        np.testing.assert_array_equal(a.spot, b.spot)

    def test_rejects_bad_inputs(self) -> None:
        sim = HybridSimulator(_params(), rng_factory=_factory())
        grid = TimeGrid(horizon_years=1.0, n_steps=10)
        with pytest.raises(ValidationError):
            sim.simulate(grid=grid, n_paths=0, initial_spot=100.0)
        with pytest.raises(ValidationError):
            sim.simulate(grid=grid, n_paths=10, initial_spot=-1.0)


class TestCholeskySimulator:
    @pytest.mark.slow
    def test_variance_martingale_property(self) -> None:
        xi = 0.04
        sim = CholeskySimulator(_params(eta=1.0, xi=xi), rng_factory=_factory(), antithetic=True)
        grid = TimeGrid(horizon_years=1.0, n_steps=50)
        paths = sim.simulate(grid=grid, n_paths=80_000, initial_spot=100.0)
        np.testing.assert_allclose(paths.variance.mean(axis=0), xi, rtol=0.03)

    def test_rejects_large_grid(self) -> None:
        sim = CholeskySimulator(_params(), rng_factory=_factory())
        grid = TimeGrid(horizon_years=1.0, n_steps=CholeskySimulator.MAX_STEPS + 1)
        with pytest.raises(ValidationError):
            sim.simulate(grid=grid, n_paths=10, initial_spot=100.0)


class TestSchemesAgree:
    @pytest.mark.slow
    def test_terminal_distribution_matches(self) -> None:
        # The hybrid and exact schemes must agree on the terminal distribution to within
        # Monte-Carlo error plus the hybrid scheme's small discretization bias.
        params = _params(hurst=0.12, eta=1.2, rho=-0.6, xi=0.04)
        grid = TimeGrid(horizon_years=0.5, n_steps=60)
        n = 80_000

        hyb = HybridSimulator(params, rng_factory=RandomFactory(1)).simulate(
            grid=grid, n_paths=n, initial_spot=100.0
        )
        cho = CholeskySimulator(params, rng_factory=RandomFactory(2)).simulate(
            grid=grid, n_paths=n, initial_spot=100.0
        )
        lr_h = hyb.terminal_log_return()
        lr_c = cho.terminal_log_return()

        assert float(lr_h.mean()) == pytest.approx(float(lr_c.mean()), abs=0.002)
        assert float(lr_h.std()) == pytest.approx(float(lr_c.std()), rel=0.03)
        # Compare tail quantiles relevant to OTM condor pricing.
        for q in (0.01, 0.05, 0.5, 0.95, 0.99):
            assert np.quantile(lr_h, q) == pytest.approx(np.quantile(lr_c, q), abs=0.01)
