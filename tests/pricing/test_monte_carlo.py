"""Tests for Monte-Carlo pricing off the rBergomi distribution.

The key validation is that, in the deterministic-variance (eta -> 0) limit, Monte-Carlo
prices converge to the analytic Black-Scholes oracle within Monte-Carlo error.
"""

from __future__ import annotations

import numpy as np
import pytest

from options_engine.core.enums import OptionRight
from options_engine.core.errors import ConvergenceError, ValidationError
from options_engine.core.random import RandomFactory
from options_engine.core.timegrid import TimeGrid
from options_engine.models.rbergomi import (
    ForwardVariance,
    HybridSimulator,
    RBergomiParams,
    build_terminal_distribution,
)
from options_engine.models.rbergomi.results import TerminalDistribution
from options_engine.pricing import bs_price
from options_engine.pricing.instruments import EuropeanOption, IronCondor
from options_engine.pricing.monte_carlo import (
    fair_iron_condor_credit,
    price_iron_condor,
    price_option,
)


def _bs_limit_distribution(
    *, xi: float, horizon: float, n_paths: int, seed: int = 1
) -> TerminalDistribution:
    """Build an rBergomi terminal distribution in the (near) Black-Scholes limit."""
    params = RBergomiParams(
        hurst=0.1, eta=1e-8, rho=0.0, forward_variance=ForwardVariance.flat(xi), rate=0.0
    )
    sim = HybridSimulator(params, rng_factory=RandomFactory(seed), antithetic=True)
    grid = TimeGrid(horizon_years=horizon, n_steps=60)
    paths = sim.simulate(grid=grid, n_paths=n_paths, initial_spot=100.0)
    return build_terminal_distribution(paths)


class TestPriceOption:
    @pytest.mark.slow
    @pytest.mark.parametrize("strike", [90.0, 100.0, 110.0])
    @pytest.mark.parametrize("right", [OptionRight.CALL, OptionRight.PUT])
    def test_converges_to_black_scholes(self, strike: float, right: OptionRight) -> None:
        xi, horizon = 0.04, 0.5
        dist = _bs_limit_distribution(xi=xi, horizon=horizon, n_paths=120_000)
        opt = EuropeanOption(strike=strike, expiry=horizon, right=right)
        mc = price_option(opt, dist, rate=0.0)
        analytic = float(bs_price(100.0, strike, horizon, np.sqrt(xi), right)[0])
        # Within 4 standard errors (very high confidence).
        assert abs(mc.estimate - analytic) < 4.0 * mc.standard_error + 1e-9

    def test_expiry_must_match_horizon(self) -> None:
        dist = _bs_limit_distribution(xi=0.04, horizon=0.5, n_paths=2000)
        opt = EuropeanOption(strike=100.0, expiry=0.25, right=OptionRight.CALL)
        with pytest.raises(ValidationError):
            price_option(opt, dist)

    def test_convergence_gate_trips(self) -> None:
        dist = _bs_limit_distribution(xi=0.04, horizon=0.5, n_paths=2000)
        opt = EuropeanOption(strike=130.0, expiry=0.5, right=OptionRight.CALL)
        with pytest.raises(ConvergenceError):
            price_option(opt, dist, max_rel_standard_error=1e-6)

    def test_price_is_non_negative(self) -> None:
        dist = _bs_limit_distribution(xi=0.04, horizon=0.5, n_paths=5000)
        opt = EuropeanOption(strike=100.0, expiry=0.5, right=OptionRight.CALL)
        assert price_option(opt, dist).estimate >= 0.0


class TestPriceIronCondor:
    def _condor(self, horizon: float = 0.5) -> IronCondor:
        return IronCondor(85.0, 92.0, 108.0, 115.0, horizon)

    @pytest.mark.slow
    def test_liability_matches_leg_sum(self) -> None:
        # The condor MC value must equal the signed sum of individually priced legs.
        xi, horizon = 0.04, 0.5
        dist = _bs_limit_distribution(xi=xi, horizon=horizon, n_paths=80_000)
        condor = self._condor(horizon)
        condor_value = price_iron_condor(condor, dist).estimate

        leg_sum = 0.0
        for leg in condor.legs():
            leg_price = price_option(
                EuropeanOption(leg.option.strike, horizon, leg.option.right), dist
            ).estimate
            leg_sum += leg.quantity * leg_price
        assert condor_value == pytest.approx(leg_sum, rel=1e-9, abs=1e-9)

    def test_fair_credit_non_negative_and_consistent(self) -> None:
        dist = _bs_limit_distribution(xi=0.04, horizon=0.5, n_paths=20_000)
        condor = self._condor()
        liability = price_iron_condor(condor, dist)
        fair = fair_iron_condor_credit(condor, dist)
        # Fair credit is the negation of the (non-positive) liability => non-negative.
        assert fair.estimate >= -1e-9
        assert fair.estimate == pytest.approx(-liability.estimate, abs=1e-12)

    @pytest.mark.slow
    def test_fair_credit_below_max_width(self) -> None:
        # The fair credit cannot exceed the max spread width (no-arbitrage on the structure).
        dist = _bs_limit_distribution(xi=0.09, horizon=0.5, n_paths=80_000)
        condor = self._condor()
        fair = fair_iron_condor_credit(condor, dist)
        assert fair.estimate <= condor.max_spread_width + 1e-9

    def test_expiry_mismatch_rejected(self) -> None:
        dist = _bs_limit_distribution(xi=0.04, horizon=0.5, n_paths=2000)
        condor = self._condor(horizon=0.25)
        with pytest.raises(ValidationError):
            price_iron_condor(condor, dist)
