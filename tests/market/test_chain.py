"""Tests for synthetic chain construction and no-arbitrage repair."""

from __future__ import annotations

import numpy as np
import pytest

from options_engine.core.errors import ValidationError
from options_engine.core.random import RandomFactory
from options_engine.core.timegrid import TimeGrid
from options_engine.market.chain import build_synthetic_chain, repair_call_curve
from options_engine.market.market_maker import AvellanedaStoikovMaker
from options_engine.models.rbergomi import (
    ForwardVariance,
    HybridSimulator,
    RBergomiParams,
    build_terminal_distribution,
)


class TestRepairCallCurve:
    def test_enforces_monotonicity(self) -> None:
        strikes = np.array([90.0, 95.0, 100.0, 105.0, 110.0])
        # Noisy, non-monotone call values.
        noisy = np.array([12.0, 9.5, 9.8, 4.0, 2.5])
        repaired = repair_call_curve(strikes, noisy, spot=100.0, discount=1.0)
        assert np.all(np.diff(repaired) <= 1e-9)

    def test_enforces_convexity(self) -> None:
        strikes = np.array([90.0, 95.0, 100.0, 105.0, 110.0])
        noisy = np.array([11.0, 5.0, 6.5, 3.0, 1.0])  # dip at 95 breaks convexity
        repaired = repair_call_curve(strikes, noisy, spot=100.0, discount=1.0)
        second_diff = np.diff(repaired, 2)
        assert np.all(second_diff >= -1e-9)

    def test_respects_bounds(self) -> None:
        strikes = np.array([90.0, 100.0, 110.0])
        # Values above spot and below intrinsic.
        bad = np.array([200.0, -5.0, 0.0])
        repaired = repair_call_curve(strikes, bad, spot=100.0, discount=1.0)
        intrinsic = np.maximum(100.0 - strikes, 0.0)
        assert np.all(repaired <= 100.0 + 1e-9)
        assert np.all(repaired >= intrinsic - 1e-9)

    def test_preserves_already_valid_curve(self) -> None:
        # A clean convex decreasing curve should be (nearly) unchanged.
        strikes = np.array([90.0, 95.0, 100.0, 105.0, 110.0])
        clean = np.array([11.0, 7.0, 4.0, 2.0, 1.0])
        repaired = repair_call_curve(strikes, clean, spot=100.0, discount=1.0)
        np.testing.assert_allclose(repaired, clean, atol=0.5)

    def test_rejects_unsorted_strikes(self) -> None:
        with pytest.raises(ValidationError):
            repair_call_curve(
                np.array([100.0, 90.0]), np.array([4.0, 11.0]), spot=100.0, discount=1.0
            )


class TestBuildSyntheticChain:
    @pytest.fixture(scope="class")
    def chain(self):
        params = RBergomiParams(
            hurst=0.1, eta=1.5, rho=-0.7, forward_variance=ForwardVariance.flat(0.04)
        )
        grid = TimeGrid.from_calendar_days(calendar_days=10, steps_per_day=8)
        dist = build_terminal_distribution(
            HybridSimulator(params, rng_factory=RandomFactory(1), antithetic=True).simulate(
                grid=grid, n_paths=40_000, initial_spot=100.0
            )
        )
        maker = AvellanedaStoikovMaker(tick_size=0.05)
        strikes = np.arange(80.0, 121.0, 2.5)
        return build_synthetic_chain(dist, maker=maker, strikes=strikes, rate=0.0)

    def test_calls_non_increasing_in_strike(self, chain) -> None:
        values = np.array([chain.call(float(k)).theoretical_value for k in chain.strikes])
        assert np.all(np.diff(values) <= 1e-9)

    def test_calls_convex_in_strike(self, chain) -> None:
        values = np.array([chain.call(float(k)).theoretical_value for k in chain.strikes])
        assert np.all(np.diff(values, 2) >= -1e-6)

    def test_put_call_parity(self, chain) -> None:
        discount = 1.0  # rate = 0
        for k in chain.strikes:
            c = chain.call(float(k)).theoretical_value
            p = chain.put(float(k)).theoretical_value
            # C - P = S - K * discount (within numerical / floor tolerance).
            assert c - p == pytest.approx(chain.spot - float(k) * discount, abs=0.05)

    def test_all_quotes_valid(self, chain) -> None:
        for k in chain.strikes:
            for quoted in (chain.call(float(k)), chain.put(float(k))):
                assert quoted.quote.bid < quoted.quote.ask
                assert quoted.quote.bid >= 0.0

    def test_otm_call_wings_thinner_and_wider(self, chain) -> None:
        atm = chain.call(100.0)
        otm = chain.call(115.0)
        assert otm.quote.bid_size <= atm.quote.bid_size
        assert otm.quote.relative_spread >= atm.quote.relative_spread

    def test_lookup_missing_strike_raises(self, chain) -> None:
        with pytest.raises(ValidationError):
            chain.call(1234.0)

    def test_rejects_bad_strikes(self) -> None:
        params = RBergomiParams(
            hurst=0.1, eta=1.5, rho=-0.7, forward_variance=ForwardVariance.flat(0.04)
        )
        grid = TimeGrid.from_calendar_days(calendar_days=10, steps_per_day=4)
        dist = build_terminal_distribution(
            HybridSimulator(params, rng_factory=RandomFactory(1)).simulate(
                grid=grid, n_paths=5000, initial_spot=100.0
            )
        )
        maker = AvellanedaStoikovMaker()
        with pytest.raises(ValidationError):
            build_synthetic_chain(dist, maker=maker, strikes=np.array([100.0, 90.0]))
