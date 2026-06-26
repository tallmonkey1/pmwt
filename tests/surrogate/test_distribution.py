"""Tests for the quantile-based SurrogateDistribution representation."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from options_engine.core.errors import ValidationError
from options_engine.surrogate.distribution import SurrogateDistribution


def _normal_quantiles(mu: float = 0.0, sigma: float = 0.1):
    levels = np.linspace(0.01, 0.99, 99)
    values = norm.ppf(levels, loc=mu, scale=sigma)
    return levels, values


class TestSurrogateDistribution:
    def _make(self, mu: float = 0.0, sigma: float = 0.1) -> SurrogateDistribution:
        levels, values = _normal_quantiles(mu, sigma)
        return SurrogateDistribution(
            quantile_levels=levels, quantile_values=values, horizon=0.04, initial_spot=100.0
        )

    def test_basic_properties(self) -> None:
        d = self._make()
        assert d.n_quantiles == 99
        assert d.horizon == 0.04

    def test_quantile_interpolation(self) -> None:
        d = self._make(sigma=0.1)
        # Median of a zero-mean normal is ~0.
        assert float(d.quantile(0.5)[0]) == pytest.approx(0.0, abs=1e-2)

    def test_cdf_inverts_quantile(self) -> None:
        d = self._make()
        x = float(d.quantile(0.3)[0])
        assert float(d.cdf(x)[0]) == pytest.approx(0.3, abs=1e-2)

    def test_probability_in_range_matches_normal(self) -> None:
        d = self._make(sigma=0.1)
        # +/- 1 sigma ~ 68%.
        p = d.probability_in_range(-0.1, 0.1)
        assert p == pytest.approx(norm.cdf(1) - norm.cdf(-1), abs=0.02)

    def test_probability_below(self) -> None:
        d = self._make()
        assert d.probability_below(0.0) == pytest.approx(0.5, abs=1e-2)

    def test_sample_inverse_transform(self) -> None:
        d = self._make(sigma=0.1)
        rng = np.random.default_rng(0)
        samples = d.sample(50_000, rng=rng)
        assert samples.shape == (50_000,)
        # Sample std should approximate sigma (slightly less due to clipped tails).
        assert float(np.std(samples)) == pytest.approx(0.1, rel=0.1)

    def test_mean_estimate(self) -> None:
        d = self._make(mu=0.02, sigma=0.1)
        assert d.mean() == pytest.approx(0.02, abs=0.01)

    def test_repairs_tiny_nonmonotonicity(self) -> None:
        levels = np.linspace(0.1, 0.9, 9)
        values = np.array([0.0, 0.1, 0.2, 0.3, 0.3 - 1e-12, 0.5, 0.6, 0.7, 0.8])
        d = SurrogateDistribution(
            quantile_levels=levels, quantile_values=values, horizon=0.04, initial_spot=100.0
        )
        assert np.all(np.diff(d.quantile_values) >= 0.0)

    def test_rejects_crossing_quantiles(self) -> None:
        levels = np.linspace(0.1, 0.9, 5)
        values = np.array([0.0, 0.5, 0.2, 0.6, 0.7])  # 0.5 -> 0.2 is a real crossing
        with pytest.raises(ValidationError):
            SurrogateDistribution(
                quantile_levels=levels, quantile_values=values, horizon=0.04, initial_spot=100.0
            )

    def test_rejects_levels_outside_open_interval(self) -> None:
        with pytest.raises(ValidationError):
            SurrogateDistribution(
                quantile_levels=np.array([0.0, 0.5, 1.0]),
                quantile_values=np.array([-0.1, 0.0, 0.1]),
                horizon=0.04,
                initial_spot=100.0,
            )

    def test_rejects_non_increasing_levels(self) -> None:
        with pytest.raises(ValidationError):
            SurrogateDistribution(
                quantile_levels=np.array([0.5, 0.3]),
                quantile_values=np.array([0.0, 0.1]),
                horizon=0.04,
                initial_spot=100.0,
            )

    def test_terminal_spot_quantile(self) -> None:
        d = self._make()
        median_price = float(d.terminal_spot_quantile(0.5)[0])
        assert median_price == pytest.approx(100.0, abs=1.0)
