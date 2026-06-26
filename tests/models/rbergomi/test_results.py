"""Tests for the simulation result containers."""

from __future__ import annotations

import numpy as np
import pytest

from options_engine.core.errors import ValidationError
from options_engine.models.rbergomi.results import SimulationPaths, TerminalDistribution


def _make_paths() -> SimulationPaths:
    times = np.array([0.0, 0.5, 1.0])
    spot = np.array([[100.0, 101.0, 102.0], [100.0, 99.0, 98.0]])
    variance = np.array([[0.04, 0.04, 0.04], [0.04, 0.05, 0.06]])
    return SimulationPaths(times=times, spot=spot, variance=variance)


class TestSimulationPaths:
    def test_basic_properties(self) -> None:
        p = _make_paths()
        assert p.n_paths == 2
        assert p.n_steps == 2
        assert p.horizon == 1.0
        assert p.initial_spot == 100.0

    def test_terminal_spot_and_log_return(self) -> None:
        p = _make_paths()
        np.testing.assert_allclose(p.terminal_spot(), [102.0, 98.0])
        np.testing.assert_allclose(
            p.terminal_log_return(), np.log(np.array([102.0, 98.0]) / 100.0)
        )

    def test_rejects_nonpositive_spot(self) -> None:
        with pytest.raises(ValidationError):
            SimulationPaths(
                times=np.array([0.0, 1.0]),
                spot=np.array([[100.0, 0.0]]),
                variance=np.array([[0.04, 0.04]]),
            )

    def test_rejects_negative_variance(self) -> None:
        with pytest.raises(ValidationError):
            SimulationPaths(
                times=np.array([0.0, 1.0]),
                spot=np.array([[100.0, 101.0]]),
                variance=np.array([[0.04, -0.01]]),
            )

    def test_rejects_shape_mismatch(self) -> None:
        with pytest.raises(ValidationError):
            SimulationPaths(
                times=np.array([0.0, 1.0, 2.0]),
                spot=np.array([[100.0, 101.0]]),
                variance=np.array([[0.04, 0.04]]),
            )

    def test_rejects_non_finite(self) -> None:
        with pytest.raises(ValidationError):
            SimulationPaths(
                times=np.array([0.0, 1.0]),
                spot=np.array([[100.0, np.nan]]),
                variance=np.array([[0.04, 0.04]]),
            )

    def test_immutable(self) -> None:
        p = _make_paths()
        with pytest.raises(ValueError):
            p.spot[0, 0] = 1.0


class TestTerminalDistribution:
    def _make(self) -> TerminalDistribution:
        rng = np.random.default_rng(0)
        lr = rng.normal(0.0, 0.1, size=10_000)
        return TerminalDistribution(
            log_returns=lr, horizon=1.0, initial_spot=100.0, mean_standard_error=0.001
        )

    def test_n_paths(self) -> None:
        assert self._make().n_paths == 10_000

    def test_terminal_spot(self) -> None:
        d = self._make()
        np.testing.assert_allclose(d.terminal_spot(), 100.0 * np.exp(d.log_returns))

    def test_quantile_monotone(self) -> None:
        d = self._make()
        q = d.quantile(np.array([0.1, 0.5, 0.9]))
        assert q[0] < q[1] < q[2]

    def test_quantile_out_of_range(self) -> None:
        with pytest.raises(ValidationError):
            self._make().quantile(1.5)

    def test_probability_below(self) -> None:
        d = self._make()
        # Median ~ 0, so P(below 0) ~ 0.5.
        assert abs(d.probability_below(0.0) - 0.5) < 0.05

    def test_probability_in_range(self) -> None:
        d = self._make()
        p = d.probability_in_range(-0.1, 0.1)
        # +/- 1 sigma ~ 68%.
        assert 0.6 < p < 0.75

    def test_probability_in_range_rejects_inverted(self) -> None:
        with pytest.raises(ValidationError):
            self._make().probability_in_range(0.1, -0.1)

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValidationError):
            TerminalDistribution(
                log_returns=np.array([]),
                horizon=1.0,
                initial_spot=100.0,
                mean_standard_error=0.0,
            )
