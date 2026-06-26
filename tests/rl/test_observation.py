"""Tests for observation construction and bounds."""

from __future__ import annotations

import numpy as np
import pytest

from options_engine.core.enums import VolRegime
from options_engine.core.errors import ValidationError
from options_engine.regime.detector import RegimeNowcast
from options_engine.rl.observation import (
    OBSERVATION_DIM,
    ObservationInputs,
    build_observation,
    observation_bounds,
)


def _regime() -> RegimeNowcast:
    return RegimeNowcast(
        current_probabilities={VolRegime.LOW: 0.8, VolRegime.MID: 0.15, VolRegime.HIGH: 0.05},
        next_probabilities={VolRegime.LOW: 0.75, VolRegime.MID: 0.2, VolRegime.HIGH: 0.05},
    )


def _inputs(**kw) -> ObservationInputs:
    defaults = {
        "prob_in_one_sigma": 0.7,
        "left_tail_prob": 0.05,
        "right_tail_prob": 0.05,
        "expected_move": 0.1,
        "regime": _regime(),
        "atm_relative_spread": 0.3,
        "margin_utilization": 0.5,
        "drawdown": 0.1,
        "open_position_fraction": 0.2,
        "time_to_expiry_fraction": 0.8,
        "news_cooloff_active": False,
    }
    defaults.update(kw)
    return ObservationInputs(**defaults)  # type: ignore[arg-type]


class TestBuildObservation:
    def test_shape_and_dtype(self) -> None:
        obs = build_observation(_inputs())
        assert obs.shape == (OBSERVATION_DIM,)
        assert obs.dtype == np.float32

    def test_within_bounds(self) -> None:
        obs = build_observation(_inputs())
        low, high = observation_bounds()
        assert np.all(obs >= low) and np.all(obs <= high)

    def test_extreme_inputs_clipped_within_bounds(self) -> None:
        # Out-of-range upstream values are clipped, keeping the observation valid.
        obs = build_observation(
            _inputs(margin_utilization=10.0, drawdown=5.0, atm_relative_spread=99.0)
        )
        low, high = observation_bounds()
        assert np.all(obs >= low) and np.all(obs <= high)

    def test_news_flag_encoded(self) -> None:
        on = build_observation(_inputs(news_cooloff_active=True))
        off = build_observation(_inputs(news_cooloff_active=False))
        assert on[-1] == 1.0
        assert off[-1] == 0.0

    def test_rejects_non_finite(self) -> None:
        with pytest.raises(ValidationError):
            build_observation(_inputs(prob_in_one_sigma=np.nan))


class TestObservationBounds:
    def test_shapes(self) -> None:
        low, high = observation_bounds()
        assert low.shape == (OBSERVATION_DIM,)
        assert high.shape == (OBSERVATION_DIM,)
        assert np.all(high >= low)
