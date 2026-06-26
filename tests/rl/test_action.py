"""Tests for the parameterized action space and decoder."""

from __future__ import annotations

import numpy as np
import pytest

from options_engine.core.enums import StrategicAction
from options_engine.core.errors import ValidationError
from options_engine.rl.action import (
    ACTION_DIM,
    ActionBounds,
    decode_action,
)


class TestDecodeAction:
    def test_strategic_argmax(self) -> None:
        # Highest logit selects the strategic action.
        theta = decode_action(np.array([5.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        gamma = decode_action(np.array([0.0, 5.0, 0.0, 0.0, 0.0, 0.0]))
        flat = decode_action(np.array([0.0, 0.0, 5.0, 0.0, 0.0, 0.0]))
        assert theta.strategic is StrategicAction.HARVEST_THETA
        assert gamma.strategic is StrategicAction.HARVEST_GAMMA
        assert flat.strategic is StrategicAction.FLAT
        assert theta.is_trade
        assert not flat.is_trade

    def test_knob_mapping_endpoints(self) -> None:
        bounds = ActionBounds(
            tail_probability=(0.05, 0.35),
            wing_width_fraction=(0.01, 0.08),
            size_fraction=(0.0, 1.0),
        )
        low = decode_action(np.array([1.0, 0, 0, -1.0, -1.0, -1.0]), bounds=bounds)
        high = decode_action(np.array([1.0, 0, 0, 1.0, 1.0, 1.0]), bounds=bounds)
        assert low.tail_probability == pytest.approx(0.05)
        assert high.tail_probability == pytest.approx(0.35)
        assert low.wing_width_fraction == pytest.approx(0.01)
        assert high.size_fraction == pytest.approx(1.0)

    def test_midpoint(self) -> None:
        bounds = ActionBounds(tail_probability=(0.05, 0.35))
        mid = decode_action(np.array([1.0, 0, 0, 0.0, 0.0, 0.0]), bounds=bounds)
        assert mid.tail_probability == pytest.approx(0.2)

    def test_out_of_range_knobs_clipped(self) -> None:
        # Wild policy outputs are clipped, never an error.
        d = decode_action(np.array([1.0, 0, 0, 100.0, -100.0, 50.0]))
        assert 0.0 < d.tail_probability < 0.5
        assert d.wing_width_fraction > 0.0

    def test_rejects_wrong_shape(self) -> None:
        with pytest.raises(ValidationError):
            decode_action(np.zeros(ACTION_DIM + 1))

    def test_rejects_non_finite(self) -> None:
        bad = np.zeros(ACTION_DIM)
        bad[3] = np.nan
        with pytest.raises(ValidationError):
            decode_action(bad)


class TestActionBounds:
    def test_rejects_inverted_range(self) -> None:
        with pytest.raises(ValidationError):
            ActionBounds(tail_probability=(0.35, 0.05))

    def test_rejects_tail_out_of_domain(self) -> None:
        with pytest.raises(ValidationError):
            ActionBounds(tail_probability=(0.05, 0.6))
