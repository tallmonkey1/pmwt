"""Tests for Generalized Advantage Estimation, validated against hand-computed values."""

from __future__ import annotations

import numpy as np
import pytest

from options_engine.agent.gae import compute_gae
from options_engine.core.errors import ValidationError


class TestComputeGAE:
    def test_hand_computed_values(self) -> None:
        # Worked example: gamma=0.9, lam=0.5, terminal at the last step.
        r = np.array([1.0, 1.0, 1.0])
        v = np.array([0.5, 0.5, 0.5])
        nv = np.array([0.5, 0.5, 0.0])
        d = np.array([False, False, True])
        adv, ret = compute_gae(r, v, d, nv, gamma=0.9, lam=0.5)
        # delta_2 = 1 - 0.5 = 0.5; A_2 = 0.5
        # delta_1 = 1 + 0.45 - 0.5 = 0.95; A_1 = 0.95 + 0.45*0.5 = 1.175
        # delta_0 = 0.95; A_0 = 0.95 + 0.45*1.175 = 1.47875
        np.testing.assert_allclose(adv, [1.47875, 1.175, 0.5])
        np.testing.assert_allclose(ret, adv + v)

    def test_reduces_to_monte_carlo_at_unit_gamma_lambda(self) -> None:
        r = np.array([1.0, 2.0, 3.0])
        v = np.zeros(3)
        nv = np.zeros(3)
        d = np.array([False, False, True])
        _, ret = compute_gae(r, v, d, nv, gamma=1.0, lam=1.0)
        # Undiscounted MC returns: [6, 5, 3].
        np.testing.assert_allclose(ret, [6.0, 5.0, 3.0])

    def test_done_masks_bootstrap(self) -> None:
        r = np.array([1.0, 1.0, 1.0])
        v = np.zeros(3)
        nv = np.array([10.0, 0.0, 0.0])  # huge bootstrap that must be masked at t0
        d = np.array([True, False, True])
        adv, _ = compute_gae(r, v, d, nv, gamma=0.9, lam=1.0)
        assert adv[0] == pytest.approx(1.0)  # bootstrap masked => just the reward

    def test_truncation_uses_bootstrap(self) -> None:
        # A non-terminal final step (truncation) should bootstrap from next_value.
        r = np.array([1.0])
        v = np.array([0.0])
        nv = np.array([5.0])
        d = np.array([False])  # truncated, not terminated
        _, ret = compute_gae(r, v, d, nv, gamma=0.9, lam=0.95)
        assert ret[0] == pytest.approx(1.0 + 0.9 * 5.0)

    def test_rejects_shape_mismatch(self) -> None:
        with pytest.raises(ValidationError):
            compute_gae(
                np.array([1.0, 2.0]),
                np.array([0.0]),
                np.array([False]),
                np.array([0.0]),
                gamma=0.9,
                lam=0.95,
            )

    def test_rejects_bad_gamma(self) -> None:
        with pytest.raises(ValidationError):
            compute_gae(
                np.array([1.0]),
                np.array([0.0]),
                np.array([True]),
                np.array([0.0]),
                gamma=1.5,
                lam=0.95,
            )

    def test_rejects_non_finite(self) -> None:
        with pytest.raises(ValidationError):
            compute_gae(
                np.array([np.nan]),
                np.array([0.0]),
                np.array([True]),
                np.array([0.0]),
                gamma=0.9,
                lam=0.95,
            )
