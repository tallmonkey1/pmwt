"""Tests for the on-policy rollout buffer."""

from __future__ import annotations

import numpy as np
import pytest

from options_engine.agent.rollout import RolloutBuffer
from options_engine.core.errors import ValidationError


def _fill(buffer: RolloutBuffer, n: int, *, reward: float = 1.0) -> None:
    for _ in range(n):
        buffer.add(
            obs=np.ones(buffer._obs_dim, dtype=np.float32),
            action=np.zeros(buffer._action_dim, dtype=np.float32),
            log_prob=0.0,
            reward=reward,
            value=0.5,
            next_value=0.5,
            done=False,
        )


class TestRolloutBuffer:
    def test_fills_and_reports_full(self) -> None:
        buf = RolloutBuffer(capacity=4, obs_dim=3, action_dim=2, gamma=0.99, lam=0.95)
        assert not buf.is_full
        _fill(buf, 4)
        assert buf.is_full

    def test_overfill_rejected(self) -> None:
        buf = RolloutBuffer(capacity=2, obs_dim=3, action_dim=2, gamma=0.99, lam=0.95)
        _fill(buf, 2)
        with pytest.raises(ValidationError):
            _fill(buf, 1)

    def test_compute_requires_full(self) -> None:
        buf = RolloutBuffer(capacity=4, obs_dim=3, action_dim=2, gamma=0.99, lam=0.95)
        _fill(buf, 2)
        with pytest.raises(ValidationError):
            buf.compute_advantages()

    def test_minibatches_cover_all_samples(self) -> None:
        buf = RolloutBuffer(capacity=10, obs_dim=3, action_dim=2, gamma=0.99, lam=0.95)
        _fill(buf, 10)
        buf.compute_advantages()
        rng = np.random.default_rng(0)
        seen = 0
        for batch in buf.iter_minibatches(batch_size=4, rng=rng):
            seen += batch.observations.shape[0]
            assert batch.observations.shape[1] == 3
            assert batch.actions.shape[1] == 2
        assert seen == 10  # every sample appears exactly once

    def test_advantage_normalization(self) -> None:
        buf = RolloutBuffer(capacity=8, obs_dim=1, action_dim=1, gamma=0.99, lam=0.95)
        _fill(buf, 8)
        buf.compute_advantages()
        rng = np.random.default_rng(0)
        all_adv = np.concatenate(
            [
                b.advantages
                for b in buf.iter_minibatches(batch_size=8, rng=rng, normalize_advantages=True)
            ]
        )
        # Normalized advantages have ~zero mean and ~unit std (single full batch).
        assert abs(float(np.mean(all_adv))) < 1e-5
        assert abs(float(np.std(all_adv)) - 1.0) < 1e-4

    def test_reset(self) -> None:
        buf = RolloutBuffer(capacity=4, obs_dim=3, action_dim=2, gamma=0.99, lam=0.95)
        _fill(buf, 4)
        buf.reset()
        assert not buf.is_full
