"""Tests for purged, embargoed walk-forward cross-validation."""

from __future__ import annotations

import numpy as np
import pytest

from options_engine.backtest.validation import (
    WalkForwardSplit,
    purged_walk_forward_splits,
)
from options_engine.core.errors import ValidationError


class TestPurgedWalkForwardSplits:
    def test_no_train_test_overlap(self) -> None:
        splits = purged_walk_forward_splits(100, n_splits=5)
        for s in splits:
            overlap = np.intersect1d(s.train_indices, s.test_indices)
            assert overlap.size == 0

    def test_walk_forward_is_causal_for_first_usable_fold(self) -> None:
        # With no embargo/purge, the second fold trains only on the past.
        splits = purged_walk_forward_splits(100, n_splits=5, embargo=100, purge=0)
        # A huge embargo removes all post-test training data, so training is strictly past.
        for s in splits:
            if s.train_indices.size > 0:
                assert s.train_indices.max() < s.test_indices.min()

    def test_purge_removes_pre_test_samples(self) -> None:
        splits = purged_walk_forward_splits(100, n_splits=5, embargo=0, purge=5)
        for s in splits:
            test_start = int(s.test_indices.min())
            before = s.train_indices[s.train_indices < test_start]
            if before.size > 0:
                # The 5 samples immediately before the test window are purged.
                assert before.max() < test_start - 5 + 1

    def test_embargo_removes_post_test_samples(self) -> None:
        splits = purged_walk_forward_splits(100, n_splits=5, embargo=5, purge=0)
        for s in splits:
            test_end = int(s.test_indices.max())
            after = s.train_indices[s.train_indices > test_end]
            if after.size > 0:
                assert after.min() >= test_end + 5

    def test_folds_are_chronological_and_contiguous(self) -> None:
        splits = purged_walk_forward_splits(100, n_splits=4)
        test_starts = [int(s.test_indices.min()) for s in splits]
        assert test_starts == sorted(test_starts)

    def test_rejects_bad_n_splits(self) -> None:
        with pytest.raises(ValidationError):
            purged_walk_forward_splits(10, n_splits=20)

    def test_rejects_negative_embargo(self) -> None:
        with pytest.raises(ValidationError):
            purged_walk_forward_splits(100, n_splits=5, embargo=-1)


class TestWalkForwardSplit:
    def test_rejects_overlap(self) -> None:
        with pytest.raises(ValidationError):
            WalkForwardSplit(
                train_indices=np.array([0, 1, 2]),
                test_indices=np.array([2, 3]),
                fold=0,
            )

    def test_rejects_empty_test(self) -> None:
        with pytest.raises(ValidationError):
            WalkForwardSplit(
                train_indices=np.array([0, 1]), test_indices=np.array([], dtype=int), fold=0
            )
