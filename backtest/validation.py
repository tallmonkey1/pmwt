r"""Purged, embargoed walk-forward cross-validation (SPEC §9).

Standard k-fold CV leaks information in time-series finance: a test fold can sit *between*
training data, letting the model peek across the boundary, and overlapping label horizons let
adjacent samples leak into one another. López de Prado's remedy is **purged, embargoed
walk-forward** validation:

* **Walk-forward** -- always train on the past and test on the immediate future (never the
  reverse), so the evaluation mirrors live deployment.
* **Purge** -- drop training samples whose label horizon overlaps the test window, removing
  the leakage from overlapping outcomes.
* **Embargo** -- additionally drop a buffer of training samples immediately *after* the test
  window, removing serial-correlation leakage.

This module produces the index splits; the caller runs a backtest on each test fold. Keeping
it as pure index arithmetic makes the leakage-prevention logic explicit and unit-testable --
the whole point is that you can *see* there is no look-ahead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..core.errors import ValidationError

__all__ = ["WalkForwardSplit", "purged_walk_forward_splits"]


@dataclass(frozen=True, slots=True)
class WalkForwardSplit:
    """Train/test index arrays for one walk-forward fold."""

    train_indices: NDArray[np.int_]
    test_indices: NDArray[np.int_]
    fold: int

    def __post_init__(self) -> None:
        if self.test_indices.size == 0:
            raise ValidationError("test_indices must be non-empty", context={"fold": self.fold})
        # The core no-look-ahead invariant: every training index precedes every test index, or
        # follows the embargo after it -- never inside the test window.
        if self.train_indices.size > 0:
            overlap = np.intersect1d(self.train_indices, self.test_indices)
            if overlap.size > 0:
                raise ValidationError(
                    "train and test indices overlap (leakage)", context={"fold": self.fold}
                )


def purged_walk_forward_splits(
    n_samples: int,
    *,
    n_splits: int,
    embargo: int = 0,
    purge: int = 0,
) -> list[WalkForwardSplit]:
    """Generate purged, embargoed walk-forward train/test splits over ``n_samples``.

    The series is divided into ``n_splits`` contiguous, equal-ish test folds in chronological
    order. For each fold, the training set is everything *before* the test window minus the
    last ``purge`` samples (overlap purge), plus everything *after* the test window starting
    ``embargo`` samples later (embargo). The first fold has no prior data and is skipped as a
    test fold (there is nothing to train on), matching honest walk-forward practice.

    Parameters
    ----------
    n_samples:
        Total number of (chronologically ordered) samples.
    n_splits:
        Number of contiguous test folds.
    embargo:
        Number of post-test-window samples to exclude from training.
    purge:
        Number of pre-test-window samples to exclude from training (overlap purge).

    Returns
    -------
    list[WalkForwardSplit]
        One split per usable fold, in chronological order.
    """
    if n_samples < 2:
        raise ValidationError("n_samples must be >= 2", context={"n_samples": n_samples})
    if n_splits < 1 or n_splits > n_samples:
        raise ValidationError(
            "n_splits must satisfy 1 <= n_splits <= n_samples",
            context={"n_splits": n_splits, "n_samples": n_samples},
        )
    if embargo < 0 or purge < 0:
        raise ValidationError("embargo and purge must be non-negative", context={})

    fold_edges = np.linspace(0, n_samples, n_splits + 1, dtype=int)
    splits: list[WalkForwardSplit] = []
    all_idx = np.arange(n_samples)

    for fold in range(n_splits):
        test_start = int(fold_edges[fold])
        test_end = int(fold_edges[fold + 1])  # exclusive
        if test_end <= test_start:
            continue
        test_idx = all_idx[test_start:test_end]

        # Training = before the (purged) test window, plus after the (embargoed) test window.
        before_end = max(0, test_start - purge)
        after_start = min(n_samples, test_end + embargo)
        train_before = all_idx[:before_end]
        train_after = all_idx[after_start:]
        train_idx = np.concatenate([train_before, train_after])

        if train_idx.size == 0:
            # No data to train on (typically the first fold); skip as a test fold.
            continue
        splits.append(WalkForwardSplit(train_indices=train_idx, test_indices=test_idx, fold=fold))

    if not splits:
        raise ValidationError(
            "no usable walk-forward folds produced; reduce n_splits/embargo/purge",
            context={"n_samples": n_samples, "n_splits": n_splits},
        )
    return splits
