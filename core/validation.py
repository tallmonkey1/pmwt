"""Lightweight, dependency-free validation helpers.

These helpers exist so that every numerical entry point in the engine can fail *early*
and *loudly* with an actionable message (institutional-grade requirements: input
validation, error handling, no silent failures).

They are deliberately small and composable. For structured/config objects we use
pydantic models elsewhere; these helpers cover the scalar/array preconditions that recur
throughout the quantitative core where pulling in pydantic would be overkill.

Every function raises :class:`~options_engine.core.errors.ValidationError` on failure and
returns the validated value on success, so they can be used inline::

    h = check_unit_interval(hurst, name="hurst", inclusive=False)
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from .errors import ValidationError

__all__ = [
    "check_array_finite",
    "check_correlation",
    "check_finite",
    "check_in_range",
    "check_non_empty",
    "check_non_negative",
    "check_positive",
    "check_probability",
    "check_same_length",
    "check_unit_interval",
]


def check_finite(value: float, *, name: str) -> float:
    """Return ``value`` if it is a finite real number, else raise.

    Rejects NaN and +/- infinity, the two failure modes most likely to silently corrupt
    a downstream computation.
    """
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValidationError(
            f"'{name}' must be a real number", context={"name": name, "value": value}
        )
    if not math.isfinite(float(value)):
        raise ValidationError(f"'{name}' must be finite", context={"name": name, "value": value})
    return float(value)


def check_positive(value: float, *, name: str) -> float:
    """Return ``value`` if finite and strictly greater than zero, else raise."""
    value = check_finite(value, name=name)
    if value <= 0.0:
        raise ValidationError(
            f"'{name}' must be strictly positive", context={"name": name, "value": value}
        )
    return value


def check_non_negative(value: float, *, name: str) -> float:
    """Return ``value`` if finite and >= 0, else raise."""
    value = check_finite(value, name=name)
    if value < 0.0:
        raise ValidationError(
            f"'{name}' must be non-negative", context={"name": name, "value": value}
        )
    return value


def check_in_range(
    value: float,
    *,
    name: str,
    low: float,
    high: float,
    inclusive: bool = True,
) -> float:
    """Return ``value`` if it lies in ``[low, high]`` (or open interval), else raise."""
    value = check_finite(value, name=name)
    if low > high:
        raise ValidationError(
            "invalid range: low exceeds high",
            context={"name": name, "low": low, "high": high},
        )
    in_bounds = low <= value <= high if inclusive else low < value < high
    if not in_bounds:
        bounds = f"[{low}, {high}]" if inclusive else f"({low}, {high})"
        raise ValidationError(
            f"'{name}' must lie in {bounds}",
            context={"name": name, "value": value, "low": low, "high": high},
        )
    return value


def check_unit_interval(value: float, *, name: str, inclusive: bool = True) -> float:
    """Return ``value`` if it lies in the unit interval, else raise."""
    return check_in_range(value, name=name, low=0.0, high=1.0, inclusive=inclusive)


def check_probability(value: float, *, name: str) -> float:
    """Return ``value`` if it is a valid probability in [0, 1], else raise."""
    return check_unit_interval(value, name=name, inclusive=True)


def check_correlation(value: float, *, name: str) -> float:
    """Return ``value`` if it is a valid correlation in [-1, 1], else raise."""
    return check_in_range(value, name=name, low=-1.0, high=1.0, inclusive=True)


def check_array_finite(array: NDArray[np.floating], *, name: str) -> NDArray[np.floating]:
    """Return ``array`` if it contains only finite values, else raise.

    Operates on the array as-is (no copy). Empty arrays are accepted here; use
    :func:`check_non_empty` separately when emptiness is itself an error.
    """
    arr = np.asarray(array)
    if arr.size and not np.all(np.isfinite(arr)):
        n_bad = int(np.count_nonzero(~np.isfinite(arr)))
        raise ValidationError(
            f"'{name}' contains {n_bad} non-finite value(s)",
            context={"name": name, "n_bad": n_bad, "size": int(arr.size)},
        )
    return arr


def check_non_empty(seq: Sequence[object] | NDArray[np.floating], *, name: str) -> None:
    """Raise if ``seq`` has length zero."""
    if len(seq) == 0:
        raise ValidationError(f"'{name}' must be non-empty", context={"name": name})


def check_same_length(*named_sequences: tuple[str, Sequence[object]]) -> None:
    """Raise unless all provided named sequences share the same length.

    Usage::

        check_same_length(("strikes", strikes), ("prices", prices))
    """
    if not named_sequences:
        return
    lengths = {name: len(seq) for name, seq in named_sequences}
    unique = set(lengths.values())
    if len(unique) > 1:
        raise ValidationError("sequences must have equal length", context={"lengths": lengths})
