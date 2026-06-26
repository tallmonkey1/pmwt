"""Tests for the validation helpers, including property-based coverage."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from options_engine.core.errors import ValidationError
from options_engine.core.validation import (
    check_array_finite,
    check_correlation,
    check_finite,
    check_in_range,
    check_non_empty,
    check_non_negative,
    check_positive,
    check_probability,
    check_same_length,
    check_unit_interval,
)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_check_finite_rejects_non_finite(bad: float) -> None:
    with pytest.raises(ValidationError):
        check_finite(bad, name="x")


def test_check_finite_rejects_bool() -> None:
    # bool is a subclass of int but is not a meaningful numeric input here.
    with pytest.raises(ValidationError):
        check_finite(True, name="flag")  # type: ignore[arg-type]


def test_check_finite_accepts_int_and_float() -> None:
    assert check_finite(3, name="x") == 3.0
    assert check_finite(2.5, name="x") == 2.5


def test_check_positive() -> None:
    assert check_positive(1e-9, name="x") == 1e-9
    for bad in (0.0, -1.0):
        with pytest.raises(ValidationError):
            check_positive(bad, name="x")


def test_check_non_negative() -> None:
    assert check_non_negative(0.0, name="x") == 0.0
    with pytest.raises(ValidationError):
        check_non_negative(-0.1, name="x")


def test_check_in_range_inclusive_and_exclusive() -> None:
    assert check_in_range(0.0, name="x", low=0.0, high=1.0) == 0.0
    with pytest.raises(ValidationError):
        check_in_range(0.0, name="x", low=0.0, high=1.0, inclusive=False)


def test_check_in_range_rejects_inverted_bounds() -> None:
    with pytest.raises(ValidationError):
        check_in_range(0.5, name="x", low=1.0, high=0.0)


def test_check_unit_interval_and_probability() -> None:
    assert check_unit_interval(0.5, name="p") == 0.5
    assert check_probability(1.0, name="p") == 1.0
    with pytest.raises(ValidationError):
        check_probability(1.0001, name="p")


def test_check_correlation() -> None:
    assert check_correlation(-1.0, name="rho") == -1.0
    assert check_correlation(1.0, name="rho") == 1.0
    with pytest.raises(ValidationError):
        check_correlation(-1.5, name="rho")


def test_check_array_finite() -> None:
    arr = np.array([1.0, 2.0, 3.0])
    assert check_array_finite(arr, name="a") is not None
    with pytest.raises(ValidationError):
        check_array_finite(np.array([1.0, np.nan]), name="a")


def test_check_array_finite_accepts_empty() -> None:
    check_array_finite(np.array([]), name="a")


def test_check_non_empty() -> None:
    check_non_empty([1], name="a")
    with pytest.raises(ValidationError):
        check_non_empty([], name="a")


def test_check_same_length() -> None:
    check_same_length(("a", [1, 2]), ("b", [3, 4]))
    with pytest.raises(ValidationError):
        check_same_length(("a", [1, 2]), ("b", [3]))


def test_check_same_length_empty_call_is_noop() -> None:
    check_same_length()


# --- Property-based tests -------------------------------------------------------------


@given(st.floats(min_value=1e-12, max_value=1e12, allow_nan=False, allow_infinity=False))
def test_positive_property(x: float) -> None:
    assert check_positive(x, name="x") == x


@given(st.floats(allow_nan=False, allow_infinity=False))
def test_in_range_is_consistent_with_comparison(x: float) -> None:
    low, high = -10.0, 10.0
    if low <= x <= high:
        assert check_in_range(x, name="x", low=low, high=high) == x
    else:
        with pytest.raises(ValidationError):
            check_in_range(x, name="x", low=low, high=high)
