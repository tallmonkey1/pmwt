"""Tests for the structured exception hierarchy."""

from __future__ import annotations

import pytest

from options_engine.core.errors import (
    CalibrationError,
    ConvergenceError,
    NumericalError,
    OptionsEngineError,
    RiskLimitError,
    ValidationError,
)


def test_all_exceptions_inherit_root() -> None:
    for exc_type in (
        ValidationError,
        NumericalError,
        ConvergenceError,
        CalibrationError,
        RiskLimitError,
    ):
        assert issubclass(exc_type, OptionsEngineError)


def test_convergence_is_numerical() -> None:
    assert issubclass(ConvergenceError, NumericalError)


def test_context_is_copied_not_aliased() -> None:
    ctx = {"a": 1}
    err = OptionsEngineError("boom", context=ctx)
    ctx["a"] = 2
    assert err.context == {"a": 1}


def test_str_includes_sorted_context() -> None:
    err = OptionsEngineError("failure", context={"b": 2, "a": 1})
    assert str(err) == "failure (a=1, b=2)"


def test_str_without_context() -> None:
    assert str(OptionsEngineError("simple")) == "simple"


def test_can_be_raised_and_caught_via_root() -> None:
    with pytest.raises(OptionsEngineError):
        raise CalibrationError("bad fit", context={"iter": 50})
