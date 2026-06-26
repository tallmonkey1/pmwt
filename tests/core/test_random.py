"""Tests for reproducible random-number generation."""

from __future__ import annotations

import numpy as np
import pytest

from options_engine.core.errors import ValidationError
from options_engine.core.random import RandomFactory, default_factory


def test_same_seed_same_named_stream_reproducible() -> None:
    a = RandomFactory(42).generator("rbergomi.paths").standard_normal(1000)
    b = RandomFactory(42).generator("rbergomi.paths").standard_normal(1000)
    np.testing.assert_array_equal(a, b)


def test_different_names_are_independent() -> None:
    factory = RandomFactory(42)
    a = factory.generator("stream.a").standard_normal(10000)
    b = factory.generator("stream.b").standard_normal(10000)
    # Independent streams should not be equal and should have low sample correlation.
    assert not np.array_equal(a, b)
    corr = float(np.corrcoef(a, b)[0, 1])
    assert abs(corr) < 0.05


def test_different_seeds_differ() -> None:
    a = RandomFactory(1).generator("x").standard_normal(100)
    b = RandomFactory(2).generator("x").standard_normal(100)
    assert not np.array_equal(a, b)


def test_anonymous_generators_are_independent_but_reproducible() -> None:
    f1 = RandomFactory(7)
    first_a = f1.generator().standard_normal(50)
    second_a = f1.generator().standard_normal(50)
    assert not np.array_equal(first_a, second_a)

    f2 = RandomFactory(7)
    first_b = f2.generator().standard_normal(50)
    second_b = f2.generator().standard_normal(50)
    np.testing.assert_array_equal(first_a, first_b)
    np.testing.assert_array_equal(second_a, second_b)


def test_spawn_returns_independent_reproducible_generators() -> None:
    gens1 = RandomFactory(99).spawn("mc.workers", 4)
    gens2 = RandomFactory(99).spawn("mc.workers", 4)
    assert len(gens1) == 4
    for g1, g2 in zip(gens1, gens2, strict=True):
        np.testing.assert_array_equal(g1.standard_normal(20), g2.standard_normal(20))
    # Distinct workers are independent.
    draws = [RandomFactory(99).spawn("mc.workers", 4)[i].standard_normal(5) for i in range(4)]
    assert not np.array_equal(draws[0], draws[1])


def test_stream_offset_is_stable() -> None:
    # Stability across calls/processes is essential for reproducibility.
    assert RandomFactory._stream_offset("abc") == RandomFactory._stream_offset("abc")
    assert RandomFactory._stream_offset("abc") != RandomFactory._stream_offset("abd")


@pytest.mark.parametrize("bad", [-1, True, 1.5, "x"])
def test_invalid_seed_rejected(bad: object) -> None:
    with pytest.raises((ValidationError, TypeError)):
        RandomFactory(bad)  # type: ignore[arg-type]


def test_empty_stream_name_rejected() -> None:
    with pytest.raises(ValidationError):
        RandomFactory(0).generator("")


def test_invalid_spawn_count_rejected() -> None:
    with pytest.raises(ValidationError):
        RandomFactory(0).spawn("x", 0)


def test_default_factory() -> None:
    assert isinstance(default_factory(3), RandomFactory)
    assert default_factory(3).seed == 3
