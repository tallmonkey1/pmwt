"""Tests for variance-reduction noise generation."""

from __future__ import annotations

import numpy as np
import pytest

from options_engine.core.errors import ValidationError
from options_engine.core.random import RandomFactory
from options_engine.models.rbergomi.noise import draw_standard_normals


def _rng() -> np.random.Generator:
    return RandomFactory(123).generator("test")


def test_shape_and_finiteness() -> None:
    x = draw_standard_normals(100, 5, rng=_rng())
    assert x.shape == (100, 5)
    assert np.all(np.isfinite(x))


def test_reproducible() -> None:
    a = draw_standard_normals(50, 3, rng=RandomFactory(7).generator("s"))
    b = draw_standard_normals(50, 3, rng=RandomFactory(7).generator("s"))
    np.testing.assert_array_equal(a, b)


def test_antithetic_exact_mirror_even() -> None:
    x = draw_standard_normals(10, 4, rng=_rng(), antithetic=True)
    np.testing.assert_array_equal(x[5:], -x[:5])


def test_antithetic_honours_odd_count() -> None:
    x = draw_standard_normals(11, 2, rng=_rng(), antithetic=True)
    assert x.shape == (11, 2)
    # First 6 are primal, next 5 mirror the first 5.
    np.testing.assert_array_equal(x[6:11], -x[:5])


def test_antithetic_sample_mean_near_zero() -> None:
    x = draw_standard_normals(10_000, 1, rng=_rng(), antithetic=True)
    # Antithetic draws make the sample mean of an odd functional exactly ~0.
    assert abs(float(np.mean(x))) < 1e-12


def test_quasi_random_shape_and_finite() -> None:
    x = draw_standard_normals(256, 4, rng=_rng(), quasi_random=True)
    assert x.shape == (256, 4)
    assert np.all(np.isfinite(x))


def test_quasi_random_better_uniformity() -> None:
    # Sobol-based normals should approximate the standard normal moments well.
    x = draw_standard_normals(4096, 1, rng=_rng(), quasi_random=True)
    assert abs(float(np.mean(x))) < 0.05
    assert abs(float(np.std(x)) - 1.0) < 0.05


def test_quasi_random_with_antithetic() -> None:
    x = draw_standard_normals(128, 2, rng=_rng(), antithetic=True, quasi_random=True)
    assert x.shape == (128, 2)
    np.testing.assert_array_equal(x[64:], -x[:64])


def test_invalid_dimensions() -> None:
    with pytest.raises(ValidationError):
        draw_standard_normals(0, 2, rng=_rng())
    with pytest.raises(ValidationError):
        draw_standard_normals(10, 0, rng=_rng())
