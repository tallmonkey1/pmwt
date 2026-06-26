"""Shared pytest fixtures and configuration.

Keeping fixtures centralized here avoids duplication and makes test setup consistent
across the suite (SPEC §10).
"""

from __future__ import annotations

import pytest

from options_engine.core import RandomFactory


@pytest.fixture
def rng_factory() -> RandomFactory:
    """A deterministic random factory seeded for reproducible tests."""
    return RandomFactory(seed=12345)
