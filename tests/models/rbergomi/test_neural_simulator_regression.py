"""Regression tests for the neural rBergomi simulator's save/load roundtrip.

These tests prevent recurrence of two bugs found by the simulation-correctness
audit:

1. ``NeuralRBergomiConfig.param_dim`` was hardcoded to 8 but the actual parameter
   vector built by ``_alpha_and_params_to_vector`` is length 10 (5 alpha
   components + 5 rBergomi scalars). The fix pins ``param_dim`` to the actual
   size in ``__post_init__``.

2. ``simulate()`` did not call ``self._network.eval()``, so a freshly-trained
   network (still in train() mode) produced different output from a freshly-loaded
   network (set to eval() by load()). The fix forces eval() inside simulate().
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch  # noqa: F401  -- used by the trained neural simulator

from options_engine.core.market_alpha import MarketAlpha
from options_engine.core.random import RandomFactory
from options_engine.core.timegrid import TimeGrid
from options_engine.models.rbergomi import (
    ForwardVariance,
    HybridSimulator,
    RBergomiParams,
)
from options_engine.models.rbergomi.neural_simulator import (
    DEFAULT_PARAM_DIM,
    NeuralRBergomiConfig,
    NeuralRBergomiSimulator,
    _alpha_and_params_to_vector,
    build_dataset_from_paths,
)


def _build_dataset():
    rng = RandomFactory(11)
    params = RBergomiParams(
        hurst=0.1, eta=1.5, rho=-0.7,
        forward_variance=ForwardVariance.flat(0.04),
    )
    grid = TimeGrid(horizon_years=0.1, n_steps=12)
    sim = HybridSimulator(params, rng_factory=rng)
    paths = sim.simulate(grid=grid, n_paths=200, initial_spot=100.0)
    return build_dataset_from_paths(
        paths, params=params, alpha=MarketAlpha.ones(), context_len=4,
    )


def test_param_dim_pinned_to_actual_size() -> None:
    """`param_dim` must equal the length of the parameter vector built by
    `_alpha_and_params_to_vector` (otherwise the linear input projection is wrong-sized).
    """
    alpha = MarketAlpha.ones()
    params = RBergomiParams(
        hurst=0.1, eta=1.5, rho=-0.7,
        forward_variance=ForwardVariance.flat(0.04),
    )
    actual = _alpha_and_params_to_vector(alpha, params)
    assert len(actual) == DEFAULT_PARAM_DIM
    cfg = NeuralRBergomiConfig()
    assert cfg.param_dim == DEFAULT_PARAM_DIM


def test_save_load_roundtrip_produces_identical_paths() -> None:
    """Save -> load -> simulate must produce identical paths as the original
    (same weights + same RNG seed = same output)."""
    cfg = NeuralRBergomiConfig(
        hidden_dim=16, n_layers=2, n_heads=4, context_len=4,
        batch_size=32, n_epochs=3, learning_rate=1e-3, seed=0,
    )
    neural = NeuralRBergomiSimulator(config=cfg)
    neural.train(_build_dataset(), config=cfg)
    with tempfile.TemporaryDirectory() as td:
        path = str(Path(td) / "neural.pt")
        neural.save(path)
        neural2 = NeuralRBergomiSimulator(config=cfg)
        neural2.load(path)
    params = RBergomiParams(
        hurst=0.1, eta=1.5, rho=-0.7,
        forward_variance=ForwardVariance.flat(0.04),
    )
    torch.manual_seed(0)
    out_a = neural.simulate(
        params=params, alpha=MarketAlpha.ones(),
        n_paths=10, n_steps=10, initial_spot=100.0,
    )
    torch.manual_seed(0)
    out_b = neural2.simulate(
        params=params, alpha=MarketAlpha.ones(),
        n_paths=10, n_steps=10, initial_spot=100.0,
    )
    assert np.allclose(out_a.spot, out_b.spot, atol=1e-5, rtol=0), (
        f"save/load roundtrip diverged: max |Δ| = "
        f"{float(np.max(np.abs(out_a.spot - out_b.spot)))}"
    )


def test_simulate_is_reproducible_across_calls() -> None:
    """Two consecutive `simulate` calls with the same inputs must produce
    identical paths (the noise is determined by `config.seed + t`).
    """
    cfg = NeuralRBergomiConfig(
        hidden_dim=16, n_layers=2, n_heads=4, context_len=4,
        batch_size=32, n_epochs=3, learning_rate=1e-3, seed=0,
    )
    neural = NeuralRBergomiSimulator(config=cfg)
    neural.train(_build_dataset(), config=cfg)
    neural._network.eval()
    params = RBergomiParams(
        hurst=0.1, eta=1.5, rho=-0.7,
        forward_variance=ForwardVariance.flat(0.04),
    )
    out_a = neural.simulate(
        params=params, alpha=MarketAlpha.ones(),
        n_paths=10, n_steps=10, initial_spot=100.0,
    )
    out_b = neural.simulate(
        params=params, alpha=MarketAlpha.ones(),
        n_paths=10, n_steps=10, initial_spot=100.0,
    )
    assert np.allclose(out_a.spot, out_b.spot, atol=1e-5, rtol=0), (
        f"two simulate calls diverged: max |Δ| = "
        f"{float(np.max(np.abs(out_a.spot - out_b.spot)))}"
    )


def test_simulate_fresh_after_load_no_explicit_eval() -> None:
    """A freshly-loaded simulator must work correctly even if the caller never
    explicitly calls `eval()` on the network (simulate() must force it).
    """
    cfg = NeuralRBergomiConfig(
        hidden_dim=16, n_layers=2, n_heads=4, context_len=4,
        batch_size=32, n_epochs=3, learning_rate=1e-3, seed=0,
    )
    neural = NeuralRBergomiSimulator(config=cfg)
    neural.train(_build_dataset(), config=cfg)
    with tempfile.TemporaryDirectory() as td:
        path = str(Path(td) / "neural.pt")
        neural.save(path)
        neural2 = NeuralRBergomiSimulator(config=cfg)
        neural2.load(path)
    # Deliberately do NOT call `eval()`; rely on simulate() to force it.
    params = RBergomiParams(
        hurst=0.1, eta=1.5, rho=-0.7,
        forward_variance=ForwardVariance.flat(0.04),
    )
    out = neural2.simulate(
        params=params, alpha=MarketAlpha.ones(),
        n_paths=5, n_steps=5, initial_spot=100.0,
    )
    assert np.all(np.isfinite(out.spot)), "simulate() produced non-finite spot"
