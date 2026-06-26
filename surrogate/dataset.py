r"""Training-data generation for the distribution surrogate.

The surrogate learns to reproduce the Monte-Carlo terminal distribution, so its training
labels *are* Monte-Carlo quantiles (SPEC §2.5: "a neural model is trained to approximate
the Monte-Carlo terminal distribution of the simulator"). This module samples rBergomi
parameter scenarios from a configurable prior, runs the simulator for each, and records the
empirical terminal-log-return quantiles as labels.

Everything is reproducible (an injected :class:`RandomFactory`) and validated. The sampling
ranges default to the empirically-plausible rough-volatility regime; they are exposed so the
training distribution can be matched to a specific universe later.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..core.errors import ValidationError
from ..core.random import RandomFactory
from ..core.timegrid import TRADING_DAYS_PER_YEAR, TimeGrid
from ..core.validation import check_positive
from ..models.rbergomi import ForwardVariance, HybridSimulator, RBergomiParams
from .features import RawInputs, build_feature_matrix

__all__ = ["ScenarioRanges", "TrainingData", "default_quantile_levels", "generate_training_data"]


def default_quantile_levels(n: int = 99) -> NDArray[np.float64]:
    """Return ``n`` evenly-spaced quantile levels in the open interval ``(0, 1)``.

    For ``n = 99`` this is the percentiles ``0.01, 0.02, ..., 0.99`` -- a dense grid that
    captures the tails relevant to deep-OTM option pricing without touching the endpoints
    (where empirical quantiles are unstable).
    """
    if n < 2:
        raise ValidationError("n must be >= 2", context={"n": n})
    return np.linspace(1.0 / (n + 1), n / (n + 1), n)


@dataclass(frozen=True, slots=True)
class ScenarioRanges:
    """Uniform sampling ranges for rBergomi scenario parameters.

    Defaults cover the rough-volatility regime (small ``H``, negative ``rho``) and short
    holding horizons consistent with the engine's NORMAL/MFD modes.
    """

    hurst: tuple[float, float] = (0.05, 0.45)
    eta: tuple[float, float] = (0.5, 3.0)
    rho: tuple[float, float] = (-0.95, -0.05)
    xi0: tuple[float, float] = (0.01, 0.16)  # variance: ~10%-40% vol
    horizon_days: tuple[float, float] = (1.0, 30.0)

    def __post_init__(self) -> None:
        for name, (lo, hi) in (
            ("hurst", self.hurst),
            ("eta", self.eta),
            ("rho", self.rho),
            ("xi0", self.xi0),
            ("horizon_days", self.horizon_days),
        ):
            if not (lo < hi):
                raise ValidationError(
                    f"{name} range must satisfy lo < hi", context={"lo": lo, "hi": hi}
                )
        if self.hurst[0] <= 0.0 or self.hurst[1] >= 0.5:
            raise ValidationError("hurst range must lie within (0, 0.5)", context={})
        if self.eta[0] <= 0.0 or self.xi0[0] <= 0.0 or self.horizon_days[0] <= 0.0:
            raise ValidationError("eta, xi0 and horizon must be strictly positive", context={})
        if self.rho[0] < -1.0 or self.rho[1] > 1.0:
            raise ValidationError("rho range must lie within [-1, 1]", context={})


@dataclass(frozen=True, slots=True)
class TrainingData:
    """A generated training set: features, raw inputs, and target MC quantiles."""

    features: NDArray[np.float64]  # (n_samples, N_FEATURES), unscaled
    quantile_levels: NDArray[np.float64]  # (Q,)
    quantiles: NDArray[np.float64]  # (n_samples, Q)
    raw_inputs: RawInputs

    @property
    def n_samples(self) -> int:
        """Number of training scenarios."""
        return int(self.features.shape[0])


def generate_training_data(
    *,
    n_scenarios: int,
    rng_factory: RandomFactory,
    ranges: ScenarioRanges | None = None,
    quantile_levels: NDArray[np.float64] | None = None,
    n_paths: int = 20_000,
    steps_per_day: int = 8,
    trading_days_per_year: int = TRADING_DAYS_PER_YEAR,
    antithetic: bool = True,
) -> TrainingData:
    """Generate surrogate training data by Monte-Carlo simulation.

    Parameters
    ----------
    n_scenarios:
        Number of (parameter, horizon) scenarios to sample and simulate.
    rng_factory:
        Reproducible randomness for both scenario sampling and the per-scenario simulation.
    ranges:
        Sampling ranges; defaults to :class:`ScenarioRanges`.
    quantile_levels:
        Probability grid for the target quantiles; defaults to
        :func:`default_quantile_levels`.
    n_paths:
        Monte-Carlo paths per scenario (more = less label noise, slower).
    steps_per_day:
        Simulation grid resolution per trading day.

    Returns
    -------
    TrainingData
        Features, the quantile grid, the per-scenario target quantiles, and the raw inputs.
    """
    if n_scenarios < 1:
        raise ValidationError("n_scenarios must be >= 1", context={"n_scenarios": n_scenarios})
    check_positive(n_paths, name="n_paths")
    if steps_per_day < 1:
        raise ValidationError(
            "steps_per_day must be >= 1", context={"steps_per_day": steps_per_day}
        )

    rng_ranges = ranges or ScenarioRanges()
    levels = default_quantile_levels() if quantile_levels is None else np.asarray(quantile_levels)
    if levels.ndim != 1 or levels.size < 2:
        raise ValidationError("quantile_levels must be 1-D with length >= 2", context={})

    sampler = rng_factory.generator("surrogate.scenarios")
    h = sampler.uniform(*rng_ranges.hurst, size=n_scenarios)
    eta = sampler.uniform(*rng_ranges.eta, size=n_scenarios)
    rho = sampler.uniform(*rng_ranges.rho, size=n_scenarios)
    xi0 = sampler.uniform(*rng_ranges.xi0, size=n_scenarios)
    horizon_days = sampler.uniform(*rng_ranges.horizon_days, size=n_scenarios)
    horizon_years = horizon_days / trading_days_per_year

    quantiles = np.empty((n_scenarios, levels.size), dtype=np.float64)
    for i in range(n_scenarios):
        params = RBergomiParams(
            hurst=float(h[i]),
            eta=float(eta[i]),
            rho=float(rho[i]),
            forward_variance=ForwardVariance.flat(float(xi0[i])),
        )
        grid = TimeGrid.from_calendar_days(
            calendar_days=float(horizon_days[i]),
            steps_per_day=steps_per_day,
            trading_days_per_year=trading_days_per_year,
        )
        # Per-scenario RNG sub-stream keeps scenarios independent yet reproducible.
        sim = HybridSimulator(
            params,
            rng_factory=RandomFactory(rng_factory.seed + 1 + i),
            antithetic=antithetic,
        )
        paths = sim.simulate(grid=grid, n_paths=n_paths, initial_spot=100.0)
        log_returns = paths.terminal_log_return()
        quantiles[i] = np.quantile(log_returns, levels)

    raw = RawInputs(hurst=h, eta=eta, rho=rho, xi0=xi0, horizon=horizon_years)
    features = build_feature_matrix(raw)
    return TrainingData(
        features=features,
        quantile_levels=levels,
        quantiles=quantiles,
        raw_inputs=raw,
    )
