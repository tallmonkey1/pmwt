r"""Episode scenario generation for the RL environment (SPEC §4.2, §4.4).

The RL environment learns inside a *simulated prop-firm world*: a sequence of market steps,
each carrying everything the agent and strategy need (spot, a forward terminal distribution,
a quoted chain, and a regime nowcast). Pre-generating an episode keeps each ``step`` cheap
and fully reproducible, and lets us apply **domain randomization** across episodes -- varying
the rough-vol parameters, the variance-risk-premium, and the market-maker regime -- so the
agent cannot overfit a single market and does not collapse to a degenerate policy
(SPEC §4.4: anti-degeneracy).

The ``alpha`` parameter (see :class:`MarketAlpha`) controls the entire market's character
for the episode: at ``alpha = ones()`` the episode is a smooth, noise-free, jump-free
near-Black-Scholes world (the easiest the helper-critic can produce); at
``alpha = zeros()`` the episode is maximally rough and noisy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..core.enums import VolRegime
from ..core.errors import ValidationError
from ..core.market_alpha import MarketAlpha, alpha_to_hurst, alpha_to_eta
from ..core.random import RandomFactory
from ..core.timegrid import TimeGrid
from ..market.chain import OptionChain, build_synthetic_chain
from ..market.market_maker import AvellanedaStoikovMaker, MarketMakerConfig
from ..models.rbergomi import (
    ForwardVariance,
    HybridSimulator,
    RBergomiParams,
    build_terminal_distribution,
)
from ..models.rbergomi.results import TerminalDistribution
from ..regime.detector import RegimeNowcast

__all__ = ["Episode", "ScenarioConfig", "ScenarioStep", "generate_episode"]


@dataclass(frozen=True, slots=True)
class ScenarioConfig:
    """Domain-randomization ranges and fixed settings for episode generation."""

    n_steps: int = 30
    horizon_days: float = 14.0
    steps_per_day: int = 8
    n_paths: int = 8_000
    realized_variance_range: tuple[float, float] = (0.02, 0.08)
    vrp_multiplier_range: tuple[float, float] = (1.2, 2.5)
    strike_step: float = 0.5
    strike_halfwidth_fraction: float = 0.35
    is_mfd: bool = False

    def __post_init__(self) -> None:
        if self.n_steps < 1:
            raise ValidationError("n_steps must be >= 1", context={"n_steps": self.n_steps})
        if self.steps_per_day < 1 or self.n_paths < 1000:
            raise ValidationError("invalid steps_per_day / n_paths", context={})
        for name, rng in (
            ("realized_variance_range", self.realized_variance_range),
            ("vrp_multiplier_range", self.vrp_multiplier_range),
        ):
            lo, hi = rng
            if not (0.0 < lo < hi):
                raise ValidationError(f"{name} must satisfy 0 < lo < hi", context={"range": rng})
        if self.vrp_multiplier_range[0] < 1.0:
            raise ValidationError(
                "vrp multiplier must be >= 1 (pricing richer than realized)", context={}
            )


@dataclass(frozen=True, slots=True)
class ScenarioStep:
    """A single causal market snapshot the environment serves to the agent."""

    spot: float
    realized_distribution: TerminalDistribution
    realized_terminal_sample: NDArray[np.float64]
    chain: OptionChain
    regime: RegimeNowcast
    atm_relative_spread: float


@dataclass(frozen=True, slots=True)
class Episode:
    """A reproducible sequence of market steps plus the realized vol regime label."""

    steps: tuple[ScenarioStep, ...]
    realized_variance: float
    vrp_multiplier: float
    alpha: MarketAlpha = MarketAlpha.ones()

    def __len__(self) -> int:
        return len(self.steps)


def _regime_for_variance(realized_variance: float, config: ScenarioConfig) -> RegimeNowcast:
    lo, hi = config.realized_variance_range
    frac = float(np.clip((realized_variance - lo) / (hi - lo), 0.0, 1.0))
    low_prob = float(np.clip(0.9 - 0.8 * frac, 0.05, 0.95))
    high_prob = float(np.clip(0.05 + 0.7 * frac, 0.02, 0.9))
    mid_prob = max(0.0, 1.0 - low_prob - high_prob)
    total = low_prob + mid_prob + high_prob
    current = {
        VolRegime.LOW: low_prob / total,
        VolRegime.MID: mid_prob / total,
        VolRegime.HIGH: high_prob / total,
    }
    return RegimeNowcast(current_probabilities=current, next_probabilities=dict(current))


def generate_episode(
    *,
    rng_factory: RandomFactory,
    config: ScenarioConfig | None = None,
    episode_index: int = 0,
    alpha: MarketAlpha | None = None,
) -> Episode:
    """Generate a single reproducible episode.

    Parameters
    ----------
    rng_factory:
        Reproducible randomness for the episode.
    config:
        Scenario configuration (defaults used if ``None``).
    episode_index:
        Used to seed independent sub-streams per episode.
    alpha:
        Optional :class:`MarketAlpha`. When supplied, it overrides the episode's Hurst /
        eta with the alpha-derived values and enables alpha-driven noise on the AS
        quotes. When ``None`` (default), the episode uses the legacy random Hurst / eta
        sampling for full backward compatibility.
    """
    cfg = config or ScenarioConfig()
    sampler = rng_factory.generator(f"rl.episode.{episode_index}")

    horizon_days = 0.1 if cfg.is_mfd else cfg.horizon_days

    realized_var = float(sampler.uniform(*cfg.realized_variance_range))
    vrp = float(sampler.uniform(*cfg.vrp_multiplier_range))
    pricing_var = realized_var * vrp

    if alpha is not None:
        hurst = alpha_to_hurst(alpha)
        eta = alpha_to_eta(alpha)
    else:
        hurst = float(sampler.uniform(0.07, 0.18))
        eta = float(sampler.uniform(1.0, 2.0))
    rho = float(sampler.uniform(-0.85, -0.45))

    grid = TimeGrid.from_calendar_days(
        calendar_days=horizon_days, steps_per_day=cfg.steps_per_day
    )
    maker = AvellanedaStoikovMaker(
        config=MarketMakerConfig(
            risk_aversion=0.08, order_flow_intensity=15.0, wing_spread_factor=1.0
        ),
        tick_size=0.05,
    )
    regime = _regime_for_variance(realized_var, cfg)

    # Alpha-driven AS noise: a per-episode RNG seeded from the master RNG so the
    # perturbation is reproducible for a given (master seed, alpha) pair.
    if alpha is not None:
        episode_noise_rng = np.random.default_rng(rng_factory.seed + 99_001 + episode_index)
    else:
        episode_noise_rng = None

    steps: list[ScenarioStep] = []
    spot = 100.0
    for t in range(cfg.n_steps):
        realized_params = RBergomiParams(
            hurst=hurst, eta=eta, rho=rho, forward_variance=ForwardVariance.flat(realized_var)
        )
        pricing_params = RBergomiParams(
            hurst=hurst, eta=eta, rho=rho, forward_variance=ForwardVariance.flat(pricing_var)
        )
        realized_paths = HybridSimulator(
            realized_params,
            rng_factory=RandomFactory(rng_factory.seed + 1 + episode_index * 10_000 + t),
            antithetic=True,
        ).simulate(grid=grid, n_paths=cfg.n_paths, initial_spot=spot)
        pricing_paths = HybridSimulator(
            pricing_params,
            rng_factory=RandomFactory(rng_factory.seed + 5_000 + episode_index * 10_000 + t),
            antithetic=True,
        ).simulate(grid=grid, n_paths=cfg.n_paths, initial_spot=spot)

        realized_dist = build_terminal_distribution(realized_paths)
        pricing_dist = build_terminal_distribution(pricing_paths)

        halfwidth = cfg.strike_halfwidth_fraction * spot
        strikes = np.arange(
            spot - halfwidth, spot + halfwidth + cfg.strike_step, cfg.strike_step
        ).astype(np.float64)
        chain = build_synthetic_chain(
            pricing_dist,
            maker=maker,
            strikes=strikes,
            rate=0.0,
            alpha=alpha,
            rng=episode_noise_rng,
        )

        atm_quote = _atm_relative_spread(chain, spot)
        steps.append(
            ScenarioStep(
                spot=spot,
                realized_distribution=realized_dist,
                realized_terminal_sample=realized_paths.terminal_spot(),
                chain=chain,
                regime=regime,
                atm_relative_spread=atm_quote,
            )
        )

        scale = np.sqrt(1.0 / horizon_days) if horizon_days > 0 else 1.0
        step_return = float(realized_paths.terminal_log_return()[0]) * scale
        spot = float(spot * np.exp(step_return))

    return Episode(
        steps=tuple(steps),
        realized_variance=realized_var,
        vrp_multiplier=vrp,
        alpha=alpha if alpha is not None else MarketAlpha.ones(),
    )


def _atm_relative_spread(chain: OptionChain, spot: float) -> float:
    nearest = float(min(chain.strikes, key=lambda k: abs(k - spot)))
    return float(chain.call(nearest).quote.relative_spread)
