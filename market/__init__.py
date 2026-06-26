"""Market-maker simulator and synthetic options chain (SPEC §3).

Turns rBergomi model values into a credible, arbitrage-free options market with realistic
spreads, sizes, and fill friction -- the basis of the history backtest and the source of the
spread-friction signal the RL agent learns from.

Public surface:

* Quotes: :class:`Quote`, :class:`QuotedOption`.
* Maker: :class:`AvellanedaStoikovMaker`, :class:`MarketMakerConfig`,
  :class:`ObligationConfig`.
* Alpha-driven noise: :func:`apply_alpha_noise`, :func:`alpha_noise_intensity`
  (the :class:`MarketAlpha`-driven perturbation injected into the AS quotes by the
  helper-critic agent's optimisation).
* Fills: :func:`simulate_fill`, :class:`FillResult`, :class:`FillModelConfig`.
* Chain: :func:`build_synthetic_chain`, :func:`repair_call_curve`, :class:`OptionChain`.
"""

from __future__ import annotations

from .alpha_noise import alpha_noise_intensity, apply_alpha_noise
from .chain import OptionChain, build_synthetic_chain, repair_call_curve
from .execution_sim import FillModelConfig, FillResult, simulate_fill
from .market_maker import (
    AvellanedaStoikovMaker,
    MarketMakerConfig,
    ObligationConfig,
)
from .quotes import Quote, QuotedOption

__all__ = [
    "AvellanedaStoikovMaker",
    "FillModelConfig",
    "FillResult",
    "MarketMakerConfig",
    "ObligationConfig",
    "OptionChain",
    "Quote",
    "QuotedOption",
    "alpha_noise_intensity",
    "apply_alpha_noise",
    "build_synthetic_chain",
    "repair_call_curve",
    "simulate_fill",
] 
