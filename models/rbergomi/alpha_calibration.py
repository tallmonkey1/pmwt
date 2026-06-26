"""Alpha-driven rBergomi parameter construction.

A single helper that turns a :class:`MarketAlpha` into a fully-specified
:class:`RBergomiParams` (hurst, eta, rho, forward variance, rate). Every component
of the alpha vector contributes to a different part of the parameter set so the
helper-critic's optimisation over alpha has a clean, monotonic mapping to the
rBergomi parameters that drive simulation.
"""

from __future__ import annotations

from ...core.market_alpha import (
    MarketAlpha,
    alpha_components,
    alpha_to_drift_noise,
    alpha_to_eta,
    alpha_to_hurst,
    alpha_to_jump_intensity,
    alpha_to_jump_size,
    alpha_to_shock_intensity,
    alpha_to_stoikov_noise,
)
from .params import ForwardVariance, RBergomiParams

__all__ = ["build_rbergomi_params_from_alpha", "alpha_diagnostics"]


def build_rbergomi_params_from_alpha(
    alpha: MarketAlpha,
    *,
    rho: float = -0.7,
    realized_variance_level: float = 0.04,
    rate: float = 0.0,
) -> RBergomiParams:
    """Build an :class:`RBergomiParams` from a :class:`MarketAlpha`.

    Parameters
    ----------
    alpha:
        The market calmness vector. Each component is in ``[0, 1]``.
    rho:
        Spot-vol correlation (fixed; alpha does not modulate it because the
        leverage effect is empirically stable across regimes). Defaults to ``-0.7``.
    realized_variance_level:
        The flat forward-variance level used when alpha is fully calm.
    rate:
        Continuously-compounded risk-free rate.
    """
    if not isinstance(alpha, MarketAlpha):
        raise TypeError(f"alpha must be a MarketAlpha, got {type(alpha).__name__}")
    hurst = alpha_to_hurst(alpha)
    eta = alpha_to_eta(alpha)
    return RBergomiParams(
        hurst=hurst,
        eta=eta,
        rho=rho,
        forward_variance=ForwardVariance.flat(realized_variance_level),
        rate=rate,
    )


def alpha_diagnostics(alpha: MarketAlpha) -> dict[str, float]:
    """Return the alpha-to-model mapping for the current alpha as a flat dict."""
    comps = alpha_components(alpha)
    return {
        **{f"alpha.{name}": value for name, value in comps.items()},
        "model.hurst": alpha_to_hurst(alpha),
        "model.eta": alpha_to_eta(alpha),
        "model.stoikov_noise": alpha_to_stoikov_noise(alpha),
        "model.drift_noise": alpha_to_drift_noise(alpha),
        "model.jump_intensity": alpha_to_jump_intensity(alpha),
        "model.jump_size": alpha_to_jump_size(alpha),
        "model.shock_intensity": alpha_to_shock_intensity(alpha),
    }
