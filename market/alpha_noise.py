"""Alpha-driven noise injection for the Avellaneda-Stoikov market maker.

The theoretical reservation price from the AS formula is a clean two-sided quote;
real markets have additional microstructure noise around the centre that the
formula does not model. This module injects a *bounded, alpha-driven* perturbation
so the simulator exposes more or less of that noise depending on the
:class:`MarketAlpha` chosen by the helper critic.

Design
------
* **Bounded**: every component is clipped so the perturbed quote never crosses
  (bid < ask) and prices are never negative.
* **Asymmetric**: bids are perturbed down and asks are perturbed up by
  independent uniform factors, which widens the spread.
* **Reproducible**: the perturbation is seeded by an injected :class:`numpy.random.Generator`.
"""

from __future__ import annotations

from numpy.random import Generator

from ..core.errors import ValidationError
from ..core.market_alpha import MarketAlpha, alpha_to_stoikov_noise
from .quotes import Quote

__all__ = ["apply_alpha_noise", "alpha_noise_intensity"]


def alpha_noise_intensity(alpha: MarketAlpha) -> float:
    """Return the current maximum per-side noise intensity for ``alpha`` (in ``[0, 1]``)."""
    return float(alpha_to_stoikov_noise(alpha))


def apply_alpha_noise(quote: Quote, *, alpha: MarketAlpha, rng: Generator) -> Quote:
    """Return a perturbed copy of ``quote`` whose noise is driven by ``alpha``."""
    if not isinstance(quote, Quote):
        raise ValidationError("quote must be a Quote", context={"type": type(quote).__name__})
    if not isinstance(alpha, MarketAlpha):
        raise ValidationError(
            "alpha must be a MarketAlpha", context={"type": type(alpha).__name__}
        )

    noise = alpha_to_stoikov_noise(alpha)
    if noise <= 0.0:
        return Quote(
            bid=quote.bid,
            ask=quote.ask,
            bid_size=quote.bid_size,
            ask_size=quote.ask_size,
        )

    bid_drag = float(rng.uniform(0.0, noise))
    ask_drag = float(rng.uniform(0.0, noise))
    new_bid = max(0.0, quote.bid * (1.0 - bid_drag))
    new_ask = max(new_bid + 1e-4, quote.ask * (1.0 + ask_drag))
    return Quote(
        bid=new_bid,
        ask=new_ask,
        bid_size=quote.bid_size,
        ask_size=quote.ask_size,
    )
