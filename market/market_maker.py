r"""Avellaneda-Stoikov market maker bounded by exchange quoting obligations (SPEC §3).

This is the engine that turns a theoretical option value into a realistic two-sided quote.
It combines four effects, each grounded in the microstructure literature and clearly
attributed:

1. **Avellaneda-Stoikov optimal spread** (Avellaneda & Stoikov, 2008). The maker quotes
   around an *inventory-adjusted reservation price* and sets a spread that trades off the
   risk of holding inventory against the rate of capturing the spread:

   .. math::

       r = s - q\,\gamma\,\sigma^2\,(T-t), \qquad
       \delta^{\text{tot}} = \gamma\,\sigma^2\,(T-t)
                           + \frac{2}{\gamma}\ln\!\Big(1 + \frac{\gamma}{\kappa}\Big),

   where ``s`` is the fair value, ``q`` the maker's signed inventory, ``gamma`` the risk
   aversion, ``sigma`` the instantaneous vol of the option value, ``kappa`` the order-flow
   liquidity intensity, and ``T - t`` the risk horizon.

2. **Wing-liquidity decay.** Deep out-of-the-money options are harder to hedge and thinner,
   so the spread is widened and the displayed size shrunk as a function of the option's
   *delta distance from at-the-money*. This is the realistic correction to the misconception
   (SPEC §0) that obligated makers quote the wings tightly.

3. **Exchange obligations.** Designated market makers must honour a *maximum* quoted spread,
   a *minimum* size, and (modelled elsewhere) an uptime fraction. The obligation acts as a
   ceiling on the spread and a floor on size -- it does **not** force tight wing quotes; it
   merely caps how wide/thin the maker may go.

4. **Alpha-driven microstructure noise** (see :mod:`options_engine.core.market_alpha`). The
   theoretical reservation price is a clean two-sided quote; real markets carry additional
   queue / latency / asymmetric-information noise around the centre that the AS formula
   does not model. The maker injects a bounded perturbation whose magnitude is driven by
   the supplied :class:`MarketAlpha`: ``alpha[1] = 1`` means clean AS quotes (no noise),
   ``alpha[1] = 0`` means the maximum bounded perturbation per call. The helper-critic
   agent uses this knob to expose the main PPO-Transformer agent to progressively noisier
   markets as its training improves.

The maker is deterministic given its inputs (quote construction is a pure function);
the alpha-driven noise draws from an injected :class:`numpy.random.Generator` so the
process is fully reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.random import Generator

from ..core.errors import ValidationError
from ..core.market_alpha import MarketAlpha
from ..core.validation import check_non_negative, check_positive
from ..pricing.instruments import EuropeanOption
from .alpha_noise import apply_alpha_noise
from .quotes import Quote, QuotedOption

__all__ = ["AvellanedaStoikovMaker", "MarketMakerConfig", "ObligationConfig"]


@dataclass(frozen=True, slots=True)
class ObligationConfig:
    """Exchange market-maker quoting obligations (caps/floors, not tightness mandates)."""

    #: Maximum quoted spread as a fraction of the option mid (e.g. 0.35 = 35%).
    max_relative_spread: float = 0.35
    #: Absolute maximum quoted spread floor (covers very low-priced options).
    max_absolute_spread: float = 5.0
    #: Minimum displayed size in contracts at each side.
    min_size: int = 1
    #: Absolute minimum spread (one tick), so quotes are never crossed/locked.
    min_absolute_spread: float = 0.01

    def __post_init__(self) -> None:
        check_positive(self.max_relative_spread, name="max_relative_spread")
        check_positive(self.max_absolute_spread, name="max_absolute_spread")
        check_positive(self.min_absolute_spread, name="min_absolute_spread")
        if self.min_size < 1:
            raise ValidationError("min_size must be >= 1", context={"min_size": self.min_size})


@dataclass(frozen=True, slots=True)
class MarketMakerConfig:
    """Parameters of the Avellaneda-Stoikov maker plus wing-liquidity decay."""

    #: Inventory risk aversion ``gamma`` (> 0). Larger -> wider spreads, more skew.
    risk_aversion: float = 1.5
    #: Order-flow liquidity intensity ``kappa`` (> 0). Larger -> tighter base spread.
    order_flow_intensity: float = 1.5
    #: Base displayed size at-the-money, in contracts.
    base_size: int = 50
    #: Strength of wing spread widening: spread is scaled by ``1 + wing_spread_factor * m``
    #: where ``m`` is the ATM delta distance in ``[0, 0.5]``.
    wing_spread_factor: float = 4.0
    #: Strength of wing size decay: size is scaled by ``exp(-wing_size_decay * m)``.
    wing_size_decay: float = 6.0
    #: Floor on the per-contract value-vol used in the AS formula (keeps spreads sane).
    min_value_vol: float = 1e-3

    def __post_init__(self) -> None:
        check_positive(self.risk_aversion, name="risk_aversion")
        check_positive(self.order_flow_intensity, name="order_flow_intensity")
        if self.base_size < 1:
            raise ValidationError("base_size must be >= 1", context={"base_size": self.base_size})
        check_non_negative(self.wing_spread_factor, name="wing_spread_factor")
        check_non_negative(self.wing_size_decay, name="wing_size_decay")
        check_positive(self.min_value_vol, name="min_value_vol")


class AvellanedaStoikovMaker:
    """Produces obligation-bounded two-sided quotes from theoretical option values.

    Parameters
    ----------
    config:
        Maker behavioural parameters.
    obligations:
        Exchange obligation caps/floors.
    tick_size:
        Price increment to which quotes are rounded (e.g. 0.05 for many index options).
    """

    def __init__(
        self,
        *,
        config: MarketMakerConfig | None = None,
        obligations: ObligationConfig | None = None,
        tick_size: float = 0.05,
    ) -> None:
        check_positive(tick_size, name="tick_size")
        self._config = config or MarketMakerConfig()
        self._obligations = obligations or ObligationConfig()
        self._tick_size = tick_size

    @property
    def config(self) -> MarketMakerConfig:
        """The maker behavioural configuration."""
        return self._config

    @property
    def obligations(self) -> ObligationConfig:
        """The exchange obligation configuration."""
        return self._obligations

    def quote(
        self,
        option: EuropeanOption,
        *,
        theoretical_value: float,
        value_volatility: float,
        atm_delta_distance: float,
        inventory: int = 0,
        risk_horizon: float | None = None,
        alpha: MarketAlpha | None = None,
        rng: Generator | None = None,
    ) -> QuotedOption:
        r"""Return an obligation-bounded quote for one option.

        Parameters
        ----------
        option:
            The option being quoted.
        theoretical_value:
            The fair (model) price ``s`` of the option (per unit underlying).
        value_volatility:
            Standard deviation of the option *value* over the risk horizon (not the
            underlying's vol). Drives the AS inventory and spread terms. Typically
            ``|vega| * sigma_iv`` or an empirical option-price vol.
        atm_delta_distance:
            Distance from at-the-money measured in option delta: ``0`` at-the-money,
            approaching ``0.5`` deep OTM. Drives wing widening/thinning.
        inventory:
            The maker's current signed inventory ``q`` in this option (long positive).
        risk_horizon:
            The AS horizon ``T - t`` in years. Defaults to the option's time to expiry.
        alpha:
            Optional :class:`MarketAlpha`. When supplied, the quote is perturbed by the
            bounded alpha-driven noise described in :mod:`.alpha_noise`. When ``None``
            (the default) the quote is the clean AS output -- full backward compatibility.
        rng:
            NumPy RNG for the alpha-driven perturbation. Required when ``alpha`` is
            supplied so the noise is reproducible. Ignored when ``alpha`` is ``None``.

        Returns
        -------
        QuotedOption
            The contract, its bounded quote, and the theoretical value.
        """
        check_non_negative(theoretical_value, name="theoretical_value")
        check_non_negative(value_volatility, name="value_volatility")
        m = float(np.clip(atm_delta_distance, 0.0, 0.5))
        horizon = option.expiry if risk_horizon is None else risk_horizon
        check_positive(horizon, name="risk_horizon")

        cfg = self._config
        sigma = max(value_volatility, cfg.min_value_vol)
        gamma = cfg.risk_aversion
        kappa = cfg.order_flow_intensity

        # Avellaneda-Stoikov reservation price (inventory skew) and base total spread.
        inventory_skew = inventory * gamma * sigma**2 * horizon
        reservation = theoretical_value - inventory_skew
        base_spread = gamma * sigma**2 * horizon + (2.0 / gamma) * np.log1p(gamma / kappa)

        # Wing widening: deep-OTM options carry wider spreads.
        spread = base_spread * (1.0 + cfg.wing_spread_factor * m)

        # Apply obligation caps/floors on the spread.
        spread = self._bound_spread(spread, theoretical_value)

        half = 0.5 * spread
        bid = max(0.0, reservation - half)
        ask = reservation + half
        bid, ask = self._round_to_tick(bid), self._round_to_tick(ask)
        # Guarantee a non-crossed market after rounding.
        if ask <= bid:
            ask = bid + self._tick_size

        # Wing size decay, with the obligation minimum-size floor.
        decayed_size = round(float(cfg.base_size * np.exp(-cfg.wing_size_decay * m)))
        size = max(self._obligations.min_size, decayed_size)

        quote = Quote(bid=bid, ask=ask, bid_size=size, ask_size=size)
        # Optional alpha-driven perturbation. Backward-compatible: alpha=None skips.
        if alpha is not None and rng is not None:
            quote = apply_alpha_noise(quote, alpha=alpha, rng=rng)
        return QuotedOption(option=option, quote=quote, theoretical_value=theoretical_value)

    def _bound_spread(self, spread: float, theoretical_value: float) -> float:
        """Clamp the spread to the exchange obligation band."""
        obl = self._obligations
        max_spread = min(
            obl.max_absolute_spread,
            max(obl.max_relative_spread * theoretical_value, obl.min_absolute_spread),
        )
        return float(np.clip(spread, obl.min_absolute_spread, max_spread))

    def _round_to_tick(self, price: float) -> float:
        """Round a price to the nearest tick (non-negative)."""
        return max(0.0, round(price / self._tick_size) * self._tick_size)
