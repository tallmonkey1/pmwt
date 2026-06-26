r"""Synthetic options-chain construction with no-arbitrage repair (SPEC §3).

The history-backtest needs a full options chain that does not exist in the historical
underlying data, so we *synthesize* one: price a grid of strikes from the rBergomi terminal
distribution (the same model used everywhere), then attach realistic two-sided quotes from
the market-maker simulator. Because Monte-Carlo prices carry sampling noise, the raw
theoretical values can violate static no-arbitrage constraints (call prices must be
non-increasing and convex in strike); we *repair* the value curve before quoting so the
synthetic chain is internally arbitrage-free -- a hard correctness requirement, since an
arbitrageable chain would let the strategy "win" on simulator artefacts.

No-arbitrage constraints enforced on the call-price curve ``C(K)``:

* **Monotonicity:** ``C(K)`` is non-increasing in ``K`` (and bounded in ``[(S-K e^{-rT})^+,
  S]``).
* **Convexity:** ``C(K)`` is convex in ``K`` (butterfly spreads have non-negative value).

Repair is performed by projecting the noisy call curve onto the convex, monotone, bounded
set; put prices are then derived by put-call parity, guaranteeing cross-consistency.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

from ..core.enums import OptionRight
from ..core.errors import ValidationError
from ..core.market_alpha import MarketAlpha
from ..core.validation import check_array_finite, check_positive
from ..models.rbergomi.results import TerminalDistribution
from ..pricing import black_scholes as bs
from ..pricing.instruments import EuropeanOption
from .market_maker import AvellanedaStoikovMaker
from .quotes import QuotedOption

__all__ = ["OptionChain", "build_synthetic_chain", "repair_call_curve"]


@dataclass(frozen=True, slots=True)
class OptionChain:
    """A synthetic options chain: quoted calls and puts across a strike grid.

    Attributes
    ----------
    spot:
        Underlying spot used to build the chain.
    expiry:
        Shared time to expiry in years.
    strikes:
        Sorted strike grid.
    calls, puts:
        Quoted options keyed by strike (one per strike).
    """

    spot: float
    expiry: float
    strikes: NDArray[np.float64]
    calls: dict[float, QuotedOption]
    puts: dict[float, QuotedOption]

    def call(self, strike: float) -> QuotedOption:
        """Return the quoted call at a strike (must be on the grid)."""
        return self._lookup(self.calls, strike)

    def put(self, strike: float) -> QuotedOption:
        """Return the quoted put at a strike (must be on the grid)."""
        return self._lookup(self.puts, strike)

    @staticmethod
    def _lookup(book: dict[float, QuotedOption], strike: float) -> QuotedOption:
        # Match on the nearest stored strike within a tight tolerance (float keys).
        for k, quoted in book.items():
            if abs(k - strike) < 1e-6:
                return quoted
        raise ValidationError("strike not present in chain", context={"strike": strike})


def repair_call_curve(
    strikes: NDArray[np.float64],
    call_values: NDArray[np.float64],
    *,
    spot: float,
    discount: float,
) -> NDArray[np.float64]:
    r"""Project a noisy call-price curve onto the no-arbitrage feasible set.

    Enforces, in order: upper/lower bounds, monotonic non-increasing in strike, and
    convexity in strike. The convex projection uses the greatest convex minorant via a
    monotone-slope pass, which is the standard isotonic-style repair for option curves.

    Parameters
    ----------
    strikes:
        Sorted, strictly increasing strikes.
    call_values:
        Raw (possibly noisy) call prices at each strike.
    spot:
        Underlying spot ``S``.
    discount:
        Discount factor ``e^{-rT}`` so the lower bound is ``(S - K * discount)^+``.

    Returns
    -------
    numpy.ndarray
        The repaired, arbitrage-free call prices.
    """
    k = np.asarray(strikes, dtype=np.float64)
    c = np.asarray(call_values, dtype=np.float64).copy()
    check_array_finite(k, name="strikes")
    check_array_finite(c, name="call_values")
    if k.ndim != 1 or k.size < 2 or k.shape != c.shape:
        raise ValidationError("strikes/call_values must be 1-D of equal length >= 2", context={})
    if not np.all(np.diff(k) > 0.0):
        raise ValidationError("strikes must be strictly increasing", context={})

    # 1. Bounds: lower = intrinsic forward value, upper = spot.
    lower = np.maximum(spot - k * discount, 0.0)
    c = np.clip(c, lower, spot)

    # 2. Convexity via the convex minorant of the points, then re-impose bounds &
    #    monotonicity. We iterate the convex-hull (lower envelope) pass until the slopes
    #    are non-decreasing, which is the discrete convexity condition.
    c = _convex_minorant(k, c)
    c = np.clip(c, lower, spot)

    # 3. Monotonic non-increasing in strike (calls cheapen as strike rises).
    c = np.minimum.accumulate(c)
    # Re-clip to the lower bound after the monotone pass (lower bound is itself decreasing).
    c = np.maximum(c, lower)
    return np.asarray(c, dtype=np.float64)


def _convex_minorant(x: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the greatest convex minorant of points ``(x, y)`` (lower convex envelope)."""
    n = x.size
    # Build the lower convex hull via a monotone-chain pass over the points.
    hull: list[int] = []
    for i in range(n):
        while len(hull) >= 2:
            j, k = hull[-2], hull[-1]
            # Cross-product test for a convex (counter-clockwise) turn on the lower hull.
            cross = (x[k] - x[j]) * (y[i] - y[j]) - (y[k] - y[j]) * (x[i] - x[j])
            if cross <= 0:
                hull.pop()
            else:
                break
        hull.append(i)
    # Interpolate the hull vertices back onto the full strike grid.
    hull_x = x[hull]
    hull_y = y[hull]
    return np.asarray(np.interp(x, hull_x, hull_y), dtype=np.float64)


def build_synthetic_chain(
    distribution: TerminalDistribution,
    *,
    maker: AvellanedaStoikovMaker,
    strikes: NDArray[np.float64],
    rate: float = 0.0,
    iv_for_value_vol: float = 0.3,
    alpha: MarketAlpha | None = None,
    rng: Generator | None = None,
) -> OptionChain:
    r"""Build a fully-quoted, arbitrage-free synthetic chain from a terminal distribution.

    Parameters
    ----------
    distribution:
        Monte-Carlo terminal distribution of the underlying (from the rBergomi simulator).
    maker:
        The market-maker simulator used to quote each strike.
    strikes:
        Strike grid (sorted, strictly increasing, strictly positive).
    rate:
        Continuously-compounded discount rate over the horizon.
    iv_for_value_vol:
        Reference implied vol used to translate option vega into the value-volatility the
        maker needs. A single reference keeps the chain construction self-contained; the
        per-strike vega still differentiates wing vs. ATM spreads.
    alpha:
        Optional :class:`MarketAlpha`. When supplied (with ``rng``), every quote in the
        chain is perturbed by the bounded alpha-driven noise described in
        :mod:`.alpha_noise`. When ``None`` (the default) the chain is the clean AS output
        -- full backward compatibility.
    rng:
        NumPy RNG for the alpha-driven perturbation; required when ``alpha`` is supplied.

    Returns
    -------
    OptionChain
        Quoted calls and puts across the grid, internally arbitrage-free.
    """
    k = np.asarray(strikes, dtype=np.float64)
    check_array_finite(k, name="strikes")
    if k.ndim != 1 or k.size < 2:
        raise ValidationError("strikes must be 1-D with length >= 2", context={})
    if np.any(k <= 0.0) or not np.all(np.diff(k) > 0.0):
        raise ValidationError("strikes must be strictly increasing and positive", context={})
    check_positive(iv_for_value_vol, name="iv_for_value_vol")

    spot = distribution.initial_spot
    expiry = distribution.horizon
    discount = float(np.exp(-rate * expiry))
    terminal_spot = distribution.terminal_spot()

    # 1. Raw Monte-Carlo call values (discounted expected payoff) per strike.
    raw_calls = np.array(
        [discount * float(np.mean(np.maximum(terminal_spot - strike, 0.0))) for strike in k]
    )

    # 2. No-arbitrage repair on the call curve.
    repaired_calls = repair_call_curve(k, raw_calls, spot=spot, discount=discount)

    # 3. Put values via put-call parity: P = C - S + K e^{-rT}.
    repaired_puts = repaired_calls - spot + k * discount
    repaired_puts = np.maximum(repaired_puts, 0.0)

    # 4. Quote each strike. The value-vol passed to the maker is |vega| * iv, and the ATM
    #    delta distance shapes wing spreads/sizes.
    calls: dict[float, QuotedOption] = {}
    puts: dict[float, QuotedOption] = {}
    for i, strike in enumerate(k):
        strike_f = float(strike)
        call_opt = EuropeanOption(strike=strike_f, expiry=expiry, right=OptionRight.CALL)
        put_opt = EuropeanOption(strike=strike_f, expiry=expiry, right=OptionRight.PUT)

        greeks = bs.greeks(spot, strike_f, expiry, iv_for_value_vol, OptionRight.CALL, rate=rate)
        vega = float(abs(greeks.vega[0]))
        call_delta = float(
            bs.greeks(spot, strike_f, expiry, iv_for_value_vol, OptionRight.CALL, rate=rate).delta[
                0
            ]
        )
        atm_distance = abs(abs(call_delta) - 0.5)
        value_vol = max(vega * iv_for_value_vol, 1e-3)

        calls[strike_f] = maker.quote(
            call_opt,
            theoretical_value=float(repaired_calls[i]),
            value_volatility=value_vol,
            atm_delta_distance=atm_distance,
            alpha=alpha,
            rng=rng,
        )
        puts[strike_f] = maker.quote(
            put_opt,
            theoretical_value=float(repaired_puts[i]),
            value_volatility=value_vol,
            atm_delta_distance=atm_distance,
            alpha=alpha,
            rng=rng,
        )

    return OptionChain(
        spot=spot,
        expiry=expiry,
        strikes=k.copy(),
        calls=calls,
        puts=puts,
    )
