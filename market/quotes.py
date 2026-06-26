r"""Quote data structures for the market-maker simulator (SPEC §3).

A :class:`Quote` is a validated two-sided market (bid/ask with sizes) for a single
instrument. :class:`QuotedOption` attaches a quote to a concrete option contract and a
theoretical (model) value, which is what downstream cost accounting and the RL environment
consume. Keeping these immutable and self-validating means a malformed quote (crossed
market, non-positive size, negative price) can never silently enter a backtest.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..core.errors import ValidationError
from ..core.validation import check_non_negative, check_positive
from ..pricing.instruments import EuropeanOption

__all__ = ["Quote", "QuotedOption"]


@dataclass(frozen=True, slots=True)
class Quote:
    """A two-sided quote: bid/ask prices with their displayed sizes.

    Parameters
    ----------
    bid:
        Best bid price (``>= 0``).
    ask:
        Best ask price (``> bid``).
    bid_size:
        Displayed size at the bid, in contracts (``>= 1``).
    ask_size:
        Displayed size at the ask, in contracts (``>= 1``).
    """

    bid: float
    ask: float
    bid_size: int
    ask_size: int

    def __post_init__(self) -> None:
        check_non_negative(self.bid, name="bid")
        check_positive(self.ask, name="ask")
        if self.ask <= self.bid:
            raise ValidationError(
                "ask must be strictly greater than bid (no crossed/locked market)",
                context={"bid": self.bid, "ask": self.ask},
            )
        for name, size in (("bid_size", self.bid_size), ("ask_size", self.ask_size)):
            if not isinstance(size, int) or isinstance(size, bool):
                raise ValidationError(f"{name} must be an int", context={name: size})
            if size < 1:
                raise ValidationError(f"{name} must be >= 1", context={name: size})

    @property
    def mid(self) -> float:
        """The mid price ``(bid + ask) / 2``."""
        return 0.5 * (self.bid + self.ask)

    @property
    def spread(self) -> float:
        """The absolute bid-ask spread."""
        return self.ask - self.bid

    @property
    def relative_spread(self) -> float:
        """The spread as a fraction of the mid (``inf`` if mid is 0)."""
        mid = self.mid
        return self.spread / mid if mid > 0.0 else float("inf")

    @property
    def half_spread(self) -> float:
        """Half the bid-ask spread."""
        return 0.5 * self.spread


@dataclass(frozen=True, slots=True)
class QuotedOption:
    """An option contract together with its simulated quote and theoretical value.

    ``theoretical_value`` is the model mid (e.g. from Monte-Carlo / Black-Scholes); the
    difference between it and the quote mid is the market maker's skew/edge, which the RL
    agent must learn to pay (SPEC §4.4).
    """

    option: EuropeanOption
    quote: Quote
    theoretical_value: float

    def __post_init__(self) -> None:
        if not isinstance(self.option, EuropeanOption):
            raise ValidationError(
                "option must be a EuropeanOption", context={"type": type(self.option).__name__}
            )
        if not isinstance(self.quote, Quote):
            raise ValidationError(
                "quote must be a Quote", context={"type": type(self.quote).__name__}
            )
        check_non_negative(self.theoretical_value, name="theoretical_value")

    @property
    def edge_to_mid(self) -> float:
        """Quote mid minus theoretical value (the maker's mid-price skew)."""
        return self.quote.mid - self.theoretical_value
