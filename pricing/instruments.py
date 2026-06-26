r"""Option, iron-condor, and vertical-spread instrument specifications.

These immutable, validated dataclasses are the contracts that pricing, payoff, and strategy
code share. Keeping the instrument definition separate from the pricing logic (Single
Responsibility) means the same ``EuropeanOption`` can be priced analytically, by Monte
Carlo, or marked against a simulated chain without changing its definition.

The iron condor (the engine's mandatory structure, SPEC §1.3) and vertical spreads are
represented explicitly by their legs so that margin, payoff, and risk are computed from
first principles rather than hard-coded.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..core.enums import OptionRight
from ..core.errors import ValidationError
from ..core.validation import check_positive

__all__ = ["EuropeanOption", "IronCondor", "OptionLeg", "BullPutSpread"]


@dataclass(frozen=True, slots=True)
class EuropeanOption:
    """A European option on a single underlying.

    Parameters
    ----------
    strike:
        Strike price ``K > 0``.
    expiry:
        Time to expiry in years ``T > 0`` (measured from the valuation date).
    right:
        Call or put.
    """

    strike: float
    expiry: float
    right: OptionRight

    def __post_init__(self) -> None:
        check_positive(self.strike, name="strike")
        check_positive(self.expiry, name="expiry")
        if not isinstance(self.right, OptionRight):
            raise ValidationError(
                "right must be an OptionRight", context={"type": type(self.right).__name__}
            )

    @property
    def is_call(self) -> bool:
        """True if this is a call option."""
        return self.right is OptionRight.CALL

    @property
    def is_put(self) -> bool:
        """True if this is a put option."""
        return self.right is OptionRight.PUT


@dataclass(frozen=True, slots=True)
class OptionLeg:
    """A signed quantity of a European option (a position in one option).

    ``quantity`` is the number of contracts: positive = long, negative = short. The
    magnitude must be a positive integer count of contracts.
    """

    option: EuropeanOption
    quantity: int

    def __post_init__(self) -> None:
        if not isinstance(self.option, EuropeanOption):
            raise ValidationError(
                "option must be a EuropeanOption", context={"type": type(self.option).__name__}
            )
        if not isinstance(self.quantity, int) or isinstance(self.quantity, bool):
            raise ValidationError("quantity must be an int", context={"quantity": self.quantity})
        if self.quantity == 0:
            raise ValidationError("quantity must be non-zero", context={})

    @property
    def is_long(self) -> bool:
        """True if the leg is long (positive quantity)."""
        return self.quantity > 0

    @property
    def is_short(self) -> bool:
        """True if the leg is short (negative quantity)."""
        return self.quantity < 0


@dataclass(frozen=True, slots=True)
class BullPutSpread:
    """A short bull put spread: sell an OTM put and buy a further OTM put for protection.

    The two strikes satisfy ``long_strike < short_strike``. The structure is *net short
    premium* and *defined risk*.

    Parameters
    ----------
    long_strike:
        The lower, protective put strike (bought).
    short_strike:
        The higher, premium-collecting put strike (sold).
    expiry:
        Shared time to expiry in years.
    quantity:
        Number of spreads (positive integer).
    """

    long_strike: float
    short_strike: float
    expiry: float
    quantity: int = 1

    def __post_init__(self) -> None:
        check_positive(self.long_strike, name="long_strike")
        check_positive(self.short_strike, name="short_strike")
        check_positive(self.expiry, name="expiry")
        if not isinstance(self.quantity, int) or isinstance(self.quantity, bool):
            raise ValidationError("quantity must be an int", context={"quantity": self.quantity})
        if self.quantity <= 0:
            raise ValidationError(
                "quantity must be a positive number of spreads",
                context={"quantity": self.quantity},
            )
        if not (self.long_strike < self.short_strike):
            raise ValidationError(
                "strikes must satisfy long_strike < short_strike",
                context={"long": self.long_strike, "short": self.short_strike},
            )

    @property
    def spread_width(self) -> float:
        """Width of the spread (``short_strike - long_strike``)."""
        return self.short_strike - self.long_strike

    @property
    def max_spread_width(self) -> float:
        """Alias for spread_width for compatibility with condor-style sizing."""
        return self.spread_width

    def legs(self) -> tuple[OptionLeg, OptionLeg]:
        """Return the two signed legs of the bull put spread."""
        q = self.quantity
        return (
            OptionLeg(EuropeanOption(self.long_strike, self.expiry, OptionRight.PUT), q),
            OptionLeg(EuropeanOption(self.short_strike, self.expiry, OptionRight.PUT), -q),
        )


@dataclass(frozen=True, slots=True)
class IronCondor:
    """A short iron condor: sell an OTM put spread and an OTM call spread.

    The four strikes satisfy ``put_long < put_short < call_short < call_long``. The
    structure is *net short premium* and *defined risk*: maximum loss is capped by the
    wider of the two spread widths minus the credit received.

    All four legs share the same expiry and the same per-leg ``quantity`` (number of
    condors). Strikes are validated to enforce the canonical ordering, which guarantees the
    payoff is a well-formed tent shape and the risk is bounded.

    Parameters
    ----------
    put_long_strike:
        Lowest strike; long put (the protective wing of the put spread).
    put_short_strike:
        Short put strike (sold).
    call_short_strike:
        Short call strike (sold).
    call_long_strike:
        Highest strike; long call (the protective wing of the call spread).
    expiry:
        Shared time to expiry in years.
    quantity:
        Number of condors (positive integer). Selling the condor is the default short
        structure; the legs encode the directions.
    """

    put_long_strike: float
    put_short_strike: float
    call_short_strike: float
    call_long_strike: float
    expiry: float
    quantity: int = 1

    def __post_init__(self) -> None:
        for name, value in (
            ("put_long_strike", self.put_long_strike),
            ("put_short_strike", self.put_short_strike),
            ("call_short_strike", self.call_short_strike),
            ("call_long_strike", self.call_long_strike),
        ):
            check_positive(value, name=name)
        check_positive(self.expiry, name="expiry")
        if not isinstance(self.quantity, int) or isinstance(self.quantity, bool):
            raise ValidationError("quantity must be an int", context={"quantity": self.quantity})
        if self.quantity <= 0:
            raise ValidationError(
                "quantity must be a positive number of condors",
                context={"quantity": self.quantity},
            )
        ordered = (
            self.put_long_strike
            < self.put_short_strike
            < self.call_short_strike
            < self.call_long_strike
        )
        if not ordered:
            raise ValidationError(
                "strikes must satisfy put_long < put_short < call_short < call_long",
                context={
                    "put_long": self.put_long_strike,
                    "put_short": self.put_short_strike,
                    "call_short": self.call_short_strike,
                    "call_long": self.call_long_strike,
                },
            )

    @property
    def put_spread_width(self) -> float:
        """Width of the put spread (``put_short - put_long``)."""
        return self.put_short_strike - self.put_long_strike

    @property
    def call_spread_width(self) -> float:
        """Width of the call spread (``call_long - call_short``)."""
        return self.call_long_strike - self.call_short_strike

    @property
    def max_spread_width(self) -> float:
        """The wider of the two spread widths (drives worst-case loss and margin)."""
        return max(self.put_spread_width, self.call_spread_width)

    @property
    def profit_zone(self) -> tuple[float, float]:
        """The ``(lower, upper)`` short strikes between which the structure keeps full credit."""
        return self.put_short_strike, self.call_short_strike

    def legs(self) -> tuple[OptionLeg, OptionLeg, OptionLeg, OptionLeg]:
        """Return the four signed legs of the (short) iron condor.

        Selling the condor means: long the wings (protection) and short the inner strikes
        (premium collection).
        """
        q = self.quantity
        return (
            OptionLeg(EuropeanOption(self.put_long_strike, self.expiry, OptionRight.PUT), q),
            OptionLeg(EuropeanOption(self.put_short_strike, self.expiry, OptionRight.PUT), -q),
            OptionLeg(EuropeanOption(self.call_short_strike, self.expiry, OptionRight.CALL), -q),
            OptionLeg(EuropeanOption(self.call_long_strike, self.expiry, OptionRight.CALL), q),
        )
