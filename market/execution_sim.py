r"""Fill simulation against a quoted market: slippage, partial fills, and impact (SPEC §3).

When the strategy sends a marketable order, it does not simply transact at the mid -- it
pays the spread, may walk the book if its size exceeds the displayed quantity, and incurs
price impact. This module models that *friction*, which is precisely what the RL agent must
learn to respect (SPEC §4.4: "the agent learns spread friction").

Model
-----
* **Crossing the spread.** A buy lifts the ask; a sell hits the bid. The first
  ``min(order, displayed_size)`` contracts fill at the touch.
* **Walking the book / impact.** Size beyond the displayed quantity fills at progressively
  worse prices: each additional displayed-size "level" is offset by a linear impact
  increment proportional to the spread. This yields a convex, size-dependent average fill
  price -- the empirically-observed shape of market impact.
* **Partial fills.** Available liquidity is capped at ``max_fill_multiple`` times the
  displayed size (deep books are still finite); an order larger than that fills partially.

The result is a :class:`FillResult` reporting filled quantity, average price, and the
explicit slippage-versus-mid cost, so cost accounting and the RL reward are exact.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..core.enums import OrderSide
from ..core.errors import ValidationError
from ..core.validation import check_non_negative, check_positive
from .quotes import Quote

__all__ = ["FillModelConfig", "FillResult", "simulate_fill"]


@dataclass(frozen=True, slots=True)
class FillModelConfig:
    """Parameters governing fill slippage and available depth."""

    #: Maximum fillable quantity as a multiple of the displayed touch size.
    max_fill_multiple: float = 5.0
    #: Linear impact per displayed-size level, as a fraction of the spread.
    impact_per_level: float = 0.5

    def __post_init__(self) -> None:
        check_positive(self.max_fill_multiple, name="max_fill_multiple")
        check_non_negative(self.impact_per_level, name="impact_per_level")


@dataclass(frozen=True, slots=True)
class FillResult:
    """Outcome of a marketable order against a quote."""

    side: OrderSide
    requested_quantity: int
    filled_quantity: int
    average_price: float
    mid_at_fill: float

    @property
    def is_complete(self) -> bool:
        """True if the entire requested quantity was filled."""
        return self.filled_quantity == self.requested_quantity

    @property
    def slippage_per_contract(self) -> float:
        """Signed slippage versus mid, per contract (positive = unfavourable cost).

        For a buy this is ``average_price - mid`` (paid above mid); for a sell it is
        ``mid - average_price`` (received below mid). Always a cost when positive.
        """
        if self.filled_quantity == 0:
            return 0.0
        if self.side is OrderSide.BUY:
            return self.average_price - self.mid_at_fill
        return self.mid_at_fill - self.average_price

    @property
    def total_slippage_cost(self) -> float:
        """Total slippage cost across the filled quantity (per unit underlying)."""
        return self.slippage_per_contract * self.filled_quantity


def simulate_fill(
    quote: Quote,
    *,
    side: OrderSide,
    quantity: int,
    config: FillModelConfig | None = None,
) -> FillResult:
    r"""Simulate a marketable order against a quote, returning the fill with slippage.

    Parameters
    ----------
    quote:
        The two-sided market to trade against.
    side:
        Buy (lifts the ask) or sell (hits the bid).
    quantity:
        Requested order size in contracts (``>= 1``).
    config:
        Fill-model parameters; defaults are used if omitted.

    Returns
    -------
    FillResult
        Filled quantity (possibly partial), volume-weighted average fill price, and the mid
        at the time of the fill (for slippage accounting).
    """
    if not isinstance(side, OrderSide):
        raise ValidationError("side must be an OrderSide", context={"side": side})
    if not isinstance(quantity, int) or isinstance(quantity, bool) or quantity < 1:
        raise ValidationError("quantity must be a positive int", context={"quantity": quantity})
    cfg = config or FillModelConfig()

    touch_size = quote.ask_size if side is OrderSide.BUY else quote.bid_size
    touch_price = quote.ask if side is OrderSide.BUY else quote.bid
    spread = quote.spread
    mid = quote.mid

    # Available depth is finite: cap at max_fill_multiple of the displayed touch size.
    max_available = int(touch_size * cfg.max_fill_multiple)
    filled = min(quantity, max_available)
    if filled <= 0:  # pragma: no cover - touch_size >= 1 guarantees max_available >= 1
        return FillResult(
            side=side,
            requested_quantity=quantity,
            filled_quantity=0,
            average_price=mid,
            mid_at_fill=mid,
        )

    # Walk the book: the first touch_size contracts fill at the touch; each subsequent
    # level adds linear impact. Compute the volume-weighted average fill price.
    sign = 1.0 if side is OrderSide.BUY else -1.0
    total_cost = 0.0
    remaining = filled
    level = 0
    while remaining > 0:
        level_qty = min(remaining, touch_size)
        level_price = touch_price + sign * spread * cfg.impact_per_level * level
        level_price = max(0.0, level_price)
        total_cost += level_price * level_qty
        remaining -= level_qty
        level += 1

    average_price = total_cost / filled
    return FillResult(
        side=side,
        requested_quantity=quantity,
        filled_quantity=filled,
        average_price=average_price,
        mid_at_fill=mid,
    )
