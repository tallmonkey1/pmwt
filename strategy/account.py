r"""Account and open-position state for the strategy layer (SPEC §5, §6).

These immutable-by-update structures track the capital and risk state the strategy and risk
supervisor reason about: equity, realized/unrealized P&L, open condor positions, and the
high-water mark needed for the trailing-drawdown kill-switch (SPEC §4.5). They contain no
trading logic -- they are the audited state of record, deliberately separated from the
decision components so the same account object flows identically through backtest and live
(SPEC §8).
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field, replace

from ..core.errors import ValidationError
from ..core.validation import check_finite, check_positive
from ..pricing.instruments import IronCondor

__all__ = ["Account", "OpenPosition"]


@dataclass(frozen=True, slots=True)
class OpenPosition:
    """An open iron-condor position with the economics needed for management.

    Parameters
    ----------
    position_id:
        Unique identifier (assigned by the order layer).
    condor:
        The iron-condor structure.
    entry_credit:
        Net credit received per condor at entry (per unit underlying, positive).
    quantity:
        Number of condors held (positive; the condor itself is the short structure).
    multiplier:
        Contract multiplier (e.g. 100).
    entry_time:
        Timezone-aware open time.
    entry_spot:
        Underlying spot at entry.
    """

    position_id: str
    condor: IronCondor
    entry_credit: float
    quantity: int
    multiplier: float
    entry_time: _dt.datetime
    entry_spot: float
    symbol: str = "SPX"

    def __post_init__(self) -> None:
        if not self.position_id.strip():
            raise ValidationError("position_id must be non-empty", context={})
        if not self.symbol.strip():
            raise ValidationError("symbol must be non-empty", context={})
        if not isinstance(self.condor, IronCondor):
            raise ValidationError("condor must be an IronCondor", context={})
        check_finite(self.entry_credit, name="entry_credit")
        if self.entry_credit <= 0.0:
            raise ValidationError(
                "entry_credit must be positive (net short premium)",
                context={"entry_credit": self.entry_credit},
            )
        if self.quantity < 1:
            raise ValidationError("quantity must be >= 1", context={"quantity": self.quantity})
        check_positive(self.multiplier, name="multiplier")
        check_positive(self.entry_spot, name="entry_spot")
        if (
            self.entry_time.tzinfo is None
            or self.entry_time.tzinfo.utcoffset(self.entry_time) is None
        ):
            raise ValidationError("entry_time must be timezone-aware", context={})

    @property
    def max_profit(self) -> float:
        """Maximum profit of the position (credit collected across all condors)."""
        return self.entry_credit * self.quantity * self.multiplier

    @property
    def max_loss(self) -> float:
        """Maximum loss: (wider spread width - credit) across all condors (a positive number)."""
        per_condor = self.condor.max_spread_width - self.entry_credit
        return max(0.0, per_condor) * self.quantity * self.multiplier

    @property
    def margin_requirement(self) -> float:
        """Defined-risk margin: the maximum loss is the capital at risk."""
        return self.max_loss


@dataclass(frozen=True, slots=True)
class Account:
    """The account's capital and risk state.

    The account is updated functionally (each mutation returns a new ``Account``) so state
    transitions are explicit and auditable. ``high_water_mark`` is the running peak of total
    equity and underpins the trailing-drawdown kill-switch.
    """

    cash: float
    high_water_mark: float
    open_positions: tuple[OpenPosition, ...] = field(default_factory=tuple)
    realized_pnl: float = 0.0

    def __post_init__(self) -> None:
        check_finite(self.cash, name="cash")
        check_finite(self.high_water_mark, name="high_water_mark")
        check_finite(self.realized_pnl, name="realized_pnl")
        if self.high_water_mark <= 0.0:
            raise ValidationError(
                "high_water_mark must be positive", context={"hwm": self.high_water_mark}
            )

    @classmethod
    def open(cls, *, starting_cash: float) -> Account:
        """Create a fresh account with the given starting capital."""
        check_positive(starting_cash, name="starting_cash")
        return cls(cash=starting_cash, high_water_mark=starting_cash)

    @property
    def total_margin(self) -> float:
        """Total margin reserved across open positions."""
        return sum(p.margin_requirement for p in self.open_positions)

    def equity(self, unrealized_pnl: float = 0.0) -> float:
        """Total account equity = cash + mark-to-market of open positions.

        ``unrealized_pnl`` is supplied by the caller from current marks (the account does
        not price positions itself, keeping pricing concerns out of the state object).
        """
        check_finite(unrealized_pnl, name="unrealized_pnl")
        return self.cash + unrealized_pnl

    def drawdown(self, unrealized_pnl: float = 0.0) -> float:
        """Current trailing drawdown as a fraction of the high-water mark (>= 0)."""
        equity = self.equity(unrealized_pnl)
        if self.high_water_mark <= 0.0:  # pragma: no cover - guarded in __post_init__
            return 0.0
        return max(0.0, (self.high_water_mark - equity) / self.high_water_mark)

    def with_high_water_mark_updated(self, unrealized_pnl: float = 0.0) -> Account:
        """Return a copy whose high-water mark is raised to the current equity if higher."""
        equity = self.equity(unrealized_pnl)
        if equity > self.high_water_mark:
            return replace(self, high_water_mark=equity)
        return self

    def add_position(self, position: OpenPosition, *, premium_received: float) -> Account:
        """Return a new account with the position opened and premium credited to cash."""
        check_finite(premium_received, name="premium_received")
        return replace(
            self,
            cash=self.cash + premium_received,
            open_positions=(*self.open_positions, position),
        )

    def close_position(self, position_id: str, *, realized: float) -> Account:
        """Return a new account with a position closed and its realized P&L booked.

        ``realized`` is the cash impact of closing (e.g. ``-cost_to_close`` plus any
        settlement); the position's premium was already credited at open.
        """
        check_finite(realized, name="realized")
        remaining = tuple(p for p in self.open_positions if p.position_id != position_id)
        if len(remaining) == len(self.open_positions):
            raise ValidationError(
                "position_id not found among open positions", context={"position_id": position_id}
            )
        updated = replace(
            self,
            cash=self.cash + realized,
            open_positions=remaining,
            realized_pnl=self.realized_pnl + realized,
        )
        return updated.with_high_water_mark_updated()

    def position_count(self) -> int:
        """Number of open positions."""
        return len(self.open_positions)
