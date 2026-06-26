r"""Live-trading arming: the typed-confirmation lock on real orders (SPEC §7, SAFETY.md).

Real capital must never be reachable by a default value, a stray boolean, or a single typo.
:class:`LiveTradingArming` requires the operator to pass a **typed confirmation phrase** that
exactly matches :data:`LiveTradingArming.REQUIRED_PHRASE`. A boolean flag alone is
insufficient; the phrase cannot be produced accidentally, which is the point.

This object is the gate the live-broker factory consults. Construct it only at the genuine
entry point of a live run, from an explicit operator action -- never with a default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ..core.errors import ConfigurationError

__all__ = ["LiveTradingArming"]


@dataclass(frozen=True, slots=True)
class LiveTradingArming:
    """A deliberate, typed authorization to trade real capital.

    Parameters
    ----------
    enable_live_trading:
        Must be explicitly ``True``. Defaults to ``False`` so the safe path is the default.
    confirmation_phrase:
        Must exactly equal :data:`REQUIRED_PHRASE`. Any mismatch (including the empty default)
        leaves the system disarmed.
    """

    #: The exact phrase the operator must type to arm live trading. A ``ClassVar`` (a true
    #: class constant, not a dataclass field), so it is never part of construction and cannot
    #: be overridden per instance.
    REQUIRED_PHRASE: ClassVar[str] = "I UNDERSTAND THIS TRADES REAL MONEY"

    enable_live_trading: bool = False
    confirmation_phrase: str = ""

    @property
    def is_armed(self) -> bool:
        """True only if both the boolean and the exact typed phrase are present."""
        return bool(self.enable_live_trading) and self.confirmation_phrase == self.REQUIRED_PHRASE

    def require_armed(self) -> None:
        """Raise :class:`ConfigurationError` unless live trading is fully armed.

        The error message deliberately does not echo the phrase, so it cannot be copy-pasted
        from a log to bypass the deliberate-typing requirement.
        """
        if not self.enable_live_trading:
            raise ConfigurationError(
                "live trading is not enabled (enable_live_trading=False)", context={}
            )
        if self.confirmation_phrase != self.REQUIRED_PHRASE:
            raise ConfigurationError(
                "live trading confirmation phrase is missing or incorrect; "
                "live trading remains disarmed",
                context={},
            )

    @classmethod
    def disarmed(cls) -> LiveTradingArming:
        """Return an explicitly disarmed arming (the safe default)."""
        return cls(enable_live_trading=False, confirmation_phrase="")
