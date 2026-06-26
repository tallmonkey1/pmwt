r"""Multi-filter entry logic for new iron-condor positions (SPEC §5).

The spec is explicit that entry must be stronger than "distribution + regime confidence":
it stacks several independent, redundant filters so a trade is opened only when *all* edges
align. This module composes them into a single, auditable decision:

1. **Regime gate** -- calm now and next step (SPEC §2.6).
2. **News/event gate** -- no scheduled-event blackout or breaking-news cool-off (SPEC §2.7).
3. **Distribution edge** -- a qualifying condor exists with sufficient model win-probability
   and executable credit net of the bid-ask spread (SPEC §5, via condor selection).
4. **Spread-cost filter** -- the credit must clear the spread cost by a configured margin.
5. **Sizing** -- fractional-Kelly size > 0 under the hard caps (SPEC §6).
6. **Risk supervisor** -- the deterministic overlay approves the sized position (SPEC §4.5).

Each filter can independently veto, and the first veto is reported. The result, when
positive, carries the fully-specified, risk-approved position ready for execution.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from ..core.config import RiskConfig
from ..core.errors import ValidationError
from ..core.validation import check_non_negative, check_positive
from ..market.chain import OptionChain
from ..news.gate import NewsGate
from ..pricing.instruments import IronCondor
from ..regime.detector import RegimeNowcast
from ..regime.gate import RegimeGateConfig, evaluate_regime_gate
from .account import Account, OpenPosition
from .condor_selection import (
    CondorCandidate,
    CondorSelectionConfig,
    select_iron_condor,
)
from .risk_supervisor import RiskSupervisor
from .sizing import SizingInputs, size_position

__all__ = ["EntryConfig", "EntryDecision", "EntryEvaluator"]


@dataclass(frozen=True, slots=True)
class EntryConfig:
    """Configuration for the entry evaluator."""

    #: Minimum ratio of net credit to spread cost for the trade to be worthwhile.
    min_credit_to_cost_ratio: float = 2.0
    #: Regime gate thresholds.
    regime_gate: RegimeGateConfig = field(default_factory=RegimeGateConfig)
    #: Condor selection parameters.
    selection: CondorSelectionConfig = field(default_factory=CondorSelectionConfig)

    def __post_init__(self) -> None:
        check_positive(self.min_credit_to_cost_ratio, name="min_credit_to_cost_ratio")


@dataclass(frozen=True, slots=True)
class EntryDecision:
    """The entry evaluator's verdict, with the proposed position if approved."""

    enter: bool
    reason: str
    position: OpenPosition | None = None
    candidate: CondorCandidate | None = None

    @property
    def rejected(self) -> bool:
        """True if no entry is to be made."""
        return not self.enter


class EntryEvaluator:
    """Composes all entry filters into a single, risk-approved decision.

    Parameters
    ----------
    risk:
        Core risk configuration used for sizing and the supervisor.
    risk_supervisor:
        The deterministic risk overlay.
    news_gate:
        The news/event gate.
    config:
        Entry configuration.
    """

    def __init__(
        self,
        *,
        risk: RiskConfig,
        risk_supervisor: RiskSupervisor,
        news_gate: NewsGate,
        config: EntryConfig | None = None,
    ) -> None:
        if not isinstance(risk, RiskConfig):
            raise ValidationError("risk must be a RiskConfig", context={})
        if not isinstance(risk_supervisor, RiskSupervisor):
            raise ValidationError("risk_supervisor must be a RiskSupervisor", context={})
        if not isinstance(news_gate, NewsGate):
            raise ValidationError("news_gate must be a NewsGate", context={})
        self._risk = risk
        self._supervisor = risk_supervisor
        self._news_gate = news_gate
        self._config = config or EntryConfig()

    def evaluate(
        self,
        *,
        now: _dt.datetime,
        symbol: str,
        distribution: object,
        chain: OptionChain,
        regime: RegimeNowcast,
        account: Account,
        position_id: str,
        multiplier: float = 100.0,
        terminal_sample: NDArray[np.float64] | None = None,
        unrealized_pnl: float = 0.0,
        risked_today: float = 0.0,
        available_margin: float,
    ) -> EntryDecision:
        """Run all entry filters in order and return the (possibly approved) decision."""
        if now.tzinfo is None or now.tzinfo.utcoffset(now) is None:
            raise ValidationError("now must be timezone-aware", context={})
        check_positive(multiplier, name="multiplier")
        check_non_negative(available_margin, name="available_margin")

        # 1. Regime gate.
        regime_decision = evaluate_regime_gate(regime, config=self._config.regime_gate)
        if not regime_decision.allow_new_risk:
            return EntryDecision(enter=False, reason=f"regime gate: {regime_decision.reason}")

        # 2. News / event gate.
        news_decision = self._news_gate.evaluate(now, symbol=symbol)
        if not news_decision.allow_new_risk:
            return EntryDecision(enter=False, reason=f"news gate: {news_decision.reason}")

        # 3. Distribution edge -> candidate condor.
        candidate = select_iron_condor(
            distribution,  # type: ignore[arg-type]
            chain,
            config=self._config.selection,
            terminal_sample=terminal_sample,
        )
        if candidate is None:
            return EntryDecision(enter=False, reason="no qualifying condor from distribution")

        # 4. Spread-cost filter.
        if candidate.spread_cost > 0.0:
            ratio = candidate.net_credit / candidate.spread_cost
            if ratio < self._config.min_credit_to_cost_ratio:
                return EntryDecision(
                    enter=False,
                    reason=(
                        f"credit/cost ratio {ratio:.2f} < "
                        f"{self._config.min_credit_to_cost_ratio:.2f}"
                    ),
                    candidate=candidate,
                )

        # 5. Sizing under fractional Kelly + hard caps.
        max_loss_per_condor = candidate.condor.max_spread_width - candidate.net_credit
        if max_loss_per_condor <= 0.0:
            return EntryDecision(
                enter=False, reason="degenerate condor: credit >= width", candidate=candidate
            )
        sizing = size_position(
            SizingInputs(
                account_equity=account.equity(unrealized_pnl),
                win_probability=candidate.win_probability,
                net_credit=candidate.net_credit,
                max_loss_per_condor=max_loss_per_condor,
                multiplier=multiplier,
                available_margin=available_margin,
                risked_today=risked_today,
            ),
            risk=self._risk,
        )
        if sizing.quantity < 1:
            return EntryDecision(
                enter=False,
                reason=f"sizing yielded 0 contracts (binding: {sizing.binding_constraint})",
                candidate=candidate,
            )

        # 6. Build the proposed position and clear it with the risk supervisor.
        position = OpenPosition(
            position_id=position_id,
            condor=_condor_with_quantity(candidate.condor, sizing.quantity),
            entry_credit=candidate.net_credit,
            quantity=sizing.quantity,
            multiplier=multiplier,
            entry_time=now,
            entry_spot=chain.spot,
        )
        approval = self._supervisor.approve_new_position(
            account,
            position,
            unrealized_pnl=unrealized_pnl,
            risked_today=risked_today,
            available_margin=available_margin,
        )
        if not approval.approved:
            return EntryDecision(
                enter=False, reason=f"risk supervisor: {approval.reason}", candidate=candidate
            )

        return EntryDecision(
            enter=True,
            reason="all entry filters passed",
            position=position,
            candidate=candidate,
        )


def _condor_with_quantity(condor: IronCondor, quantity: int) -> IronCondor:
    """Return a copy of an IronCondor with the given quantity."""
    return IronCondor(
        put_long_strike=condor.put_long_strike,
        put_short_strike=condor.put_short_strike,
        call_short_strike=condor.call_short_strike,
        call_long_strike=condor.call_long_strike,
        expiry=condor.expiry,
        quantity=quantity,
    )
