r"""Position selection from the terminal distribution (SPEC §5).

Given a terminal log-return distribution (Monte-Carlo or surrogate) and a quoted chain, this
module constructs candidate iron condors or vertical spreads and scores them on a
risk-adjusted objective, then returns the best. The construction follows the spec: short
strikes are placed at target *distribution* probabilities, wings are a configurable width
out, and the candidate is scored by

    score = expected_pnl - lambda_cvar * CVaR - lambda_cost * spread_cost

over the distribution, evaluated against the *executable* credit net of the bid-ask spread.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from ..core.errors import ValidationError
from ..core.validation import check_non_negative, check_positive, check_unit_interval
from ..market.chain import OptionChain
from ..pricing.instruments import IronCondor, BullPutSpread
from ..pricing.payoff import iron_condor_payoff, bull_put_spread_payoff

__all__ = [
    "CondorCandidate", 
    "BullPutSpreadCandidate", 
    "CondorSelectionConfig", 
    "select_iron_condor", 
    "select_bull_put_spread"
]


@runtime_checkable
class _Distribution(Protocol):
    """Minimal terminal-distribution interface used by selection (MC or surrogate)."""

    @property
    def initial_spot(self) -> float: ...

    def quantile(self, q: float | NDArray[np.float64]) -> NDArray[np.float64]: ...


@dataclass(frozen=True, slots=True)
class CondorSelectionConfig:
    """Parameters governing position construction and scoring."""

    target_tail_probability: float = 0.15
    wing_width_fraction: float = 0.05
    cvar_weight: float = 1.0
    cost_weight: float = 1.0
    cvar_alpha: float = 0.05
    min_win_probability: float = 0.60
    min_net_credit: float = 0.05

    def __post_init__(self) -> None:
        check_unit_interval(
            self.target_tail_probability, name="target_tail_probability", inclusive=False
        )
        check_positive(self.wing_width_fraction, name="wing_width_fraction")
        check_non_negative(self.cvar_weight, name="cvar_weight")
        check_non_negative(self.cost_weight, name="cost_weight")
        check_unit_interval(self.cvar_alpha, name="cvar_alpha", inclusive=False)
        check_unit_interval(self.min_win_probability, name="min_win_probability")
        check_non_negative(self.min_net_credit, name="min_net_credit")


@dataclass(frozen=True, slots=True)
class CondorCandidate:
    """A scored candidate iron condor with its model economics."""

    condor: IronCondor
    net_credit: float
    win_probability: float
    expected_pnl: float
    cvar: float
    spread_cost: float
    score: float


@dataclass(frozen=True, slots=True)
class BullPutSpreadCandidate:
    """A scored candidate bull put spread with its model economics."""

    spread: BullPutSpread
    net_credit: float
    win_probability: float
    expected_pnl: float
    cvar: float
    spread_cost: float
    score: float


def _nearest_strike(strikes: NDArray[np.float64], target: float) -> float:
    """Return the chain strike nearest to a target price."""
    idx = int(np.argmin(np.abs(strikes - target)))
    return float(strikes[idx])


def _resolve_terminal_sample(
    distribution: _Distribution,
    terminal_sample: NDArray[np.float64] | None,
    spot: float,
) -> NDArray[np.float64]:
    """Return a terminal-spot sample for expectation/CVaR estimation."""
    if terminal_sample is not None:
        sample = np.asarray(terminal_sample, dtype=np.float64)
        if sample.ndim != 1 or sample.size < 2:
            raise ValidationError("terminal_sample must be 1-D length >= 2", context={})
        return sample
    grid = np.linspace(0.005, 0.995, 999)
    log_returns = distribution.quantile(grid)
    return spot * np.exp(np.asarray(log_returns, dtype=np.float64))


def select_iron_condor(
    distribution: _Distribution,
    chain: OptionChain,
    *,
    config: CondorSelectionConfig | None = None,
    terminal_sample: NDArray[np.float64] | None = None,
) -> CondorCandidate | None:
    cfg = config or CondorSelectionConfig()
    spot = distribution.initial_spot
    strikes = np.asarray(chain.strikes, dtype=np.float64)
    if strikes.size < 4:
        raise ValidationError("chain needs >= 4 strikes to build a condor", context={})

    terminal_spot = _resolve_terminal_sample(distribution, terminal_sample, spot)

    p = cfg.target_tail_probability
    put_short_target = float(distribution.quantile(p)[0])
    call_short_target = float(distribution.quantile(1.0 - p)[0])
    put_short_price = spot * np.exp(put_short_target)
    call_short_price = spot * np.exp(call_short_target)

    wing = cfg.wing_width_fraction * spot
    put_long_target = put_short_price - wing
    call_long_target = call_short_price + wing

    put_long = _nearest_strike(strikes, put_long_target)
    put_short = _nearest_strike(strikes, put_short_price)
    call_short = _nearest_strike(strikes, call_short_price)
    call_long = _nearest_strike(strikes, call_long_target)

    if not (put_long < put_short < call_short < call_long):
        return None

    condor = IronCondor(
        put_long_strike=put_long,
        put_short_strike=put_short,
        call_short_strike=call_short,
        call_long_strike=call_long,
        expiry=chain.expiry,
    )
    candidate = _evaluate_condor(condor, chain, terminal_spot, cfg)
    if candidate is None:
        return None
    if (
        candidate.win_probability < cfg.min_win_probability
        or candidate.net_credit < cfg.min_net_credit
    ):
        return None
    return candidate


def _evaluate_condor(
    condor: IronCondor,
    chain: OptionChain,
    terminal_spot: NDArray[np.float64],
    cfg: CondorSelectionConfig,
) -> CondorCandidate | None:
    sp = chain.put(condor.put_short_strike).quote
    lp = chain.put(condor.put_long_strike).quote
    sc = chain.call(condor.call_short_strike).quote
    lc = chain.call(condor.call_long_strike).quote

    net_credit = (sp.bid + sc.bid) - (lp.ask + lc.ask)
    if net_credit <= 0.0:
        return None
    
    mid_credit = (sp.mid + sc.mid) - (lp.mid + lc.mid)
    spread_cost = max(0.0, mid_credit - net_credit)

    payoff = iron_condor_payoff(condor, terminal_spot)
    pnl = net_credit + payoff
    expected_pnl = float(np.mean(pnl))

    lower, upper = condor.profit_zone
    win_probability = float(np.mean((terminal_spot >= lower) & (terminal_spot <= upper)))

    var_threshold = float(np.quantile(pnl, cfg.cvar_alpha))
    tail = pnl[pnl <= var_threshold]
    cvar = float(-np.mean(tail)) if tail.size > 0 else float(-var_threshold)

    score = expected_pnl - cfg.cvar_weight * max(0.0, cvar) - cfg.cost_weight * spread_cost

    return CondorCandidate(
        condor=condor,
        net_credit=net_credit,
        win_probability=win_probability,
        expected_pnl=expected_pnl,
        cvar=cvar,
        spread_cost=spread_cost,
        score=score,
    )


def select_bull_put_spread(
    distribution: _Distribution,
    chain: OptionChain,
    *,
    config: CondorSelectionConfig | None = None,
    terminal_sample: NDArray[np.float64] | None = None,
) -> BullPutSpreadCandidate | None:
    cfg = config or CondorSelectionConfig()
    spot = distribution.initial_spot
    strikes = np.asarray(chain.strikes, dtype=np.float64)
    if strikes.size < 2:
        return None

    terminal_spot = _resolve_terminal_sample(distribution, terminal_sample, spot)

    p = cfg.target_tail_probability
    short_target = float(distribution.quantile(p)[0])
    short_price = spot * np.exp(short_target)

    wing = cfg.wing_width_fraction * spot
    long_target = short_price - wing

    short_put = _nearest_strike(strikes, short_price)
    long_put = _nearest_strike(strikes, long_target)

    if not (long_put < short_put):
        return None

    spread = BullPutSpread(
        long_strike=long_put,
        short_strike=short_put,
        expiry=chain.expiry,
    )
    
    candidate = _evaluate_spread(spread, chain, terminal_spot, cfg)
    if candidate is None:
        return None
    if (
        candidate.win_probability < cfg.min_win_probability
        or candidate.net_credit < cfg.min_net_credit
    ):
        return None
    return candidate


def _evaluate_spread(
    spread: BullPutSpread,
    chain: OptionChain,
    terminal_spot: NDArray[np.float64],
    cfg: CondorSelectionConfig,
) -> BullPutSpreadCandidate | None:
    sp = chain.put(spread.short_strike).quote
    lp = chain.put(spread.long_strike).quote
    
    net_credit = sp.bid - lp.ask
    if net_credit <= 0.0:
        return None
    
    mid_credit = sp.mid - lp.mid
    spread_cost = max(0.0, mid_credit - net_credit)

    payoff = bull_put_spread_payoff(spread, terminal_spot)
    pnl = net_credit + payoff
    expected_pnl = float(np.mean(pnl))

    win_probability = float(np.mean(terminal_spot >= spread.short_strike))

    var_threshold = float(np.quantile(pnl, cfg.cvar_alpha))
    tail = pnl[pnl <= var_threshold]
    cvar = float(-np.mean(tail)) if tail.size > 0 else float(-var_threshold)

    score = expected_pnl - cfg.cvar_weight * max(0.0, cvar) - cfg.cost_weight * spread_cost

    return BullPutSpreadCandidate(
        spread=spread,
        net_credit=net_credit,
        win_probability=win_probability,
        expected_pnl=expected_pnl,
        cvar=cvar,
        spread_cost=spread_cost,
        score=score,
    )
