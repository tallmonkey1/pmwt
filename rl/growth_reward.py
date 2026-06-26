r"""Growth-optimal reward that cannot collapse to "do nothing" (SPEC §4.4).

The original additive reward (:mod:`options_engine.rl.reward`) has a free, risk-free zero at
the FLAT action, so an advantage-normalized PPO policy can rationally collapse to never trading
even when the strategy has positive expected edge. This module implements the redesigned reward
studied in ``analysis/REWARD_DESIGN.md``, whose optimum is **trade-the-edge**:

* **Log-wealth (Kelly) core.** The per-step reward is the log-wealth increment
  :math:`\log(1 + \Delta\mathrm{PnL}/\text{equity})`. Maximizing its expectation maximizes
  long-run growth; a positive-edge bet has *positive expected log-growth* (Kelly), so trading
  edge is provably better than the zero baseline, while losses are punished concavely (built-in,
  correct risk aversion for a short-gamma book).

* **Opportunity cost of inaction (self-gating).** Declining a *qualifying, positive-edge*
  condor incurs a small penalty, removing the free-zero attractor. It is self-gating: when the
  market offers no qualifying condor (high vol, news blackout, no edge), the available edge is
  zero and sitting out is free -- so the agent is never pushed into a bad trade.

* **Expected-edge shaping (dense, decayed).** A potential-based shaping term rewards the
  model's *expected* risk-adjusted P&L of the chosen condor (a low-variance teacher), decayed to
  zero over training so the final policy is judged on realized growth.

* **Soft risk penalty + hard breach penalty.** A small normalized CVaR penalty plus the
  non-negotiable hard limit-breach penalty.

Every term is normalized by equity before weighting, so the weights are interpretable and the
soft penalties never swamp the edge. The function is pure arithmetic and is verified by unit
tests against its provable properties -- no model is trained to validate it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..core.errors import ValidationError
from ..core.validation import check_finite, check_non_negative, check_positive

__all__ = [
    "GrowthRewardBreakdown",
    "GrowthRewardConfig",
    "GrowthRewardInputs",
    "compute_growth_reward",
]

# Floor on (1 + step_return) so the log is always finite even at a (clamped) total loss.
_WEALTH_FLOOR = 1e-6


@dataclass(frozen=True, slots=True)
class GrowthRewardConfig:
    """Weights for the growth reward. Defaults are calibrated so edge survives the penalties.

    All non-growth terms act on equity-normalized quantities, so a weight of ``1.0`` means
    "as important as one unit of log-growth". The opportunity weight is deliberately the
    largest soft term: declining real edge should be the clearest mistake.
    """

    opportunity_weight: float = 1.0  # penalty for declining qualifying positive edge
    edge_shaping_weight: float = 0.5  # dense expected-edge teacher (decayed by shaping_coef)
    risk_weight: float = 0.1  # soft incremental-CVaR penalty
    growth_scale: float = 1.0  # multiplier on the log-wealth core (keep at 1.0 normally)

    def __post_init__(self) -> None:
        check_non_negative(self.opportunity_weight, name="opportunity_weight")
        check_non_negative(self.edge_shaping_weight, name="edge_shaping_weight")
        check_non_negative(self.risk_weight, name="risk_weight")
        check_positive(self.growth_scale, name="growth_scale")


@dataclass(frozen=True, slots=True)
class GrowthRewardInputs:
    """Per-step quantities for the growth reward (account currency unless noted).

    Parameters
    ----------
    pnl_change:
        Realized + unrealized P&L change this step.
    equity:
        Account equity at the start of the step (the growth denominator); must be positive.
    traded:
        Whether a trade was opened this step.
    chosen_expected_edge:
        The model's expected risk-adjusted P&L of the condor that was traded (>= 0 typically;
        ignored unless ``traded``). The dense shaping signal.
    best_available_edge:
        The expected risk-adjusted P&L of the best *qualifying* condor the agent could have
        traded this step but did not (>= 0; ignored unless NOT ``traded``). Zero when no
        qualifying condor exists -- which is what makes the opportunity cost self-gating.
    incremental_cvar:
        Added portfolio tail risk this step (>= 0 means risk was added).
    limit_breached:
        Whether a hard risk limit was breached this step.
    """

    pnl_change: float
    equity: float
    traded: bool
    chosen_expected_edge: float = 0.0
    best_available_edge: float = 0.0
    incremental_cvar: float = 0.0
    limit_breached: bool = False

    def __post_init__(self) -> None:
        check_finite(self.pnl_change, name="pnl_change")
        check_positive(self.equity, name="equity")
        check_finite(self.chosen_expected_edge, name="chosen_expected_edge")
        check_non_negative(self.best_available_edge, name="best_available_edge")
        check_finite(self.incremental_cvar, name="incremental_cvar")


@dataclass(frozen=True, slots=True)
class GrowthRewardBreakdown:
    """The growth reward decomposed into its components (audit trail / diagnostics)."""

    total: float
    growth_term: float
    opportunity_term: float
    edge_shaping_term: float
    risk_term: float
    breach_term: float


def compute_growth_reward(
    inputs: GrowthRewardInputs,
    *,
    config: GrowthRewardConfig | None = None,
    shaping_coef: float = 1.0,
    limit_breach_penalty: float = 5.0,
) -> GrowthRewardBreakdown:
    r"""Compute the growth-optimal per-step reward and its breakdown.

    Parameters
    ----------
    inputs:
        The per-step P&L, equity, trade flag, edges, risk, and breach quantities.
    config:
        Reward weights; defaults to :class:`GrowthRewardConfig`.
    shaping_coef:
        Multiplier in ``[0, 1]`` on the expected-edge shaping term; the training loop decays
        it to zero so the final policy is judged on realized growth.
    limit_breach_penalty:
        Fixed penalty when a hard limit is breached (large enough to dominate).

    Returns
    -------
    GrowthRewardBreakdown
        The total reward and each additive component.

    Notes
    -----
    Provable properties (see ``tests/rl/test_growth_reward.py``):

    * FLAT with no available edge -> reward == 0 (sitting out a bad market is free).
    * FLAT with available edge    -> reward < 0 (the free-zero attractor is removed).
    * a positive-edge trade realizing its expectation strictly beats declining that edge.
    """
    cfg = config or GrowthRewardConfig()
    if not 0.0 <= shaping_coef <= 1.0:
        raise ValidationError(
            "shaping_coef must lie in [0, 1]", context={"shaping_coef": shaping_coef}
        )
    check_non_negative(limit_breach_penalty, name="limit_breach_penalty")

    # --- Kelly log-wealth core (always present) ---------------------------------------
    step_return = inputs.pnl_change / inputs.equity
    wealth = max(_WEALTH_FLOOR, 1.0 + step_return)
    growth_term = cfg.growth_scale * math.log(wealth)

    # --- Opportunity cost of inaction (FLAT steps only) -------------------------------
    # Penalize declining qualifying positive edge; zero edge => zero penalty (self-gating).
    if inputs.traded:
        opportunity_term = 0.0
    else:
        opportunity_term = (
            -cfg.opportunity_weight * max(0.0, inputs.best_available_edge) / inputs.equity
        )

    # --- Expected-edge shaping (TRADE steps only, decayed) ----------------------------
    if inputs.traded:
        edge_shaping_term = (
            cfg.edge_shaping_weight * shaping_coef * (inputs.chosen_expected_edge / inputs.equity)
        )
    else:
        edge_shaping_term = 0.0

    # --- Soft incremental-CVaR penalty (only added risk) ------------------------------
    risk_term = -cfg.risk_weight * max(0.0, inputs.incremental_cvar) / inputs.equity

    # --- Hard limit-breach penalty ----------------------------------------------------
    breach_term = -limit_breach_penalty if inputs.limit_breached else 0.0

    total = growth_term + opportunity_term + edge_shaping_term + risk_term + breach_term
    return GrowthRewardBreakdown(
        total=total,
        growth_term=growth_term,
        opportunity_term=opportunity_term,
        edge_shaping_term=edge_shaping_term,
        risk_term=risk_term,
        breach_term=breach_term,
    )
