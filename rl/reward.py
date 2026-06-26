r"""Reward function for the RL trading agent (SPEC §4.4).

The reward is engineered to optimize **risk-adjusted** P&L while charging real transaction
costs every step (so the agent learns spread friction) and penalizing tail/limit breaches.
It is the literal implementation of the spec's reward:

.. math::

    r_t = \Delta\mathrm{PnL}_t
        - \lambda_\text{risk}\, \text{incremental\_CVaR}
        - \lambda_\text{cost}\, (\text{spread} + \text{slippage} + \text{commission})
        - \lambda_\text{margin}\, \text{margin\_utilization}
        - \lambda_\text{tail}\, \text{limit\_breach\_penalty}
        + \text{theta\_capture\_credit (shaping, decayed over training)}

Design safeguards against reward hacking (SPEC §4.4):

* P&L is the realized + mark-to-market change, in *risk-normalized* units (divided by a
  reference risk amount) so the scale is stable across account sizes and episodes.
* Transaction costs are charged on the cash actually paid, so "churning" is never free.
* The theta-capture shaping term is multiplied by a caller-supplied ``shaping_coef`` that the
  training loop decays to zero, so the final policy is judged on true reward, not the shaping.
* A hard ``limit_breach_penalty`` makes risk-limit violations strictly dominated, steering
  the agent to stay inside the supervisor's constraints rather than fighting them.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..core.errors import ValidationError
from ..core.validation import check_finite, check_non_negative, check_positive

__all__ = ["RewardBreakdown", "RewardConfig", "RewardInputs", "compute_reward"]


@dataclass(frozen=True, slots=True)
class RewardConfig:
    """Weights for the reward components (the ``lambda`` coefficients)."""

    risk_weight: float = 1.0  # lambda_risk on incremental CVaR
    cost_weight: float = 1.0  # lambda_cost on transaction costs
    margin_weight: float = 0.1  # lambda_margin on margin utilization
    tail_weight: float = 1.0  # lambda_tail on limit-breach penalty
    theta_shaping_weight: float = 0.05  # base shaping credit (further scaled by shaping_coef)
    reference_risk: float = 1000.0  # risk-normalization denominator (account currency)

    def __post_init__(self) -> None:
        check_non_negative(self.risk_weight, name="risk_weight")
        check_non_negative(self.cost_weight, name="cost_weight")
        check_non_negative(self.margin_weight, name="margin_weight")
        check_non_negative(self.tail_weight, name="tail_weight")
        check_non_negative(self.theta_shaping_weight, name="theta_shaping_weight")
        check_positive(self.reference_risk, name="reference_risk")


@dataclass(frozen=True, slots=True)
class RewardInputs:
    """The per-step quantities the reward is computed from (account currency)."""

    pnl_change: float  # realized + unrealized PnL change this step
    incremental_cvar: float  # change in portfolio tail risk (>= 0 is added risk)
    transaction_cost: float  # spread + slippage + commission actually paid (>= 0)
    margin_utilization: float  # total margin / equity (>= 0)
    theta_captured: float  # premium decay captured this step (shaping signal, >= 0)
    limit_breached: bool  # whether a hard risk limit was breached this step

    def __post_init__(self) -> None:
        check_finite(self.pnl_change, name="pnl_change")
        check_finite(self.incremental_cvar, name="incremental_cvar")
        check_non_negative(self.transaction_cost, name="transaction_cost")
        check_non_negative(self.margin_utilization, name="margin_utilization")
        check_finite(self.theta_captured, name="theta_captured")


@dataclass(frozen=True, slots=True)
class RewardBreakdown:
    """The reward decomposed into its components, for diagnostics and the audit trail."""

    total: float
    pnl_term: float
    risk_term: float
    cost_term: float
    margin_term: float
    tail_term: float
    shaping_term: float


def compute_reward(
    inputs: RewardInputs,
    *,
    config: RewardConfig | None = None,
    shaping_coef: float = 1.0,
    limit_breach_penalty: float = 5.0,
) -> RewardBreakdown:
    r"""Compute the per-step reward and its component breakdown.

    Parameters
    ----------
    inputs:
        The per-step P&L, risk, cost, margin, and breach quantities.
    config:
        Reward weights; defaults to :class:`RewardConfig`.
    shaping_coef:
        Multiplier on the theta-capture shaping term, in ``[0, 1]``. The training loop decays
        this to zero so the final policy is evaluated on the unshaped objective (potential-
        based shaping discipline -- the shaping never changes the optimal policy, only the
        learning speed).
    limit_breach_penalty:
        The fixed (risk-normalized) penalty applied when a hard limit is breached. Large
        enough that breaching is always dominated by not breaching.

    Returns
    -------
    RewardBreakdown
        The total reward and each additive component.
    """
    cfg = config or RewardConfig()
    if not 0.0 <= shaping_coef <= 1.0:
        raise ValidationError(
            "shaping_coef must lie in [0, 1]", context={"shaping_coef": shaping_coef}
        )
    check_non_negative(limit_breach_penalty, name="limit_breach_penalty")

    scale = cfg.reference_risk

    pnl_term = inputs.pnl_change / scale
    # Only *added* tail risk is penalized; risk reduction is not rewarded as free P&L.
    risk_term = -cfg.risk_weight * max(0.0, inputs.incremental_cvar) / scale
    cost_term = -cfg.cost_weight * inputs.transaction_cost / scale
    margin_term = -cfg.margin_weight * inputs.margin_utilization
    tail_term = -cfg.tail_weight * limit_breach_penalty if inputs.limit_breached else 0.0
    shaping_term = (
        cfg.theta_shaping_weight * shaping_coef * max(0.0, inputs.theta_captured) / scale
    )

    total = pnl_term + risk_term + cost_term + margin_term + tail_term + shaping_term
    return RewardBreakdown(
        total=total,
        pnl_term=pnl_term,
        risk_term=risk_term,
        cost_term=cost_term,
        margin_term=margin_term,
        tail_term=tail_term,
        shaping_term=shaping_term,
    )
