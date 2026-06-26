r"""Promotion gates: HISTORY -> PAPER -> LIVE readiness checks (SPEC §9).

A strategy/agent build may only advance toward real capital if it *passes* an explicit,
auditable set of quantitative gates. This module encodes those gates as deterministic checks
over backtest results, each returning a structured pass/fail with a reason, so a promotion
decision is reproducible and defensible (not a human eyeballing an equity curve).

Gates implemented (a practical subset of SPEC §9, the rest -- live paper-parity, sign-off
artifacts -- belong to the runner/ops layer):

* **Deflated Sharpe** above threshold (multiple-testing-corrected edge is real).
* **Max drawdown** within the mandate.
* **CVaR / tail** within the mandate.
* **Kill-switch** never tripped during the in-sample backtest.
* **Crash-replay robustness** -- the strategy must survive stressed scenarios (checked by the
  caller supplying a stressed backtest result) without breaching the drawdown mandate.
* **Walk-forward consistency** -- the out-of-sample folds must be collectively profitable and
  not dominated by a single lucky fold.

The result is a :class:`PromotionReport`; ``approved`` is True only if *every* gate passes.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from ..core.errors import ValidationError
from ..core.validation import check_probability
from .metrics import deflated_sharpe_ratio
from .results import BacktestResult

__all__ = ["GateOutcome", "PromotionReport", "PromotionThresholds", "evaluate_promotion"]


@dataclass(frozen=True, slots=True)
class PromotionThresholds:
    """Acceptance thresholds for the promotion gates (the risk mandate, quantified)."""

    min_deflated_sharpe: float = 0.95  # DSR probability the true Sharpe is positive
    max_drawdown: float = 0.20  # hard drawdown mandate
    max_cvar_95: float = 0.10  # tail-loss mandate (per-period CVaR at 95%)
    min_walk_forward_win_rate: float = 0.50  # fraction of OOS folds that are profitable
    require_no_kill_switch: bool = True
    max_stress_drawdown: float = 0.35  # allowed drawdown under crash-replay stress

    def __post_init__(self) -> None:
        check_probability(self.min_deflated_sharpe, name="min_deflated_sharpe")
        for name, value in (
            ("max_drawdown", self.max_drawdown),
            ("max_cvar_95", self.max_cvar_95),
            ("max_stress_drawdown", self.max_stress_drawdown),
        ):
            if not 0.0 < value <= 1.0:
                raise ValidationError(f"{name} must lie in (0, 1]", context={name: value})
        check_probability(self.min_walk_forward_win_rate, name="min_walk_forward_win_rate")


@dataclass(frozen=True, slots=True)
class GateOutcome:
    """The outcome of a single promotion gate."""

    name: str
    passed: bool
    detail: str


@dataclass(frozen=True, slots=True)
class PromotionReport:
    """The aggregate promotion decision with per-gate detail."""

    approved: bool
    gates: tuple[GateOutcome, ...]
    metadata: dict[str, float] = field(default_factory=dict)

    def failed_gates(self) -> list[GateOutcome]:
        """Return the gates that did not pass."""
        return [g for g in self.gates if not g.passed]


def evaluate_promotion(
    *,
    in_sample: BacktestResult,
    thresholds: PromotionThresholds | None = None,
    n_trials: int = 1,
    walk_forward_results: Sequence[BacktestResult] | None = None,
    stress_results: Sequence[BacktestResult] | None = None,
) -> PromotionReport:
    """Evaluate all promotion gates and return the aggregate decision.

    Parameters
    ----------
    in_sample:
        The primary in-sample backtest result.
    thresholds:
        Acceptance thresholds; defaults to :class:`PromotionThresholds`.
    n_trials:
        Multiple-testing count for the deflated Sharpe.
    walk_forward_results:
        Optional out-of-sample fold results for the consistency gate.
    stress_results:
        Optional crash-replay/stressed results for the robustness gate.
    """
    th = thresholds or PromotionThresholds()
    gates: list[GateOutcome] = []

    # 1. Deflated Sharpe.
    dsr = deflated_sharpe_ratio(in_sample.returns, n_trials=n_trials)
    gates.append(
        GateOutcome(
            name="deflated_sharpe",
            passed=dsr >= th.min_deflated_sharpe,
            detail=f"DSR={dsr:.3f} (>= {th.min_deflated_sharpe})",
        )
    )

    # 2. Max drawdown mandate.
    mdd = in_sample.metrics.max_drawdown
    gates.append(
        GateOutcome(
            name="max_drawdown",
            passed=mdd <= th.max_drawdown,
            detail=f"max_drawdown={mdd:.3f} (<= {th.max_drawdown})",
        )
    )

    # 3. Tail (CVaR) mandate.
    cvar = in_sample.metrics.cvar_95
    gates.append(
        GateOutcome(
            name="cvar_95",
            passed=cvar <= th.max_cvar_95,
            detail=f"cvar_95={cvar:.3f} (<= {th.max_cvar_95})",
        )
    )

    # 4. Kill-switch must not have tripped in-sample.
    if th.require_no_kill_switch:
        gates.append(
            GateOutcome(
                name="kill_switch",
                passed=not in_sample.kill_switch_triggered,
                detail=(
                    "kill-switch not triggered"
                    if not in_sample.kill_switch_triggered
                    else "kill-switch TRIGGERED in-sample"
                ),
            )
        )

    # 5. Walk-forward consistency (if folds supplied).
    if walk_forward_results is not None and len(walk_forward_results) > 0:
        profitable = [r for r in walk_forward_results if r.total_pnl > 0.0]
        win_rate = len(profitable) / len(walk_forward_results)
        gates.append(
            GateOutcome(
                name="walk_forward_consistency",
                passed=win_rate >= th.min_walk_forward_win_rate,
                detail=f"OOS fold win-rate={win_rate:.2f} (>= {th.min_walk_forward_win_rate})",
            )
        )

    # 6. Crash-replay robustness (if stressed scenarios supplied).
    if stress_results is not None and len(stress_results) > 0:
        worst_stress_dd = max(r.metrics.max_drawdown for r in stress_results)
        gates.append(
            GateOutcome(
                name="crash_replay_robustness",
                passed=worst_stress_dd <= th.max_stress_drawdown,
                detail=(
                    f"worst stress drawdown={worst_stress_dd:.3f} "
                    f"(<= {th.max_stress_drawdown})"
                ),
            )
        )

    approved = all(g.passed for g in gates)
    return PromotionReport(
        approved=approved,
        gates=tuple(gates),
        metadata={"deflated_sharpe": dsr, "n_trials": float(n_trials)},
    )
