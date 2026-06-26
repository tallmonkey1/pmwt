r"""Regime trade gate (SPEC §2.6).

The strategy only opens new short-premium risk in calmly-confident conditions:

    "Trade only when P(low-vol now) AND P(low-vol next step) both exceed calibrated
     thresholds, with the threshold itself tuned on out-of-sample data."

This module turns a :class:`RegimeNowcast` into a binary go/no-go with a structured,
auditable reason. It deliberately also blocks when the *high*-vol probability is
non-trivial, even if low-vol is the modal class -- a defensive asymmetry appropriate for a
short-gamma book where being wrong in high vol is the expensive failure mode.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..core.enums import VolRegime
from ..core.validation import check_probability
from .detector import RegimeNowcast

__all__ = ["RegimeGateConfig", "RegimeGateDecision", "evaluate_regime_gate"]


@dataclass(frozen=True, slots=True)
class RegimeGateConfig:
    """Thresholds for the regime trade gate (tune on out-of-sample data)."""

    min_low_now: float = 0.65
    min_low_next: float = 0.60
    max_high_now: float = 0.15
    max_high_next: float = 0.20

    def __post_init__(self) -> None:
        check_probability(self.min_low_now, name="min_low_now")
        check_probability(self.min_low_next, name="min_low_next")
        check_probability(self.max_high_now, name="max_high_now")
        check_probability(self.max_high_next, name="max_high_next")


@dataclass(frozen=True, slots=True)
class RegimeGateDecision:
    """Outcome of the regime gate, with a machine-readable reason code."""

    allow_new_risk: bool
    reason: str
    low_now: float
    low_next: float
    high_now: float
    high_next: float


def evaluate_regime_gate(
    nowcast: RegimeNowcast, *, config: RegimeGateConfig | None = None
) -> RegimeGateDecision:
    """Return the go/no-go decision for opening new risk given a regime nowcast.

    The gate passes only if *all* conditions hold: low-vol probability is high enough both
    now and next step, and high-vol probability is low enough both now and next step. The
    first failing condition is reported as the reason, making rejections auditable.
    """
    cfg = config or RegimeGateConfig()
    low_now = nowcast.current_prob(VolRegime.LOW)
    low_next = nowcast.next_prob(VolRegime.LOW)
    high_now = nowcast.current_prob(VolRegime.HIGH)
    high_next = nowcast.next_prob(VolRegime.HIGH)

    checks = (
        (low_now >= cfg.min_low_now, f"low_now {low_now:.3f} < {cfg.min_low_now}"),
        (low_next >= cfg.min_low_next, f"low_next {low_next:.3f} < {cfg.min_low_next}"),
        (high_now <= cfg.max_high_now, f"high_now {high_now:.3f} > {cfg.max_high_now}"),
        (high_next <= cfg.max_high_next, f"high_next {high_next:.3f} > {cfg.max_high_next}"),
    )
    for passed, message in checks:
        if not passed:
            return RegimeGateDecision(
                allow_new_risk=False,
                reason=message,
                low_now=low_now,
                low_next=low_next,
                high_now=high_now,
                high_next=high_next,
            )

    return RegimeGateDecision(
        allow_new_risk=True,
        reason="all regime conditions satisfied",
        low_now=low_now,
        low_next=low_next,
        high_now=high_now,
        high_next=high_next,
    )
