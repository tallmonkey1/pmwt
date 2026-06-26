r"""Surrogate-vs-Monte-Carlo fallback guardrail (SPEC §2.5, mandatory).

The surrogate is fast but approximate; the spec is explicit that the system must *never*
silently trust it. This module enforces that:

    "at inference we periodically re-run full MC and measure surrogate error
     (Wasserstein / CRPS). If error > tolerance, the system falls back to direct MC
     and flags the surrogate for retraining. No silent degradation."

:class:`SurrogateGuardrail` wraps a trained surrogate plus the simulator. On each request it
returns the surrogate distribution by default but, on a configurable audit cadence (and
always on the first call), it runs a full Monte-Carlo simulation, measures the discrepancy,
and:

* if the discrepancy is within tolerance -> serve the surrogate (fast path), or
* if it exceeds tolerance -> serve the Monte-Carlo distribution instead and raise a
  ``degraded`` flag with structured logging so operators/retraining are alerted.

Every decision is recorded in a :class:`GuardrailDecision` for the audit trail (SPEC §11).
"""

from __future__ import annotations

from dataclasses import dataclass

from ..core.errors import ValidationError
from ..core.logging import get_logger
from ..core.random import RandomFactory
from ..core.timegrid import TRADING_DAYS_PER_YEAR, TimeGrid
from ..core.validation import check_positive
from ..models.rbergomi import (
    ForwardVariance,
    HybridSimulator,
    RBergomiParams,
    build_terminal_distribution,
)
from ..models.rbergomi.results import TerminalDistribution
from .distribution import SurrogateDistribution
from .metrics import wasserstein1_from_quantiles
from .surrogate import DistributionSurrogate

__all__ = ["GuardrailConfig", "GuardrailDecision", "SurrogateGuardrail"]

_logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class GuardrailConfig:
    """Configuration for the fallback guardrail."""

    #: Max acceptable Wasserstein-1 distance (in log-return units) between surrogate and MC.
    max_wasserstein: float = 0.005
    #: Run a full-MC audit every ``audit_every`` requests (the first request always audits).
    audit_every: int = 50
    #: Monte-Carlo paths used for an audit / fallback.
    audit_n_paths: int = 40_000
    #: Simulation grid resolution per trading day for audits.
    steps_per_day: int = 8
    trading_days_per_year: int = TRADING_DAYS_PER_YEAR

    def __post_init__(self) -> None:
        check_positive(self.max_wasserstein, name="max_wasserstein")
        if self.audit_every < 1:
            raise ValidationError(
                "audit_every must be >= 1", context={"audit_every": self.audit_every}
            )
        check_positive(self.audit_n_paths, name="audit_n_paths")
        if self.steps_per_day < 1:
            raise ValidationError("steps_per_day must be >= 1", context={})


@dataclass(frozen=True, slots=True)
class GuardrailDecision:
    """Record of one guardrail evaluation (audit trail)."""

    used_monte_carlo: bool
    audited: bool
    wasserstein: float | None
    degraded: bool

    @property
    def used_surrogate(self) -> bool:
        """True if the fast surrogate path was served."""
        return not self.used_monte_carlo


class SurrogateGuardrail:
    """Serves surrogate distributions with periodic Monte-Carlo auditing and fallback."""

    def __init__(
        self,
        surrogate: DistributionSurrogate,
        *,
        rng_factory: RandomFactory,
        config: GuardrailConfig | None = None,
    ) -> None:
        if not isinstance(surrogate, DistributionSurrogate):
            raise ValidationError("surrogate must be a DistributionSurrogate", context={})
        if not surrogate.is_trained:
            raise ValidationError("surrogate must be trained", context={})
        self._surrogate = surrogate
        self._rng_factory = rng_factory
        self._config = config or GuardrailConfig()
        self._request_count = 0
        self._last_decision: GuardrailDecision | None = None

    @property
    def last_decision(self) -> GuardrailDecision | None:
        """The most recent guardrail decision, for inspection / auditing."""
        return self._last_decision

    def _run_monte_carlo(
        self,
        *,
        hurst: float,
        eta: float,
        rho: float,
        xi0: float,
        horizon: float,
        initial_spot: float,
    ) -> TerminalDistribution:
        """Run a full Monte-Carlo simulation for the scenario."""
        horizon_days = horizon * self._config.trading_days_per_year
        grid = TimeGrid.from_calendar_days(
            calendar_days=horizon_days,
            steps_per_day=self._config.steps_per_day,
            trading_days_per_year=self._config.trading_days_per_year,
        )
        params = RBergomiParams(
            hurst=hurst, eta=eta, rho=rho, forward_variance=ForwardVariance.flat(xi0)
        )
        # Distinct, reproducible audit RNG stream keyed by the rolling request count.
        factory = RandomFactory(self._rng_factory.seed + 10_000 + self._request_count)
        sim = HybridSimulator(params, rng_factory=factory, antithetic=True)
        paths = sim.simulate(
            grid=grid, n_paths=self._config.audit_n_paths, initial_spot=initial_spot
        )
        return build_terminal_distribution(paths)

    def distribution(
        self,
        *,
        hurst: float,
        eta: float,
        rho: float,
        xi0: float,
        horizon: float,
        initial_spot: float = 100.0,
        force_audit: bool = False,
    ) -> SurrogateDistribution | TerminalDistribution:
        """Return a terminal distribution, auditing and falling back to MC when required.

        On non-audit requests the surrogate distribution is returned directly (fast path).
        On audit requests (the first request, every ``audit_every`` requests, or when
        ``force_audit`` is set) a full Monte-Carlo distribution is computed; if the
        surrogate's Wasserstein-1 discrepancy from it exceeds the tolerance, the
        Monte-Carlo distribution is returned and a degradation is flagged.
        """
        self._request_count += 1
        surrogate_dist = self._surrogate.predict(
            hurst=hurst, eta=eta, rho=rho, xi0=xi0, horizon=horizon, initial_spot=initial_spot
        )

        should_audit = (
            force_audit
            or self._request_count == 1
            or (self._request_count % self._config.audit_every == 0)
        )
        if not should_audit:
            self._last_decision = GuardrailDecision(
                used_monte_carlo=False, audited=False, wasserstein=None, degraded=False
            )
            return surrogate_dist

        mc_dist = self._run_monte_carlo(
            hurst=hurst, eta=eta, rho=rho, xi0=xi0, horizon=horizon, initial_spot=initial_spot
        )
        levels = self._surrogate.quantile_levels
        mc_quantiles = mc_dist.quantile(levels)
        surrogate_quantiles = surrogate_dist.quantile(levels)
        distance = wasserstein1_from_quantiles(surrogate_quantiles, mc_quantiles, levels)

        degraded = distance > self._config.max_wasserstein
        self._last_decision = GuardrailDecision(
            used_monte_carlo=degraded, audited=True, wasserstein=distance, degraded=degraded
        )

        if degraded:
            _logger.warning(
                "surrogate_degraded_fallback_to_mc",
                extra={
                    "wasserstein": distance,
                    "tolerance": self._config.max_wasserstein,
                    "horizon": horizon,
                },
            )
            return mc_dist

        _logger.debug(
            "surrogate_audit_passed",
            extra={"wasserstein": distance, "tolerance": self._config.max_wasserstein},
        )
        return surrogate_dist
