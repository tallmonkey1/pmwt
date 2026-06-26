r"""Monte-Carlo pricing off a simulated terminal distribution.

This is how the engine prices options under the rough-Bergomi model, which has no
closed-form solution (SPEC §2.2). Given a Monte-Carlo sample of terminal underlying prices
(produced by the rBergomi simulator), the price of any European payoff is the discounted
sample mean of its payoff, and every estimate is returned **with its Monte-Carlo standard
error** so downstream code can enforce convergence tolerances (SPEC §2.3).

The pricer is agnostic to how the terminal prices were generated -- it accepts a
:class:`~options_engine.models.rbergomi.results.TerminalDistribution` (or a raw terminal
sample) -- which keeps it reusable and independently testable. In the deterministic-variance
(Black-Scholes) limit its prices converge to the analytic formula, which is the validation
oracle used in the tests.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..core.errors import ConvergenceError, ValidationError
from ..core.validation import check_finite, check_positive
from ..models.rbergomi.diagnostics import MonteCarloSummary, mean_standard_error
from ..models.rbergomi.results import TerminalDistribution
from .instruments import EuropeanOption, IronCondor
from .payoff import iron_condor_payoff, option_payoff

__all__ = ["fair_iron_condor_credit", "price_iron_condor", "price_option"]


def _discounted_summary(payoffs: NDArray[np.float64], discount: float) -> MonteCarloSummary:
    """Return the discounted sample-mean estimate with Monte-Carlo standard error."""
    summary = mean_standard_error(payoffs)
    return MonteCarloSummary(
        estimate=discount * summary.estimate,
        standard_error=discount * summary.standard_error,
        n_samples=summary.n_samples,
    )


def price_option(
    option: EuropeanOption,
    distribution: TerminalDistribution,
    *,
    rate: float = 0.0,
    max_rel_standard_error: float | None = None,
) -> MonteCarloSummary:
    r"""Price a European option as the discounted expected payoff under the MC distribution.

    Parameters
    ----------
    option:
        The option to price. Its ``expiry`` must match the distribution horizon (within a
        small tolerance) so the discounting and the payoff refer to the same date.
    distribution:
        Monte-Carlo terminal distribution of the underlying.
    rate:
        Continuously-compounded discount rate applied over the horizon.
    max_rel_standard_error:
        Optional convergence gate: if the relative standard error of the price exceeds this
        tolerance a :class:`~options_engine.core.errors.ConvergenceError` is raised.

    Returns
    -------
    MonteCarloSummary
        Discounted price estimate, its standard error, and the sample size.
    """
    check_finite(rate, name="rate")
    if abs(option.expiry - distribution.horizon) > 1e-9:
        raise ValidationError(
            "option expiry must match the distribution horizon",
            context={"expiry": option.expiry, "horizon": distribution.horizon},
        )
    terminal_spot = distribution.terminal_spot()
    payoffs = option_payoff(option, terminal_spot)
    discount = float(np.exp(-rate * option.expiry))
    summary = _discounted_summary(payoffs, discount)
    _enforce_tolerance(summary, max_rel_standard_error, label="option price")
    return summary


def price_iron_condor(
    condor: IronCondor,
    distribution: TerminalDistribution,
    *,
    rate: float = 0.0,
    max_rel_standard_error: float | None = None,
) -> MonteCarloSummary:
    r"""Return the discounted expected value of the iron-condor *liability* at expiry.

    The result is the present value of the (non-positive) terminal payoff of the short
    structure -- i.e. the expected amount paid out at expiry, discounted. The *fair entry
    credit* that makes the structure zero-NPV is the negative of this value; see
    :func:`fair_iron_condor_credit`.
    """
    check_finite(rate, name="rate")
    if abs(condor.expiry - distribution.horizon) > 1e-9:
        raise ValidationError(
            "condor expiry must match the distribution horizon",
            context={"expiry": condor.expiry, "horizon": distribution.horizon},
        )
    payoffs = iron_condor_payoff(condor, distribution.terminal_spot())
    discount = float(np.exp(-rate * condor.expiry))
    summary = _discounted_summary(payoffs, discount)
    _enforce_tolerance(summary, max_rel_standard_error, label="iron condor value")
    return summary


def fair_iron_condor_credit(
    condor: IronCondor,
    distribution: TerminalDistribution,
    *,
    rate: float = 0.0,
) -> MonteCarloSummary:
    r"""Return the model-fair entry credit of the iron condor (zero-NPV premium).

    Because the short condor's terminal payoff is non-positive, its expected discounted
    liability is :math:`\le 0`; the fair credit is the negative of that expectation, hence
    non-negative. Selling above the fair credit is the model's notion of edge (SPEC §1.3).
    """
    liability = price_iron_condor(condor, distribution, rate=rate)
    return MonteCarloSummary(
        estimate=-liability.estimate,
        standard_error=liability.standard_error,
        n_samples=liability.n_samples,
    )


def _enforce_tolerance(
    summary: MonteCarloSummary, max_rel_standard_error: float | None, *, label: str
) -> None:
    """Raise a ConvergenceError if the relative standard error exceeds the tolerance."""
    if max_rel_standard_error is None:
        return
    check_positive(max_rel_standard_error, name="max_rel_standard_error")
    rel = summary.relative_standard_error
    if rel > max_rel_standard_error:
        raise ConvergenceError(
            f"{label} Monte-Carlo standard error exceeds tolerance; increase n_paths",
            context={
                "relative_standard_error": rel,
                "tolerance": max_rel_standard_error,
                "n_samples": summary.n_samples,
            },
        )
