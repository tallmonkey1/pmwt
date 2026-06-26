r"""Jump detection via the Barndorff-Nielsen & Shephard ratio test.

SPEC §2.4 calls for a likelihood-ratio-style test to decide whether the jump component of
the model should be enabled. The standard, robust, non-parametric tool is the
**BNS jump test** (Barndorff-Nielsen & Shephard, 2006; Huang & Tauchen, 2005), which
compares realized variance (sensitive to jumps) against bipower variation (robust to
jumps). The relative-jump statistic

.. math::

    RJ = \frac{RV - BV}{RV}

is asymptotically normal with zero mean under the no-jump null and a known variance
expressed through tri-power quarticity. A large positive ``RJ`` is evidence of jumps.

This module returns a structured :class:`JumpTestResult` with the statistic, its p-value,
and the decision at a configurable significance level, so the calling calibration can turn
jumps on/off transparently and auditable.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import lgamma

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from ..core.errors import ValidationError
from ..core.validation import check_unit_interval
from .realized import bipower_variation, realized_variance

__all__ = ["JumpTestResult", "bns_jump_test"]

# theta for the relative-jump statistic variance: (pi/2)^2 + pi - 5.
_THETA = (np.pi / 2.0) ** 2 + np.pi - 5.0
# mu_{4/3} raised to the -3 power, the scaling constant of tri-power quarticity.
_MU_43 = 2.0 ** (2.0 / 3.0) * float(np.exp(lgamma(7.0 / 6.0)) / np.exp(lgamma(0.5)))


@dataclass(frozen=True, slots=True)
class JumpTestResult:
    """Outcome of the BNS jump test over a window of returns."""

    statistic: float
    p_value: float
    jumps_detected: bool
    realized_variance: float
    bipower_variation: float
    significance: float

    @property
    def relative_jump(self) -> float:
        """The relative-jump measure ``(RV - BV) / RV`` (the jump share of variance)."""
        if self.realized_variance <= 0.0:
            return 0.0
        return (self.realized_variance - self.bipower_variation) / self.realized_variance


def _tripower_quarticity(returns: NDArray[np.float64]) -> float:
    r"""Return the tri-power quarticity, a jump-robust estimator of integrated quarticity."""
    n = returns.size
    abs_r = np.abs(returns)
    # product of three consecutive |r|^{4/3}
    powers = abs_r ** (4.0 / 3.0)
    triples = powers[2:] * powers[1:-1] * powers[:-2]
    scale = n * (_MU_43**-3) * (n / (n - 2.0))
    return float(scale * np.sum(triples))


def bns_jump_test(
    returns: NDArray[np.float64],
    *,
    significance: float = 0.01,
) -> JumpTestResult:
    r"""Run the Barndorff-Nielsen & Shephard relative-jump test on a return window.

    Parameters
    ----------
    returns:
        One-dimensional array of (intraday) log-returns for the window under test. Needs at
        least four observations for the tri-power quarticity to be defined.
    significance:
        One-sided significance level for declaring jumps (default 1%). The test is
        one-sided because jumps can only *increase* RV relative to BV.

    Returns
    -------
    JumpTestResult
        The statistic, p-value, decision, and the underlying RV/BV values.
    """
    check_unit_interval(significance, name="significance", inclusive=False)
    r = np.asarray(returns, dtype=np.float64)
    if r.ndim != 1 or r.size < 4:
        raise ValidationError(
            "returns must be a 1-D array with at least four observations",
            context={"size": int(r.size)},
        )
    if not np.all(np.isfinite(r)):
        raise ValidationError("returns contain non-finite values", context={})

    rv = realized_variance(r)
    bv = bipower_variation(r)
    n = r.size

    if rv <= 0.0:
        # A flat window has no variance and trivially no jumps.
        return JumpTestResult(
            statistic=0.0,
            p_value=1.0,
            jumps_detected=False,
            realized_variance=rv,
            bipower_variation=bv,
            significance=significance,
        )

    relative_jump = (rv - bv) / rv
    quarticity = _tripower_quarticity(r)
    # Guard the variance term; quarticity is non-negative by construction.
    denom = _THETA * (1.0 / n) * max(quarticity / (bv * bv), 1e-12)
    statistic = relative_jump / np.sqrt(denom)

    # One-sided upper-tail p-value.
    p_value = float(norm.sf(statistic))
    jumps_detected = p_value < significance

    return JumpTestResult(
        statistic=float(statistic),
        p_value=p_value,
        jumps_detected=bool(jumps_detected),
        realized_variance=float(rv),
        bipower_variation=float(bv),
        significance=float(significance),
    )
