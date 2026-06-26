"""Containers for calibration outputs (SPEC §2.4).

Every calibrated quantity carries enough metadata to be auditable and to be *rejected* when
stale or low-quality (SPEC §13: reproducibility, observability). A point estimate without a
timestamp, a sample size, and a quality diagnostic is not institutional-grade -- it is a
silent liability -- so these containers make that metadata mandatory.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field

from ..core.errors import ValidationError
from ..core.validation import check_finite, check_positive

__all__ = ["CalibrationResult", "ParameterEstimate"]


@dataclass(frozen=True, slots=True)
class ParameterEstimate:
    """A single calibrated parameter with an uncertainty band and goodness-of-fit.

    Parameters
    ----------
    name:
        Parameter name (e.g. ``"hurst"``).
    value:
        Point estimate.
    std_error:
        Estimated standard error of the point estimate (``>= 0``). Use ``0.0`` only when an
        analytic/exact value is supplied.
    r_squared:
        Optional goodness-of-fit in ``[0, 1]`` for regression-based estimators. ``None`` if
        not applicable.
    n_observations:
        Number of observations used. Must be positive.
    """

    name: str
    value: float
    std_error: float
    n_observations: int
    r_squared: float | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValidationError("name must be non-empty", context={})
        check_finite(self.value, name=f"{self.name}.value")
        if self.std_error < 0.0:
            raise ValidationError(
                "std_error must be non-negative",
                context={"name": self.name, "std_error": self.std_error},
            )
        check_finite(self.std_error, name=f"{self.name}.std_error")
        if self.n_observations <= 0:
            raise ValidationError(
                "n_observations must be positive",
                context={"name": self.name, "n_observations": self.n_observations},
            )
        if self.r_squared is not None and not (0.0 <= self.r_squared <= 1.0):
            raise ValidationError(
                "r_squared must lie in [0, 1]",
                context={"name": self.name, "r_squared": self.r_squared},
            )

    def confidence_interval(self, n_sigma: float = 2.0) -> tuple[float, float]:
        """Return a symmetric ``+/- n_sigma`` interval around the estimate."""
        check_positive(n_sigma, name="n_sigma")
        half = n_sigma * self.std_error
        return self.value - half, self.value + half


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    """The full set of calibrated rBergomi parameters with provenance.

    This is the hand-off from calibration to the simulator. The ``as_of`` timestamp and
    ``data_start``/``data_end`` window make staleness checks possible (SPEC §2.4:
    "stale calibrations are rejected"); :meth:`is_stale` implements that policy.
    """

    hurst: ParameterEstimate
    eta: ParameterEstimate
    rho: ParameterEstimate
    xi0_level: ParameterEstimate
    as_of: _dt.datetime
    data_start: _dt.datetime
    data_end: _dt.datetime
    jumps_detected: bool = False
    diagnostics: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, est in (
            ("hurst", self.hurst),
            ("eta", self.eta),
            ("rho", self.rho),
            ("xi0_level", self.xi0_level),
        ):
            if not isinstance(est, ParameterEstimate):
                raise ValidationError(
                    f"{name} must be a ParameterEstimate", context={"type": type(est).__name__}
                )
        if self.data_end < self.data_start:
            raise ValidationError(
                "data_end must not precede data_start",
                context={"start": self.data_start.isoformat(), "end": self.data_end.isoformat()},
            )
        # Enforce the validity domain of the model (mirrors RBergomiParams).
        if not (0.0 < self.hurst.value < 0.5):
            raise ValidationError(
                "calibrated hurst outside (0, 0.5)", context={"hurst": self.hurst.value}
            )
        if self.eta.value <= 0.0:
            raise ValidationError(
                "calibrated eta must be positive", context={"eta": self.eta.value}
            )
        if not (-1.0 <= self.rho.value <= 1.0):
            raise ValidationError(
                "calibrated rho outside [-1, 1]", context={"rho": self.rho.value}
            )
        if self.xi0_level.value <= 0.0:
            raise ValidationError(
                "calibrated xi0 level must be positive", context={"xi0": self.xi0_level.value}
            )

    def is_stale(self, *, now: _dt.datetime, max_age: _dt.timedelta) -> bool:
        """Return True if the calibration is older than ``max_age`` relative to ``now``."""
        return (now - self.as_of) > max_age
