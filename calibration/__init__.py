"""rBergomi parameter calibration (SPEC §2.4).

Layered, walk-forward calibration from observed prices (and optionally an implied-vol
surface):

* Realized-variance proxies: :func:`log_returns`, :func:`realized_variance`,
  :func:`bipower_variation`, :func:`daily_realized_variance`, :func:`log_variance_proxy`.
* Hurst exponent: :func:`estimate_hurst` (log-moment scaling).
* Vol-of-vol & correlation: :func:`estimate_eta`, :func:`estimate_rho`
  (simulation-based moment matching).
* Forward variance: :func:`estimate_xi0_level`, :func:`estimate_xi0_curve`.
* Jumps: :func:`bns_jump_test`, :class:`JumpTestResult`.
* Orchestration: :func:`calibrate_rbergomi`, :class:`CalibrationConfig`.
* Walk-forward: :func:`generate_windows`, :func:`run_walk_forward`, :class:`WalkForwardWindow`.
* Results: :class:`ParameterEstimate`, :class:`CalibrationResult`.
"""

from __future__ import annotations

from .calibrator import CalibrationConfig, calibrate_rbergomi
from .forward_variance import estimate_xi0_curve, estimate_xi0_level
from .hurst import estimate_hurst
from .jumps import JumpTestResult, bns_jump_test
from .realized import (
    bipower_variation,
    daily_realized_variance,
    log_returns,
    log_variance_proxy,
    realized_variance,
)
from .results import CalibrationResult, ParameterEstimate
from .vol_of_vol import estimate_eta, estimate_rho, structure_function
from .walk_forward import WalkForwardWindow, generate_windows, run_walk_forward

__all__ = [
    "CalibrationConfig",
    "CalibrationResult",
    "JumpTestResult",
    "ParameterEstimate",
    "WalkForwardWindow",
    "bipower_variation",
    "bns_jump_test",
    "calibrate_rbergomi",
    "daily_realized_variance",
    "estimate_eta",
    "estimate_hurst",
    "estimate_rho",
    "estimate_xi0_curve",
    "estimate_xi0_level",
    "generate_windows",
    "log_returns",
    "log_variance_proxy",
    "realized_variance",
    "run_walk_forward",
    "structure_function",
]
