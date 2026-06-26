"""Volatility-regime detection and the trade gate (SPEC §2.6).

A Gaussian hidden-Markov backbone produces calibrated low/mid/high-vol nowcasts and
next-step forecasts; the trade gate permits new short-premium risk only in
high-confidence calm conditions.

Public surface:

* HMM: :class:`GaussianHMM`, :class:`HMMFitReport`.
* Features: :func:`build_regime_features`, :data:`N_REGIME_FEATURES`.
* Detector: :class:`RegimeDetector`, :class:`RegimeNowcast`.
* Calibration: :class:`TemperatureScaler`.
* Metrics: :func:`brier_score`, :func:`log_loss`, :func:`expected_calibration_error`,
  :func:`reliability_curve`, :class:`ReliabilityCurve`.
* Gate: :func:`evaluate_regime_gate`, :class:`RegimeGateConfig`, :class:`RegimeGateDecision`.
"""

from __future__ import annotations

from .calibration import TemperatureScaler
from .detector import RegimeDetector, RegimeNowcast
from .features import N_REGIME_FEATURES, build_regime_features
from .gate import RegimeGateConfig, RegimeGateDecision, evaluate_regime_gate
from .hmm import GaussianHMM, HMMFitReport
from .metrics import (
    ReliabilityCurve,
    brier_score,
    expected_calibration_error,
    log_loss,
    reliability_curve,
)

__all__ = [
    "N_REGIME_FEATURES",
    "GaussianHMM",
    "HMMFitReport",
    "RegimeDetector",
    "RegimeGateConfig",
    "RegimeGateDecision",
    "RegimeNowcast",
    "ReliabilityCurve",
    "TemperatureScaler",
    "brier_score",
    "build_regime_features",
    "evaluate_regime_gate",
    "expected_calibration_error",
    "log_loss",
    "reliability_curve",
]
