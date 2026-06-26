"""Neural distribution surrogate with Monte-Carlo fallback (SPEC §2.5).

A monotone quantile network learns to approximate the rBergomi Monte-Carlo terminal
distribution for fast in-the-loop evaluation, guarded at runtime by a Wasserstein-based
fallback to direct Monte Carlo.

Public surface:

* Representation: :class:`SurrogateDistribution`.
* Features: :class:`RawInputs`, :class:`FeatureScaler`, :func:`build_feature_matrix`.
* Model: :class:`MonotoneQuantileNetwork`, :func:`pinball_loss`.
* Data: :class:`ScenarioRanges`, :class:`TrainingData`, :func:`generate_training_data`,
  :func:`default_quantile_levels`.
* Training/serving: :class:`DistributionSurrogate`, :class:`TrainingConfig`,
  :class:`TrainingReport`.
* Metrics: :func:`crps_from_quantiles`, :func:`wasserstein1_from_quantiles`,
  :func:`calibration_error`, :func:`pit_values`, :func:`quantile_loss_numpy`.
* Guardrail: :class:`SurrogateGuardrail`, :class:`GuardrailConfig`,
  :class:`GuardrailDecision`.
"""

from __future__ import annotations

from .dataset import (
    ScenarioRanges,
    TrainingData,
    default_quantile_levels,
    generate_training_data,
)
from .distribution import SurrogateDistribution
from .features import FeatureScaler, RawInputs, build_feature_matrix
from .guardrail import GuardrailConfig, GuardrailDecision, SurrogateGuardrail
from .losses import pinball_loss
from .metrics import (
    calibration_error,
    crps_from_quantiles,
    pit_values,
    quantile_loss_numpy,
    wasserstein1_from_quantiles,
)
from .quantile_network import MonotoneQuantileNetwork
from .surrogate import DistributionSurrogate, TrainingConfig, TrainingReport

__all__ = [
    "DistributionSurrogate",
    "FeatureScaler",
    "GuardrailConfig",
    "GuardrailDecision",
    "MonotoneQuantileNetwork",
    "RawInputs",
    "ScenarioRanges",
    "SurrogateDistribution",
    "SurrogateGuardrail",
    "TrainingConfig",
    "TrainingData",
    "TrainingReport",
    "build_feature_matrix",
    "calibration_error",
    "crps_from_quantiles",
    "default_quantile_levels",
    "generate_training_data",
    "pinball_loss",
    "pit_values",
    "quantile_loss_numpy",
    "wasserstein1_from_quantiles",
]
