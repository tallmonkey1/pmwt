r"""Volatility-regime detector: HMM backbone + labeling + calibration + forecast.

This is the public regime-layer object (SPEC §2.6). It wraps the Gaussian HMM with the
domain semantics the trading layer needs:

* **Labeling** -- HMM states are unordered, so we map them to LOW / MID / HIGH by sorting on
  the learned realized-volatility level (feature 0). For ``K != 3`` states, states are
  grouped into thirds by rank; LOW is the calmest group, HIGH the most stressed.
* **Nowcast** -- the *filtered* (causal) posterior over regimes at the most recent step,
  using only information available now (leakage-free).
* **Next-step forecast** -- propagate the current filtered posterior one step through the
  transition matrix, then collapse to regime probabilities. This is the calibrated
  "next hour / next day" regime probability the trade gate consumes.
* **Calibration** -- an optional :class:`TemperatureScaler` fit on held-out data corrects
  over/under-confidence before the probabilities are used as a gate.

The detector exposes both the per-state HMM and the per-regime aggregation, and records the
state->regime mapping so results are fully interpretable and auditable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..core.enums import VolRegime
from ..core.errors import ModelStateError, ValidationError
from ..core.logging import get_logger
from ..core.random import RandomFactory
from .calibration import TemperatureScaler
from .hmm import GaussianHMM, HMMFitReport

__all__ = ["RegimeDetector", "RegimeNowcast"]

_logger = get_logger(__name__)

# Canonical regime ordering from calmest to most stressed.
_REGIME_ORDER: tuple[VolRegime, ...] = (VolRegime.LOW, VolRegime.MID, VolRegime.HIGH)


@dataclass(frozen=True, slots=True)
class RegimeNowcast:
    """A point-in-time regime assessment with current and next-step probabilities."""

    current_probabilities: dict[VolRegime, float]
    next_probabilities: dict[VolRegime, float]

    def most_likely_current(self) -> VolRegime:
        """Return the highest-probability current regime."""
        return max(self.current_probabilities, key=lambda r: self.current_probabilities[r])

    def current_prob(self, regime: VolRegime) -> float:
        """Return the current probability of a regime."""
        return self.current_probabilities.get(regime, 0.0)

    def next_prob(self, regime: VolRegime) -> float:
        """Return the next-step probability of a regime."""
        return self.next_probabilities.get(regime, 0.0)


class RegimeDetector:
    """Fits a Gaussian HMM and serves calibrated regime nowcasts/forecasts.

    Parameters
    ----------
    n_states:
        Number of HMM hidden states. Defaults to 3 (LOW/MID/HIGH). More states give finer
        resolution that is still grouped into three regimes for the gate.
    n_features:
        Regime feature dimension (see :func:`build_regime_features`).
    """

    def __init__(self, *, n_states: int = 3, n_features: int = 4) -> None:
        if n_states < 3:
            raise ValidationError(
                "n_states must be >= 3 to span low/mid/high regimes",
                context={"n_states": n_states},
            )
        self._hmm = GaussianHMM(n_states=n_states, n_features=n_features)
        self._state_to_regime: dict[int, VolRegime] | None = None
        self._calibrator: TemperatureScaler | None = None

    @property
    def is_fitted(self) -> bool:
        """True once the detector has been fit."""
        return self._hmm.is_fitted and self._state_to_regime is not None

    @property
    def hmm(self) -> GaussianHMM:
        """The underlying Gaussian HMM."""
        return self._hmm

    @property
    def state_to_regime(self) -> dict[int, VolRegime]:
        """Mapping from HMM state index to volatility regime."""
        if self._state_to_regime is None:
            raise ModelStateError("detector must be fitted first", context={})
        return dict(self._state_to_regime)

    def fit(
        self,
        features: NDArray[np.float64],
        *,
        rng_factory: RandomFactory,
        n_restarts: int = 5,
        max_iter: int = 200,
    ) -> HMMFitReport:
        """Fit the HMM and derive the state->regime labeling from the learned vol levels."""
        report = self._hmm.fit(
            features, rng_factory=rng_factory, n_restarts=n_restarts, max_iter=max_iter
        )
        self._state_to_regime = self._build_state_mapping(self._hmm.means[:, 0])
        _logger.info(
            "regime_detector_fitted",
            extra={"mapping": {int(k): v.value for k, v in self._state_to_regime.items()}},
        )
        return report

    def _build_state_mapping(self, vol_levels: NDArray[np.float64]) -> dict[int, VolRegime]:
        """Map states to LOW/MID/HIGH by tertiles of the learned vol level."""
        order = np.argsort(vol_levels)  # ascending: calmest first
        n = order.size
        mapping: dict[int, VolRegime] = {}
        for rank, state in enumerate(order):
            # Assign the calmest third to LOW, middle third to MID, top third to HIGH.
            tertile = min(2, (rank * 3) // n)
            mapping[int(state)] = _REGIME_ORDER[tertile]
        return mapping

    def _aggregate_to_regimes(self, state_probs: NDArray[np.float64]) -> dict[VolRegime, float]:
        """Sum per-state probabilities into per-regime probabilities."""
        assert self._state_to_regime is not None
        result = dict.fromkeys(_REGIME_ORDER, 0.0)
        for state, prob in enumerate(state_probs):
            result[self._state_to_regime[state]] += float(prob)
        return result

    def fit_calibration(
        self, features: NDArray[np.float64], regime_labels: NDArray[np.int_]
    ) -> None:
        """Fit the temperature calibrator on held-out (features, true-regime) data.

        ``regime_labels`` use the canonical ordering ``0=LOW, 1=MID, 2=HIGH``.
        """
        if not self.is_fitted:
            raise ModelStateError("fit the detector before calibration", context={})
        regime_probs = self._regime_posteriors(features)
        self._calibrator = TemperatureScaler().fit(regime_probs, regime_labels)

    def _regime_posteriors(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return smoothed per-regime posteriors for every row (N x 3)."""
        state_probs = self._hmm.predict_proba(features)
        out = np.zeros((state_probs.shape[0], len(_REGIME_ORDER)), dtype=np.float64)
        assert self._state_to_regime is not None
        for state in range(state_probs.shape[1]):
            regime_idx = _REGIME_ORDER.index(self._state_to_regime[state])
            out[:, regime_idx] += state_probs[:, state]
        return out

    def regime_posteriors(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        """Public: calibrated (if available) per-regime smoothed posteriors (N x 3)."""
        if not self.is_fitted:
            raise ModelStateError("detector must be fitted first", context={})
        probs = self._regime_posteriors(features)
        if self._calibrator is not None and self._calibrator.is_fitted:
            probs = self._calibrator.transform(probs)
        return probs

    def nowcast(self, features: NDArray[np.float64]) -> RegimeNowcast:
        """Return the leakage-free current + next-step regime assessment.

        Uses the *filtered* posterior at the final observation as the current nowcast, and
        propagates it one step through the transition matrix for the next-step forecast.
        """
        if not self.is_fitted:
            raise ModelStateError("detector must be fitted first", context={})
        filtered = self._hmm.filter_proba(features)
        current_state_probs = filtered[-1]  # (K,)

        # Propagate one step: next_state = current_state @ A.
        next_state_probs = current_state_probs @ self._hmm.transition_matrix

        current = self._aggregate_to_regimes(current_state_probs)
        nxt = self._aggregate_to_regimes(next_state_probs)

        if self._calibrator is not None and self._calibrator.is_fitted:
            current = self._calibrate_regime_dict(current)
            nxt = self._calibrate_regime_dict(nxt)

        return RegimeNowcast(current_probabilities=current, next_probabilities=nxt)

    def _calibrate_regime_dict(
        self, regime_probs: dict[VolRegime, float]
    ) -> dict[VolRegime, float]:
        """Apply the temperature calibrator to a single regime-probability dict."""
        assert self._calibrator is not None
        vector = np.array([[regime_probs[r] for r in _REGIME_ORDER]], dtype=np.float64)
        calibrated = self._calibrator.transform(vector)[0]
        return {r: float(calibrated[i]) for i, r in enumerate(_REGIME_ORDER)}
