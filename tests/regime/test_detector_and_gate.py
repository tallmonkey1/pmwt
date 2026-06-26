"""Tests for the regime detector, nowcast, and trade gate."""

from __future__ import annotations

import numpy as np
import pytest

from options_engine.core.enums import VolRegime
from options_engine.core.errors import ModelStateError, ValidationError
from options_engine.core.random import RandomFactory
from options_engine.regime.detector import RegimeDetector, RegimeNowcast
from options_engine.regime.features import build_regime_features
from options_engine.regime.gate import (
    RegimeGateConfig,
    evaluate_regime_gate,
)
from options_engine.regime.metrics import log_loss


def _returns_with_regimes(n: int = 4000, seed: int = 0):
    """Synthetic daily returns driven by sticky vol regimes; returns (rets, states)."""
    rng = np.random.default_rng(seed)
    vol = np.array([0.007, 0.018, 0.040])
    persistence = 0.99
    off = (1 - persistence) / 2
    a = np.array([[persistence, off, off], [off, persistence, off], [off, off, persistence]])
    states = np.zeros(n, dtype=int)
    for t in range(1, n):
        states[t] = rng.choice(3, p=a[states[t - 1]])
    rets = rng.normal(0, vol[states])
    return rets, states


class TestRegimeDetectorConstruction:
    def test_rejects_too_few_states(self) -> None:
        with pytest.raises(ValidationError):
            RegimeDetector(n_states=2, n_features=4)

    def test_use_before_fit_raises(self) -> None:
        det = RegimeDetector(n_states=3, n_features=4)
        with pytest.raises(ModelStateError):
            det.nowcast(np.zeros((10, 4)))


@pytest.fixture(scope="module")
def fitted_detector():
    """A detector fit once and shared across structural tests (keeps the suite fast)."""
    rets, _ = _returns_with_regimes(n=2000, seed=1)
    feats = build_regime_features(rets, window=10)
    det = RegimeDetector(n_states=3, n_features=feats.shape[1])
    det.fit(feats, rng_factory=RandomFactory(3), n_restarts=2, max_iter=60)
    return det, feats


class TestNowcastStructure:
    def test_nowcast_probabilities_sum_to_one(self, fitted_detector) -> None:
        det, feats = fitted_detector
        nc = det.nowcast(feats)
        assert sum(nc.current_probabilities.values()) == pytest.approx(1.0, abs=1e-6)
        assert sum(nc.next_probabilities.values()) == pytest.approx(1.0, abs=1e-6)

    def test_nowcast_covers_all_regimes(self, fitted_detector) -> None:
        det, feats = fitted_detector
        nc = det.nowcast(feats)
        assert set(nc.current_probabilities) == {VolRegime.LOW, VolRegime.MID, VolRegime.HIGH}

    def test_state_mapping_is_ordered(self, fitted_detector) -> None:
        det, _ = fitted_detector
        mapping = det.state_to_regime
        # The state with the lowest learned vol level must map to LOW, highest to HIGH.
        levels = det.hmm.means[:, 0]
        assert mapping[int(np.argmin(levels))] == VolRegime.LOW
        assert mapping[int(np.argmax(levels))] == VolRegime.HIGH


@pytest.mark.slow
class TestDetectorSkill:
    def test_high_accuracy_on_persistent_regimes(self) -> None:
        rets, states = _returns_with_regimes(n=6000, seed=2)
        window = 10
        feats = build_regime_features(rets, window=window)
        true = states[window:]
        det = RegimeDetector(n_states=3, n_features=feats.shape[1])
        det.fit(feats[:4500], rng_factory=RandomFactory(5), n_restarts=5, max_iter=150)
        post = det.regime_posteriors(feats[4500:])
        accuracy = float(np.mean(post.argmax(1) == true[4500:]))
        assert accuracy > 0.85

    def test_calibration_improves_log_loss(self) -> None:
        rets, states = _returns_with_regimes(n=6000, seed=3)
        window = 10
        feats = build_regime_features(rets, window=window)
        true = states[window:]
        det = RegimeDetector(n_states=3, n_features=feats.shape[1])
        det.fit(feats[:4000], rng_factory=RandomFactory(6), n_restarts=5, max_iter=150)
        before = log_loss(det.regime_posteriors(feats[5000:]), true[5000:])
        det.fit_calibration(feats[4000:5000], true[4000:5000])
        after = log_loss(det.regime_posteriors(feats[5000:]), true[5000:])
        assert after <= before + 1e-6


class TestRegimeGate:
    def _nowcast(self, low_now, low_next, high_now, high_next) -> RegimeNowcast:
        mid_now = max(0.0, 1.0 - low_now - high_now)
        mid_next = max(0.0, 1.0 - low_next - high_next)
        return RegimeNowcast(
            current_probabilities={
                VolRegime.LOW: low_now,
                VolRegime.MID: mid_now,
                VolRegime.HIGH: high_now,
            },
            next_probabilities={
                VolRegime.LOW: low_next,
                VolRegime.MID: mid_next,
                VolRegime.HIGH: high_next,
            },
        )

    def test_allows_in_confident_calm(self) -> None:
        nc = self._nowcast(0.85, 0.80, 0.05, 0.05)
        decision = evaluate_regime_gate(nc)
        assert decision.allow_new_risk

    def test_blocks_when_low_now_insufficient(self) -> None:
        nc = self._nowcast(0.50, 0.80, 0.05, 0.05)
        decision = evaluate_regime_gate(nc)
        assert not decision.allow_new_risk
        assert "low_now" in decision.reason

    def test_blocks_when_high_next_too_large(self) -> None:
        nc = self._nowcast(0.85, 0.80, 0.05, 0.30)
        decision = evaluate_regime_gate(nc)
        assert not decision.allow_new_risk
        assert "high_next" in decision.reason

    def test_custom_thresholds(self) -> None:
        nc = self._nowcast(0.70, 0.65, 0.10, 0.10)
        strict = RegimeGateConfig(min_low_now=0.9, min_low_next=0.9)
        assert not evaluate_regime_gate(nc, config=strict).allow_new_risk
        lenient = RegimeGateConfig(min_low_now=0.6, min_low_next=0.6)
        assert evaluate_regime_gate(nc, config=lenient).allow_new_risk

    def test_config_rejects_bad_probability(self) -> None:
        with pytest.raises(ValidationError):
            RegimeGateConfig(min_low_now=1.5)
