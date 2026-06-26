"""Tests for probability calibration and skill metrics."""

from __future__ import annotations

import numpy as np
import pytest

from options_engine.core.errors import ModelStateError, ValidationError
from options_engine.regime.calibration import TemperatureScaler
from options_engine.regime.metrics import (
    brier_score,
    expected_calibration_error,
    log_loss,
    reliability_curve,
)


def _overconfident_probs(labels: np.ndarray, n_classes: int, sharpness: float) -> np.ndarray:
    """Build over-confident probabilities that still predict the right class often."""
    rng = np.random.default_rng(0)
    n = labels.size
    logits = rng.normal(0, 1.0, size=(n, n_classes))
    # Boost the true class to make predictions usually correct but over-sharp.
    logits[np.arange(n), labels] += sharpness
    logits *= sharpness  # exaggerate -> over-confident
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


class TestMetrics:
    def test_brier_zero_for_perfect(self) -> None:
        probs = np.array([[1.0, 0.0], [0.0, 1.0]])
        labels = np.array([0, 1])
        assert brier_score(probs, labels) == pytest.approx(0.0)

    def test_brier_worst_case(self) -> None:
        probs = np.array([[0.0, 1.0], [1.0, 0.0]])
        labels = np.array([0, 1])
        assert brier_score(probs, labels) == pytest.approx(2.0)

    def test_log_loss_zero_for_perfect(self) -> None:
        probs = np.array([[1.0, 0.0], [0.0, 1.0]])
        labels = np.array([0, 1])
        assert log_loss(probs, labels) == pytest.approx(0.0, abs=1e-9)

    def test_log_loss_uniform(self) -> None:
        # Uniform predictions give log(K) loss.
        probs = np.full((4, 3), 1 / 3)
        labels = np.array([0, 1, 2, 0])
        assert log_loss(probs, labels) == pytest.approx(np.log(3), abs=1e-9)

    def test_ece_zero_for_calibrated(self) -> None:
        # Perfectly calibrated, perfectly confident predictions.
        probs = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
        labels = np.array([0, 1, 0])
        assert expected_calibration_error(probs, labels) == pytest.approx(0.0)

    def test_reliability_curve_shape(self) -> None:
        rng = np.random.default_rng(1)
        probs = rng.dirichlet([1, 1, 1], size=200)
        labels = rng.integers(0, 3, size=200)
        curve = reliability_curve(probs, labels, class_index=0, n_bins=10)
        assert curve.bin_confidence.shape == (10,)
        assert curve.bin_count.sum() == 200

    def test_rejects_unnormalized_probs(self) -> None:
        with pytest.raises(ValidationError):
            brier_score(np.array([[0.6, 0.6]]), np.array([0]))

    def test_rejects_label_out_of_range(self) -> None:
        with pytest.raises(ValidationError):
            log_loss(np.array([[0.5, 0.5]]), np.array([2]))


class TestTemperatureScaler:
    def test_transform_before_fit_raises(self) -> None:
        with pytest.raises(ModelStateError):
            TemperatureScaler().transform(np.array([[0.5, 0.5]]))

    def test_improves_calibration_of_overconfident_model(self) -> None:
        rng = np.random.default_rng(2)
        labels = rng.integers(0, 3, size=2000)
        probs = _overconfident_probs(labels, 3, sharpness=2.5)
        before = log_loss(probs, labels)

        scaler = TemperatureScaler().fit(probs, labels)
        calibrated = scaler.transform(probs)
        after = log_loss(calibrated, labels)

        # Temperature scaling should not worsen log-loss and typically improves it.
        assert after <= before + 1e-6
        assert scaler.temperature_ is not None and scaler.temperature_ > 0.0

    def test_preserves_argmax(self) -> None:
        rng = np.random.default_rng(3)
        labels = rng.integers(0, 3, size=500)
        probs = _overconfident_probs(labels, 3, sharpness=2.0)
        scaler = TemperatureScaler().fit(probs, labels)
        calibrated = scaler.transform(probs)
        # Temperature scaling never changes the predicted class.
        np.testing.assert_array_equal(probs.argmax(1), calibrated.argmax(1))

    def test_state_dict_round_trip(self) -> None:
        rng = np.random.default_rng(4)
        labels = rng.integers(0, 2, size=300)
        probs = _overconfident_probs(labels, 2, sharpness=2.0)
        scaler = TemperatureScaler().fit(probs, labels)
        restored = TemperatureScaler.from_state_dict(scaler.state_dict())
        np.testing.assert_allclose(scaler.transform(probs), restored.transform(probs))
