"""Tests for the monotone quantile network and the pinball loss."""

from __future__ import annotations

import pytest
import torch

from options_engine.core.errors import ValidationError
from options_engine.surrogate.losses import pinball_loss
from options_engine.surrogate.quantile_network import MonotoneQuantileNetwork


class TestMonotoneQuantileNetwork:
    def test_output_shape(self) -> None:
        net = MonotoneQuantileNetwork(n_features=6, n_quantiles=20, hidden_sizes=(16,))
        out = net(torch.randn(8, 6))
        assert out.shape == (8, 20)

    def test_quantiles_are_monotone_for_all_inputs(self) -> None:
        # The structural guarantee: outputs never cross, for any input.
        net = MonotoneQuantileNetwork(n_features=6, n_quantiles=50, hidden_sizes=(32,))
        torch.manual_seed(0)
        out = net(torch.randn(100, 6) * 10.0)  # large, varied inputs
        diffs = out[:, 1:] - out[:, :-1]
        assert bool(torch.all(diffs > 0.0))

    def test_initialization_is_narrow(self) -> None:
        # Smart init should not produce an absurdly wide initial distribution.
        net = MonotoneQuantileNetwork(n_features=6, n_quantiles=99, hidden_sizes=(16,))
        with torch.no_grad():
            out = net(torch.zeros(1, 6))
        width = float(out[0, -1] - out[0, 0])
        assert width == pytest.approx(1.0, abs=0.2)

    def test_rejects_bad_dims(self) -> None:
        with pytest.raises(ValidationError):
            MonotoneQuantileNetwork(n_features=0, n_quantiles=10)
        with pytest.raises(ValidationError):
            MonotoneQuantileNetwork(n_features=6, n_quantiles=1)

    def test_rejects_wrong_feature_width(self) -> None:
        net = MonotoneQuantileNetwork(n_features=6, n_quantiles=10, hidden_sizes=(8,))
        with pytest.raises(ValidationError):
            net(torch.randn(4, 5))


class TestPinballLoss:
    def test_minimized_at_true_quantile(self) -> None:
        # For samples, the pinball loss at level tau is minimized by the empirical
        # tau-quantile. Check the loss is lower at the truth than away from it.
        torch.manual_seed(0)
        samples = torch.randn(10000)
        tau = torch.tensor([0.7])
        true_q = torch.quantile(samples, 0.7).reshape(1, 1)
        # Build (batch=samples, Q=1) by broadcasting: evaluate loss of a constant prediction.
        targets = samples.reshape(-1, 1)
        loss_true = pinball_loss(true_q.expand_as(targets), targets, tau)
        loss_off = pinball_loss((true_q + 0.5).expand_as(targets), targets, tau)
        assert float(loss_true) < float(loss_off)

    def test_symmetric_at_median(self) -> None:
        preds = torch.tensor([[0.0]])
        targets = torch.tensor([[1.0]])
        tau = torch.tensor([0.5])
        loss = pinball_loss(preds, targets, tau)
        # At the median, loss = 0.5 * |error|.
        assert float(loss) == pytest.approx(0.5)

    def test_reductions(self) -> None:
        preds = torch.zeros(3, 4)
        targets = torch.ones(3, 4)
        levels = torch.linspace(0.1, 0.9, 4)
        none = pinball_loss(preds, targets, levels, reduction="none")
        assert none.shape == (3, 4)
        assert float(pinball_loss(preds, targets, levels, reduction="sum")) == pytest.approx(
            float(none.sum())
        )

    def test_shape_mismatch_rejected(self) -> None:
        with pytest.raises(ValidationError):
            pinball_loss(torch.zeros(2, 3), torch.zeros(2, 4), torch.linspace(0.1, 0.9, 3))

    def test_levels_length_checked(self) -> None:
        with pytest.raises(ValidationError):
            pinball_loss(torch.zeros(2, 3), torch.zeros(2, 3), torch.linspace(0.1, 0.9, 4))
