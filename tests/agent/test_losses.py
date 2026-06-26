"""Tests for PPO and distributional-critic losses, validated against known properties."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from options_engine.agent.losses import ppo_clip_loss, quantile_huber_loss
from options_engine.agent.networks import quantile_fractions
from options_engine.core.errors import ValidationError


class TestPPOClipLoss:
    def test_loss_at_unit_ratio(self) -> None:
        # When new == old, ratio = 1, loss = -mean(advantages).
        new = torch.zeros(4)
        old = torch.zeros(4)
        adv = torch.tensor([1.0, 2.0, -1.0, 0.5])
        loss, clip_frac = ppo_clip_loss(
            new_log_probs=new, old_log_probs=old, advantages=adv, clip_epsilon=0.2
        )
        assert loss.item() == pytest.approx(-adv.mean().item())
        assert clip_frac.item() == 0.0

    def test_positive_advantage_is_clipped(self) -> None:
        # A large positive log-prob shift with positive advantage gets clipped at 1+eps.
        new = torch.tensor([5.0])
        old = torch.tensor([0.0])
        adv = torch.tensor([1.0])
        loss, clip_frac = ppo_clip_loss(
            new_log_probs=new, old_log_probs=old, advantages=adv, clip_epsilon=0.2
        )
        assert loss.item() == pytest.approx(-1.2)  # clipped surrogate = 1.2
        assert clip_frac.item() == 1.0

    def test_gradient_flows(self) -> None:
        new = torch.zeros(3, requires_grad=True)
        old = torch.zeros(3)
        adv = torch.tensor([1.0, -1.0, 0.5])
        loss, _ = ppo_clip_loss(
            new_log_probs=new, old_log_probs=old, advantages=adv, clip_epsilon=0.2
        )
        loss.backward()
        assert new.grad is not None

    def test_rejects_bad_epsilon(self) -> None:
        with pytest.raises(ValidationError):
            ppo_clip_loss(
                new_log_probs=torch.zeros(2),
                old_log_probs=torch.zeros(2),
                advantages=torch.zeros(2),
                clip_epsilon=1.5,
            )

    def test_rejects_shape_mismatch(self) -> None:
        with pytest.raises(ValidationError):
            ppo_clip_loss(
                new_log_probs=torch.zeros(2),
                old_log_probs=torch.zeros(3),
                advantages=torch.zeros(2),
                clip_epsilon=0.2,
            )


class TestQuantileHuberLoss:
    def test_zero_when_perfect(self) -> None:
        # If every quantile equals a constant target, the loss is ~0.
        taus = quantile_fractions(8)
        target = torch.full((5,), 2.0)
        pred = torch.full((5, 8), 2.0)
        loss = quantile_huber_loss(predicted_quantiles=pred, target_returns=target, taus=taus)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    @pytest.mark.slow
    def test_recovers_distribution_quantiles(self) -> None:
        # Optimizing the loss against a sample recovers the sample's quantiles.
        torch.manual_seed(0)
        k = 21
        taus = quantile_fractions(k)
        samples = torch.randn(4000)
        theta = torch.zeros(1, k, requires_grad=True)
        opt = torch.optim.Adam([theta], lr=0.05)
        for _ in range(600):
            opt.zero_grad()
            pred = theta.expand(samples.shape[0], k)
            loss = quantile_huber_loss(predicted_quantiles=pred, target_returns=samples, taus=taus)
            loss.backward()
            opt.step()
        learned = theta.detach().numpy().ravel()
        # Median quantile should be near 0; quantiles increasing.
        assert abs(learned[k // 2]) < 0.12
        assert np.all(np.diff(learned) >= -1e-2)

    def test_rejects_shape_mismatch(self) -> None:
        taus = quantile_fractions(4)
        with pytest.raises(ValidationError):
            quantile_huber_loss(
                predicted_quantiles=torch.zeros(3, 4), target_returns=torch.zeros(2), taus=taus
            )

    def test_rejects_bad_kappa(self) -> None:
        taus = quantile_fractions(4)
        with pytest.raises(ValidationError):
            quantile_huber_loss(
                predicted_quantiles=torch.zeros(2, 4),
                target_returns=torch.zeros(2),
                taus=taus,
                kappa=0.0,
            )
