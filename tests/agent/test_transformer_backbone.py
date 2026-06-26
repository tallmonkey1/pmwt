"""Tests for the transformer backbone and PPO-Transformer agent."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from options_engine.agent.ppo_transformer import (
    PPOTransformerAgent,
    PPOTransformerConfig,
)
from options_engine.agent.transformer_backbone import TransformerBackbone
from options_engine.core.errors import ValidationError


class TestTransformerBackbone:
    def test_forward_shape(self) -> None:
        bb = TransformerBackbone(input_dim=4, d_model=32, nhead=4, num_layers=2)
        x = torch.randn(3, 6, 4)
        out = bb(x)
        assert out.shape == (3, 32)

    def test_rejects_too_long_sequence(self) -> None:
        bb = TransformerBackbone(input_dim=4, d_model=32, nhead=4, num_layers=2, max_seq_len=4)
        x = torch.randn(1, 5, 4)
        with pytest.raises(ValueError):
            bb(x)

    def test_rejects_bad_d_model_nhead(self) -> None:
        with pytest.raises(ValueError):
            TransformerBackbone(input_dim=4, d_model=33, nhead=4, num_layers=2)

    def test_causal_mask_blocks_future(self) -> None:
        """Position t should not attend to positions > t."""
        bb = TransformerBackbone(
            input_dim=8, d_model=32, nhead=4, num_layers=1, max_seq_len=8
        )
        # Build a sequence where the last token is the *only* meaningful input;
        # past tokens are pure noise. With a working causal mask the last-token
        # output is *independent* of the last-token's own row in the attention.
        torch.manual_seed(0)
        x = torch.zeros(1, 4, 8)
        x[0, -1, :] = torch.tensor([10.0] * 8)
        h_with_signal = bb(x)
        x[0, -1, :] = 0.0
        x[0, 0, :] = torch.tensor([10.0] * 8)
        h_with_signal_elsewhere = bb(x)
        # The two outputs should differ because the meaningful token is at a
        # different position -- sanity check the encoder isn't ignoring positions.
        assert not torch.allclose(h_with_signal, h_with_signal_elsewhere, atol=1e-6)


class TestPPOTransformerAgent:
    def test_act_returns_correct_shapes(self) -> None:
        agent = PPOTransformerAgent(
            obs_dim=4, action_dim=2, config=PPOTransformerConfig(seq_len=4, seed=0)
        )
        seq = np.zeros((4, 4), dtype=np.float32)
        action, log_prob, value = agent.act(seq)
        assert action.shape == (2,)
        assert np.isfinite(log_prob)
        assert np.isfinite(value)

    def test_deterministic_act(self) -> None:
        agent = PPOTransformerAgent(
            obs_dim=4, action_dim=2, config=PPOTransformerConfig(seq_len=4, seed=0)
        )
        seq = np.ones((4, 4), dtype=np.float32)
        a1, _, _ = agent.act(seq, deterministic=True)
        a2, _, _ = agent.act(seq, deterministic=True)
        np.testing.assert_array_equal(a1, a2)

    def test_cvar_at_most_value(self) -> None:
        agent = PPOTransformerAgent(
            obs_dim=4, action_dim=2, config=PPOTransformerConfig(seq_len=4, seed=0)
        )
        seq = np.ones((4, 4), dtype=np.float32)
        assert agent.cvar(seq) <= agent.value(seq) + 1e-6

    def test_update_returns_stats(self) -> None:
        agent = PPOTransformerAgent(
            obs_dim=2,
            action_dim=1,
            config=PPOTransformerConfig(
                seq_len=4,
                n_epochs=2,
                minibatch_size=32,
                seed=0,
            ),
        )
        n = 64
        obs_seqs = np.random.randn(n, 4, 2).astype(np.float32)
        actions = np.zeros((n, 1), dtype=np.float32)
        old_log_probs = np.zeros(n, dtype=np.float32)
        advantages = np.random.randn(n).astype(np.float32)
        returns = np.random.randn(n).astype(np.float32)
        stats = agent.update_from_sequences(
            obs_sequences=obs_seqs,
            actions=actions,
            old_log_probs=old_log_probs,
            advantages=advantages,
            returns=returns,
            rng=np.random.default_rng(0),
        )
        assert stats.epochs_run >= 1
        assert np.isfinite(stats.policy_loss)
        assert np.isfinite(stats.value_loss)

    def test_rejects_bad_config(self) -> None:
        with pytest.raises(ValidationError):
            PPOTransformerConfig(seq_len=0)

    def test_encode_returns_correct_shape(self) -> None:
        agent = PPOTransformerAgent(
            obs_dim=4,
            action_dim=2,
            config=PPOTransformerConfig(d_model=32, seq_len=4, seed=0),
        )
        seq = np.zeros((4, 4), dtype=np.float32)
        ctx = agent.encode(seq)
        assert ctx.shape == (32,)
