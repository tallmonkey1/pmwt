"""Transformer encoder backbone for the PPO-Transformer agent (SPEC §4.2).

A small, from-scratch transformer encoder (multi-head *causal* self-attention +
position-wise feed-forward) gives the agent explicit memory over the recent
history of observations. The rBergomi model is **non-Markovian** -- the conditional
distribution of future variance depends on the path of past variance via the
Volterra kernel -- so an MLP that sees only the current observation cannot exploit
this dependency. The transformer fills exactly that gap.
"""

from __future__ import annotations

import math

import torch
from torch import nn

__all__ = ["SinusoidalPositionalEncoding", "TransformerBackbone"]


class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (Vaswani et al. 2017)."""

    def __init__(self, d_model: int, *, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerBackbone(nn.Module):
    """Causal transformer encoder producing a context vector from a feature sequence."""

    def __init__(
        self,
        *,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.0,
        max_seq_len: int = 64,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        if input_dim < 1 or d_model < 1 or nhead < 1 or num_layers < 1:
            raise ValueError(
                "input_dim, d_model, nhead and num_layers must all be >= 1"
            )
        if d_model % nhead != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by nhead ({nhead})"
            )
        if max_seq_len < 1 or dim_feedforward < 1:
            raise ValueError("max_seq_len and dim_feedforward must be >= 1")
        if activation not in ("relu", "gelu"):
            raise ValueError(
                f"activation must be 'relu' or 'gelu' (PyTorch Transformer constraint), got {activation!r}"
            )

        self.input_dim = input_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=max_seq_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation=activation,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self._causal_cache: dict[int, torch.Tensor] = {}

    def _causal_mask(self, length: int, device: torch.device) -> torch.Tensor:
        """Return the ``(length, length)`` causal attention mask (True = blocked)."""
        key = (length, device.type, device.index if device.index is not None else -1)
        cached = self._causal_cache.get(key)
        if cached is not None:
            return cached
        mask = torch.triu(torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1)
        self._causal_cache[key] = mask
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode ``(B, K, input_dim)`` -> a single context vector ``(B, d_model)``."""
        if x.dim() != 3:
            raise ValueError(
                f"TransformerBackbone expects (B, K, F), got shape {tuple(x.shape)}"
            )
        b, k, _ = x.shape
        if k > self.max_seq_len:
            raise ValueError(
                f"sequence length {k} exceeds max_seq_len {self.max_seq_len}"
            )
        h = self.input_proj(x)
        h = self.pos_enc(h)
        mask = self._causal_mask(k, x.device)
        h = self.encoder(h, mask=mask, is_causal=True)
        return h[:, -1, :]

    def encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for :meth:`forward`; returns ``(B, K, d_model)`` (all positions)."""
        if x.dim() != 3:
            raise ValueError(
                f"TransformerBackbone expects (B, K, F), got shape {tuple(x.shape)}"
            )
        k = x.size(1)
        if k > self.max_seq_len:
            raise ValueError(
                f"sequence length {k} exceeds max_seq_len {self.max_seq_len}"
            )
        h = self.input_proj(x)
        h = self.pos_enc(h)
        mask = self._causal_mask(k, x.device)
        return self.encoder(h, mask=mask, is_causal=True)
