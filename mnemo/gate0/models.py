"""Gate 0 model rungs. torch imported here only (never at package top level)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from mnemo.gate0.data import INPUT_DIM

# PyTorch's eval-mode MultiheadAttention "fast path" mishandles our 3D float
# relational attention bias and returns NaN (training-mode slow path is fine).
# Disable it process-wide for this experiment module. The unbiased path is
# numerically unchanged (only slower); the biased path becomes correct.
if hasattr(torch.backends, "mha"):
    torch.backends.mha.set_fastpath_enabled(False)


class MeanPredictor:
    """Rung 1: predict the train-set mean label. No training, no torch."""

    def __init__(self) -> None:
        self.value = 0.0

    def fit(self, train_labels: np.ndarray) -> "MeanPredictor":
        self.value = float(np.mean(train_labels))
        return self

    def predict(self, n: int) -> np.ndarray:
        return np.full(n, self.value, dtype=np.float64)


class Linear(nn.Module):
    """Rung 2: linear map from the query node's own 21-dim row -> label."""

    def __init__(self, in_dim: int = INPUT_DIM) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, query_row: torch.Tensor) -> torch.Tensor:  # [B, 21]
        return self.fc(query_row).squeeze(-1)


class PooledMLP(nn.Module):
    """Rung 3: MLP over concat[query row, mean-pool, max-pool]. Permutation-
    invariant over nodes -> aggregate content, NOT topology."""

    def __init__(self, in_dim: int = INPUT_DIM, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim * 3, hidden), nn.ReLU(), nn.Linear(hidden, 1),
        )

    def forward(self, query_row: torch.Tensor, mean_pool: torch.Tensor,
                max_pool: torch.Tensor) -> torch.Tensor:
        x = torch.cat([query_row, mean_pool, max_pool], dim=-1)
        return self.net(x).squeeze(-1)


class TinyTransformer(nn.Module):
    """Rung 4: node tokens -> linear proj to d_model, 2-layer full self-
    attention, read out the is_query token. Topology-aware via the positional
    features in the tokens (level one-hot, start/end/span_rel, n_children) and,
    when a relation matrix is supplied, an explicit per-head per-relation
    additive attention bias (parent/child/sibling/self/other) shared across
    layers. The bias is the ONLY architectural change vs. the baseline; with
    rel=None the module behaves exactly as before."""

    N_RELATIONS = 5

    def __init__(self, in_dim: int = INPUT_DIM, d_model: int = 48,
                 nhead: int = 4, layers: int = 2, ff: int = 64) -> None:
        super().__init__()
        self.nhead = nhead
        self.proj = nn.Linear(in_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ff,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.readout = nn.Linear(d_model, 1)
        # Learned bias B[head, relation]; added to attention logits each layer.
        self.rel_bias = nn.Parameter(torch.zeros(nhead, self.N_RELATIONS))

    def forward(self, tokens: torch.Tensor, query_idx: torch.Tensor,
                pad_mask: torch.Tensor | None = None,
                rel: torch.Tensor | None = None) -> torch.Tensor:
        # tokens [B, N, in_dim]; pad_mask [B, N] True == padding;
        # rel [B, N, N] long relation ids, or None (no structural bias).
        h = self.proj(tokens)
        if rel is None:
            h = self.encoder(h, src_key_padding_mask=pad_mask)
        else:
            # bias[b, head, i, j] = rel_bias[head, rel[b, i, j]]
            bias = self.rel_bias[:, rel]                  # [head, B, N, N]
            bias = bias.permute(1, 0, 2, 3).contiguous()  # [B, head, N, N]
            b, hh, n, _ = bias.shape
            if pad_mask is not None:
                # Fold key-padding into the float mask and drop
                # src_key_padding_mask: mixing a 3D float attn_mask with a
                # separate key_padding_mask, and using -inf, drives the encoder's
                # eval-mode fast path to NaN. A large finite negative is softmax-
                # equivalent and fast-path safe.
                neg = torch.zeros(b, 1, 1, n, device=h.device)
                neg = neg.masked_fill(pad_mask[:, None, None, :], -1e9)
                bias = bias + neg                          # broadcast over heads/queries
            attn_mask = bias.view(b * hh, n, n)            # [B*head, N, N]
            h = self.encoder(h, mask=attn_mask, src_key_padding_mask=None)
        q = h[torch.arange(h.size(0)), query_idx]  # gather query token
        return self.readout(q).squeeze(-1)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
