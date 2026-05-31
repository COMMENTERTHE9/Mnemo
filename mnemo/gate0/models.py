"""Gate 0 model rungs. torch imported here only (never at package top level)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from mnemo.gate0.data import INPUT_DIM


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
    features in the tokens (level one-hot, start/end/span_rel, n_children)."""

    def __init__(self, in_dim: int = INPUT_DIM, d_model: int = 48,
                 nhead: int = 4, layers: int = 2, ff: int = 64) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ff,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.readout = nn.Linear(d_model, 1)

    def forward(self, tokens: torch.Tensor, query_idx: torch.Tensor,
                pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        # tokens [B, N, 21]; pad_mask [B, N] True == padding
        h = self.proj(tokens)
        h = self.encoder(h, src_key_padding_mask=pad_mask)
        q = h[torch.arange(h.size(0)), query_idx]  # gather query token
        return self.readout(q).squeeze(-1)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
