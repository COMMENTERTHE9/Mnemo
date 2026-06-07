"""Ternary QAT building blocks + the gate-3 masked-verbalization model.

TernaryLinear: shadow FP master weight; forward quantizes to {-gamma,0,+gamma}
with gamma = mean(|W|) per tensor and a straight-through estimator to the shadow.
Biases and activations are FP (declared scope: weights-only). A custom
(fast-path-free) transformer encoder and an explicit GRU built from
TernaryLinear let us ternarize exactly the QKV/out/FFN/GRU matrices while keeping
the FP-kept list (input projection, token embedding, LayerNorms, vocab head, all
biases) in FP. The FP baseline uses the SAME architecture with quantize=False,
so iso-shape is a clean quantization-only comparison.
"""
from __future__ import annotations
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def ternarize(w: torch.Tensor) -> torch.Tensor:
    """{-gamma,0,+gamma} with gamma=mean(|w|); straight-through to the shadow."""
    gamma = w.abs().mean()
    if float(gamma.detach()) == 0.0:
        return w
    t = torch.clamp(torch.round(w / gamma), -1.0, 1.0) * gamma
    return w + (t - w).detach()  # STE: forward=t, backward=identity


class TernaryLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 quantize: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.quantize = quantize
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(in_features)
            nn.init.uniform_(self.bias, -bound, bound)

    def effective_weight(self) -> torch.Tensor:
        return ternarize(self.weight) if self.quantize else self.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.effective_weight(), self.bias)


class TernaryMHA(nn.Module):
    def __init__(self, d_model: int, nhead: int, quantize: bool) -> None:
        super().__init__()
        assert d_model % nhead == 0
        self.h = nhead
        self.dh = d_model // nhead
        self.q = TernaryLinear(d_model, d_model, quantize=quantize)
        self.k = TernaryLinear(d_model, d_model, quantize=quantize)
        self.v = TernaryLinear(d_model, d_model, quantize=quantize)
        self.o = TernaryLinear(d_model, d_model, quantize=quantize)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor | None) -> torch.Tensor:
        b, n, d = x.shape

        def split(t):
            return t.view(b, n, self.h, self.dh).transpose(1, 2)  # b,h,n,dh
        q, k, v = split(self.q(x)), split(self.k(x)), split(self.v(x))
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)  # b,h,n,n
        if pad_mask is not None:
            scores = scores.masked_fill(pad_mask[:, None, None, :], float("-inf"))
        attn = scores.softmax(-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(b, n, d)
        return self.o(out)


class TernaryEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, ff, quantize, dropout=0.1) -> None:
        super().__init__()
        self.mha = TernaryMHA(d_model, nhead, quantize)
        self.lin1 = TernaryLinear(d_model, ff, quantize=quantize)
        self.lin2 = TernaryLinear(ff, d_model, quantize=quantize)
        self.norm1 = nn.LayerNorm(d_model)  # FP-kept
        self.norm2 = nn.LayerNorm(d_model)  # FP-kept
        self.drop = nn.Dropout(dropout)

    def forward(self, x, pad_mask):
        x = self.norm1(x + self.drop(self.mha(x, pad_mask)))
        ff = self.lin2(self.drop(F.relu(self.lin1(x))))
        return self.norm2(x + self.drop(ff))


class TernaryEncoder(nn.Module):
    def __init__(self, d_model, nhead, ff, layers, quantize) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [TernaryEncoderLayer(d_model, nhead, ff, quantize) for _ in range(layers)])

    def forward(self, x, pad_mask):
        for layer in self.layers:
            x = layer(x, pad_mask)
        return x


class TernaryGRU(nn.Module):
    """Explicit single-layer GRU built from TernaryLinear; matches nn.GRU math
    (batch_first) exactly when quantize=False and weights are copied."""

    def __init__(self, in_size, hidden, quantize) -> None:
        super().__init__()
        self.hidden = hidden
        self.ih = TernaryLinear(in_size, 3 * hidden, quantize=quantize)
        self.hh = TernaryLinear(hidden, 3 * hidden, quantize=quantize)

    def _step(self, x_t, h):
        gi = self.ih(x_t)
        gh = self.hh(h)
        i_r, i_z, i_n = gi.chunk(3, -1)
        h_r, h_z, h_n = gh.chunk(3, -1)
        r = torch.sigmoid(i_r + h_r)
        z = torch.sigmoid(i_z + h_z)
        n = torch.tanh(i_n + r * h_n)
        return (1 - z) * n + z * h

    def forward(self, x, h0):
        # x [B,N,I]; h0 [B,H]. Returns (outputs [B,N,H], h_final [B,H]).
        b, n, _ = x.shape
        h = h0
        outs = []
        for t in range(n):
            h = self._step(x[:, t], h)
            outs.append(h)
        return torch.stack(outs, dim=1), h


class Gate3Model(nn.Module):
    """Rung-1.5 masked-verbalization model with ternary internals. Same
    interface as gate2.TreeToText. FP-kept: proj, embedding, LayerNorms, head,
    all biases. Ternary: encoder QKV/out/FFN + GRU matrices."""

    def __init__(self, vocab_size, in_dim=21, d_model=32, nhead=2, layers=1,
                 ff=None, quantize=True, conditioned=True) -> None:
        super().__init__()
        ff = ff if ff is not None else 2 * d_model
        self.d_model = d_model
        self.conditioned = conditioned
        self.proj = nn.Linear(in_dim, d_model)               # FP-kept
        self.encoder = TernaryEncoder(d_model, nhead, ff, layers, quantize)
        self.const = nn.Parameter(torch.zeros(d_model))
        self.emb = nn.Embedding(vocab_size, d_model)         # FP-kept
        self.gru = TernaryGRU(d_model, d_model, quantize)
        self.head = nn.Linear(d_model, vocab_size)           # FP-kept

    def condition(self, tokens, query_idx, pad_mask):
        if not self.conditioned:
            return self.const.unsqueeze(0).expand(tokens.size(0), -1)
        h = self.encoder(self.proj(tokens), pad_mask)
        return h[torch.arange(h.size(0)), query_idx]

    def forward(self, tokens, query_idx, pad_mask, tgt_in):
        cond = self.condition(tokens, query_idx, pad_mask)
        emb = self.emb(tgt_in) + cond.unsqueeze(1)
        out, _ = self.gru(emb, cond)
        return self.head(out)

    @torch.no_grad()
    def generate(self, tokens, query_idx, pad_mask, bos, eos, max_len=20):
        self.eval()
        b = tokens.size(0)
        cond = self.condition(tokens, query_idx, pad_mask)
        h = cond
        cur = torch.full((b,), bos, dtype=torch.long)
        done = torch.zeros(b, dtype=torch.bool)
        seqs = [[] for _ in range(b)]
        for _ in range(max_len):
            h = self.gru._step(self.emb(cur) + cond, h)
            nxt = self.head(h).argmax(-1)
            for i in range(b):
                if not done[i]:
                    tok = int(nxt[i])
                    if tok == eos:
                        done[i] = True
                    else:
                        seqs[i].append(tok)
            if bool(done.all()):
                break
            cur = nxt
        return seqs


# ── accounting + audit ───────────────────────────────────────────────────────
def footprint(model: nn.Module, quantized: bool) -> dict:
    """Total bits: ternary weights at 2 bits (only if quantized), all FP-kept
    params + biases at 16 bits. quantized=False -> everything 16 bits."""
    ternary_params = fp_params = 0
    for m in model.modules():
        if isinstance(m, TernaryLinear):
            if quantized:
                ternary_params += m.weight.numel()
            else:
                fp_params += m.weight.numel()
            if m.bias is not None:
                fp_params += m.bias.numel()  # biases always FP
    # everything not inside a TernaryLinear
    seen = {id(p) for m in model.modules() if isinstance(m, TernaryLinear)
            for p in m.parameters()}
    for p in model.parameters():
        if id(p) not in seen:
            fp_params += p.numel()
    bits = (ternary_params * 2 if quantized else 0) + fp_params * 16
    return {"bits": bits, "ternary_params": ternary_params,
            "fp_params": fp_params,
            "total_params": ternary_params + fp_params}


def audit_fp_kept(model: Gate3Model) -> None:
    """Every Linear-like leaf is either a kept FP nn.Linear (proj, head) or a
    TernaryLinear; embeddings/norms are FP by type; biases are FP by construction."""
    kept = {id(model.proj), id(model.head)}
    for m in model.modules():
        if isinstance(m, nn.Linear):
            assert id(m) in kept, f"un-kept FP nn.Linear found: {m}"
        if isinstance(m, TernaryLinear):
            assert m.bias is None or m.bias.dtype == torch.float32
