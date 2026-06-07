"""Gate 2 rung-1 decoder. torch here only.

Conditioned arm: gate0 TinyTransformer encoder over the video's node tokens ->
the query segment's contextual embedding -> GRU word-level decoder.
Unconditioned twin: identical decoder, but the condition vector is a learned
CONSTANT (sees no tree). Same decoder params and training budget; only the
source of the GRU's initial hidden state differs.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from mnemo.gate0.models import TinyTransformer


class TreeToText(nn.Module):
    def __init__(self, vocab_size: int, in_dim: int = 19, d_model: int = 32,
                 nhead: int = 2, layers: int = 1, ff: int = 64,
                 conditioned: bool = True) -> None:
        super().__init__()
        self.conditioned = conditioned
        self.d_model = d_model
        # Reused gate0 tree encoder (rel unused for rung 1 -> structure-blind).
        self.encoder = TinyTransformer(in_dim=in_dim, d_model=d_model,
                                       nhead=nhead, layers=layers, ff=ff)
        self.const = nn.Parameter(torch.zeros(d_model))  # condition when blind
        self.emb = nn.Embedding(vocab_size, d_model)
        self.gru = nn.GRU(d_model, d_model, batch_first=True)
        self.out = nn.Linear(d_model, vocab_size)

    def condition(self, tokens: torch.Tensor, query_idx: torch.Tensor,
                  pad_mask: torch.Tensor) -> torch.Tensor:
        if not self.conditioned:
            return self.const.unsqueeze(0).expand(tokens.size(0), -1)
        h = self.encoder.proj(tokens)
        h = self.encoder.encoder(h, src_key_padding_mask=pad_mask)  # rel=None
        return h[torch.arange(h.size(0)), query_idx]  # [B, d_model]

    def forward(self, tokens, query_idx, pad_mask, tgt_in) -> torch.Tensor:
        cond = self.condition(tokens, query_idx, pad_mask)  # [B, d]
        # Inject the condition at EVERY step (added to each token embedding) so
        # it cannot be forgotten over the sequence; init hidden = cond too.
        emb = self.emb(tgt_in) + cond.unsqueeze(1)          # [B, L, d]
        out, _ = self.gru(emb, cond.unsqueeze(0).contiguous())
        return self.out(out)                               # [B, L, vocab]

    @torch.no_grad()
    def generate(self, tokens, query_idx, pad_mask, bos: int, eos: int,
                 max_len: int = 20) -> list[list[int]]:
        self.eval()
        b = tokens.size(0)
        cond = self.condition(tokens, query_idx, pad_mask)
        h = cond.unsqueeze(0).contiguous()
        cur = torch.full((b, 1), bos, dtype=torch.long)
        done = torch.zeros(b, dtype=torch.bool)
        seqs: list[list[int]] = [[] for _ in range(b)]
        for _ in range(max_len):
            o, h = self.gru(self.emb(cur) + cond.unsqueeze(1), h)
            nxt = self.out(o[:, -1]).argmax(-1)  # [B]
            for i in range(b):
                if not done[i]:
                    tok = int(nxt[i])
                    if tok == eos:
                        done[i] = True
                    else:
                        seqs[i].append(tok)
            if bool(done.all()):
                break
            cur = nxt.unsqueeze(1)
        return seqs
