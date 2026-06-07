"""Gate 1a' — learned per-weight commitment (plan §2b).

A per-weight commitment scalar c (binary {0,1}, never decays) provides two
coupled effects:
  - Protection : add lambda * sum_i c_i (w_i - a_i)^2 to the loss (anchor a_i).
  - Recruitment: grad *= (1 - c) elementwise before the step (per-weight LR
    gating). Under SGD+momentum this is made an exact freeze for c=1 weights by
    also zeroing their momentum at the boundary where they are committed.

At each task boundary, score the currently-UNCOMMITTED weights
(commit_score = rank_norm(importance) * rank_norm(stability), per layer), and
harden the top 5%-of-layer (subject to <=40% committed per layer): c<-1, a<-w.

  importance_i = SI path-integral omega (reuses methods.SI)
  stability_i  = 1 / (1 + vol_i / (layer-median vol + eps)),
                 vol_i = mean |delta w_i| over the LAST epoch of the task.

torch is imported here only (never at gate1 package top level).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from mnemo.gate1.methods import (
    MLP, EWC, SI, ReplayBuffer, accuracy, BASE_LR, EPOCHS, BATCH,
)

MOMENTUM = 0.9
HARDEN_FRAC = 0.05   # of layer, per boundary
COMMIT_CAP = 0.40    # max committed per layer
EPS = 1e-8


def _rank_norm(x: torch.Tensor) -> torch.Tensor:
    n = x.numel()
    if n <= 1:
        return torch.zeros_like(x)
    ranks = x.argsort().argsort().to(torch.float64)
    return (ranks / (n - 1)).to(x.dtype)


class CommitmentState:
    """Per-weight buffers c (commitment) and a (anchor), plus the protection
    penalty, the recruitment gate, and the boundary hardening rules."""

    def __init__(self, model: MLP, lam: float, uniform_c: float | None = None) -> None:
        self.lam = lam
        self.c: dict[str, torch.Tensor] = {}
        self.a: dict[str, torch.Tensor] = {}
        for nm, p in model.named_parameters():
            self.c[nm] = (torch.full_like(p, float(uniform_c)) if uniform_c is not None
                          else torch.zeros_like(p))
            self.a[nm] = p.detach().clone()
        self.harden_history: list[dict[str, int]] = []  # counts per boundary

    def penalty(self, model: MLP) -> torch.Tensor:
        tot = torch.zeros(())
        for nm, p in model.named_parameters():
            tot = tot + (self.c[nm] * (p - self.a[nm]) ** 2).sum()
        return self.lam * tot

    def gate_grads(self, model: MLP) -> None:
        with torch.no_grad():
            for nm, p in model.named_parameters():
                if p.grad is not None:
                    p.grad.mul_(1.0 - self.c[nm])

    def reanchor(self, model: MLP) -> None:
        for nm, p in model.named_parameters():
            self.a[nm] = p.detach().clone()

    def committed_fraction(self) -> float:
        tc = sum(int((c > 0).sum()) for c in self.c.values())
        tot = sum(c.numel() for c in self.c.values())
        return tc / tot

    def per_layer_committed(self) -> dict[str, float]:
        return {nm: float((c > 0).float().mean()) for nm, c in self.c.items()}

    def _budget(self, nm: str) -> tuple[int, torch.Tensor]:
        c = self.c[nm].view(-1)
        size = c.numel()
        unc = (c == 0)
        already = size - int(unc.sum())
        n = max(0, min(int(HARDEN_FRAC * size), int(COMMIT_CAP * size) - already,
                       int(unc.sum())))
        return n, unc

    def harden_topk(self, model: MLP, omega: dict[str, torch.Tensor],
                    vol: dict[str, torch.Tensor]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for nm, p in model.named_parameters():
            n, unc = self._budget(nm)
            counts[nm] = n
            if n == 0:
                continue
            c = self.c[nm].view(-1)
            a = self.a[nm].view(-1)
            w = p.detach().view(-1)
            unc_idx = unc.nonzero().view(-1)
            imp = omega[nm].view(-1)[unc_idx]
            v = vol[nm].view(-1)[unc_idx]
            med = v.median()
            stab = 1.0 / (1.0 + v / (med + EPS))
            score = _rank_norm(imp) * _rank_norm(stab)
            top = unc_idx[torch.topk(score, n).indices]
            c[top] = 1.0
            a[top] = w[top]
        self.harden_history.append(counts)
        return counts

    def harden_random(self, model: MLP, rng: np.random.Generator) -> dict[str, int]:
        counts: dict[str, int] = {}
        for nm, p in model.named_parameters():
            n, unc = self._budget(nm)
            counts[nm] = n
            if n == 0:
                continue
            c = self.c[nm].view(-1)
            a = self.a[nm].view(-1)
            w = p.detach().view(-1)
            unc_idx = unc.nonzero().view(-1).numpy()
            chosen = torch.tensor(rng.choice(unc_idx, size=n, replace=False))
            c[chosen] = 1.0
            a[chosen] = w[chosen]
        self.harden_history.append(counts)
        return counts

    def harden_lowest(self, model: MLP, scores: dict[str, torch.Tensor]) -> dict[str, int]:
        """Oracle selection: among uncommitted, harden the n with the LOWEST
        scores (e.g. lowest future movement). Same budget/cap as the others."""
        counts: dict[str, int] = {}
        for nm, p in model.named_parameters():
            n, unc = self._budget(nm)
            counts[nm] = n
            if n == 0:
                continue
            c = self.c[nm].view(-1)
            a = self.a[nm].view(-1)
            w = p.detach().view(-1)
            unc_idx = unc.nonzero().view(-1)
            sc = scores[nm].view(-1)[unc_idx]
            low = unc_idx[torch.topk(sc, n, largest=False).indices]
            c[low] = 1.0
            a[low] = w[low]
        self.harden_history.append(counts)
        return counts


def _zero_committed_momentum(opt, model: MLP, state: CommitmentState) -> None:
    """Make c=1 an exact freeze under momentum: zero velocity for committed
    weights (their gated grad is already 0, so they stay frozen thereafter)."""
    for nm, p in model.named_parameters():
        buf = opt.state.get(p, {}).get("momentum_buffer")
        if buf is not None:
            buf.mul_(1.0 - (state.c[nm] > 0).to(buf.dtype))


def train_arm(tasks, seed: int, *, penalty: bool, gating: bool,
              harden_mode: str | None, uniform_c: float | None = None,
              lam: float = 100.0, replay: bool = False, oracle_moves=None):
    """Train the T tasks sequentially under one commitment configuration.
    Returns (R matrix, CommitmentState).

    replay=True adds a 200/task ring buffer with 50/50 current/replay mixing.
    Replay batches flow through the gate like any batch — a c=1 weight stays
    frozen on replay gradients too (no special-casing).

    harden_mode='oracle' selects, at boundary i, the lowest-future-movement
    uncommitted weights from oracle_moves[i] (precomputed from a seed-matched
    naive run — perfect foresight). Byte-identical .2 mechanics otherwise."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MLP()
    state = CommitmentState(model, lam, uniform_c=uniform_c)
    opt = torch.optim.SGD(model.parameters(), lr=BASE_LR, momentum=MOMENTUM)
    shuffle_gen = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed + 7)
    replay_buf = ReplayBuffer() if replay else None
    replay_rng = np.random.default_rng(seed + 11)
    buf_rng = np.random.default_rng(seed + 13)
    si = SI(0.0, lambda m: list(m.parameters())) if harden_mode == "topk" else None
    if si is not None:
        si.init(model)

    T = len(tasks)
    R = np.zeros((T, T))
    names = [nm for nm, _ in model.named_parameters()]

    for i, task in enumerate(tasks):
        x, y = task.train_x, task.train_y
        n = len(x)
        vol_acc = {nm: torch.zeros_like(p) for nm, p in model.named_parameters()}
        vol_steps = 0
        for ep in range(EPOCHS):
            last = ep == EPOCHS - 1
            perm = torch.randperm(n, generator=shuffle_gen).numpy()
            for s in range(0, n, BATCH):
                bi = perm[s:s + BATCH]
                xb = torch.tensor(x[bi])
                yb = torch.tensor(y[bi])
                if replay_buf is not None and replay_buf.total() > 0:
                    rs = replay_buf.sample(len(bi), replay_rng)  # 50/50 mix
                    if rs is not None:
                        xb = torch.cat([xb, torch.tensor(rs[0])])
                        yb = torch.cat([yb, torch.tensor(rs[1])])
                opt.zero_grad()
                loss = F.cross_entropy(model(xb), yb)
                if penalty:
                    loss = loss + state.penalty(model)
                loss.backward()
                grads = ([pp.grad.detach().clone() if pp.grad is not None else None
                          for pp in model.parameters()] if si is not None else None)
                if gating:
                    state.gate_grads(model)
                wb = ({nm: p.detach().clone() for nm, p in model.named_parameters()}
                      if last else None)
                opt.step()
                if si is not None:
                    si.accumulate(model, grads)
                if last:
                    for nm, p in model.named_parameters():
                        vol_acc[nm] += (p.detach() - wb[nm]).abs()
                    vol_steps += 1
        if si is not None:
            si.consolidate(model)
        # task boundary (only between tasks: after the last task there is no
        # next task to protect, so no hardening/re-anchor happens there)
        boundary = i < T - 1
        if boundary and harden_mode == "topk":
            vol = {nm: vol_acc[nm] / max(1, vol_steps) for nm in vol_acc}
            omega = {nm: si.omega[k] for k, nm in enumerate(names)}
            state.harden_topk(model, omega, vol)
            _zero_committed_momentum(opt, model, state)
        elif boundary and harden_mode == "random":
            state.harden_random(model, rng)
            _zero_committed_momentum(opt, model, state)
        elif boundary and harden_mode == "oracle":
            state.harden_lowest(model, oracle_moves[i])
            _zero_committed_momentum(opt, model, state)
        elif boundary and uniform_c is not None:
            state.reanchor(model)
        if replay_buf is not None:
            replay_buf.add_task(x, y, buf_rng)
        for j, tj in enumerate(tasks):
            R[i, j] = accuracy(model, tj.test_x, tj.test_y)
    return R, state


def train_ewc(tasks, seed: int, lam: float) -> np.ndarray:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MLP()
    reg = EWC(lam, lambda m: list(m.parameters()))
    opt = torch.optim.SGD(model.parameters(), lr=BASE_LR, momentum=MOMENTUM)
    gen = torch.Generator().manual_seed(seed)
    T = len(tasks)
    R = np.zeros((T, T))
    for i, task in enumerate(tasks):
        x, y = task.train_x, task.train_y
        n = len(x)
        for _ in range(EPOCHS):
            perm = torch.randperm(n, generator=gen).numpy()
            for s in range(0, n, BATCH):
                bi = perm[s:s + BATCH]
                opt.zero_grad()
                loss = F.cross_entropy(model(torch.tensor(x[bi])), torch.tensor(y[bi]))
                loss = loss + reg.penalty(model)
                loss.backward()
                opt.step()
        reg.consolidate(model, x, y)
        for j, tj in enumerate(tasks):
            R[i, j] = accuracy(model, tj.test_x, tj.test_y)
    return R


def naive_snapshots(tasks, seed: int):
    """Plain naive training (SGD+momentum), byte-matching train_arm's naive
    trajectory, snapshotting weights after each task. Returns (R, future_moves)
    where future_moves[k][name] = |w_end - w_after_task_k| for k in 0..T-2
    (the oracle's perfect-foresight signal). Reads nothing but tasks+seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MLP()
    opt = torch.optim.SGD(model.parameters(), lr=BASE_LR, momentum=MOMENTUM)
    gen = torch.Generator().manual_seed(seed)
    T = len(tasks)
    R = np.zeros((T, T))
    snaps = []
    for i, task in enumerate(tasks):
        x, y = task.train_x, task.train_y
        n = len(x)
        for _ in range(EPOCHS):
            perm = torch.randperm(n, generator=gen).numpy()
            for s in range(0, n, BATCH):
                bi = perm[s:s + BATCH]
                opt.zero_grad()
                loss = F.cross_entropy(model(torch.tensor(x[bi])), torch.tensor(y[bi]))
                loss.backward()
                opt.step()
        snaps.append({nm: p.detach().clone() for nm, p in model.named_parameters()})
        for j, tj in enumerate(tasks):
            R[i, j] = accuracy(model, tj.test_x, tj.test_y)
    w_end = snaps[-1]
    future_moves = [{nm: (w_end[nm] - snaps[k][nm]).abs() for nm in w_end}
                    for k in range(T - 1)]
    return R, future_moves


def train_joint(tasks, seed: int) -> float:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MLP()
    ux = np.concatenate([t.train_x for t in tasks])
    uy = np.concatenate([t.train_y for t in tasks])
    n = len(ux)
    target = len(tasks) * EPOCHS * int(np.ceil(len(tasks[0].train_x) / BATCH))
    opt = torch.optim.SGD(model.parameters(), lr=BASE_LR, momentum=MOMENTUM)
    gen = torch.Generator().manual_seed(seed)
    steps = 0
    while steps < target:
        perm = torch.randperm(n, generator=gen).numpy()
        for s in range(0, n, BATCH):
            if steps >= target:
                break
            bi = perm[s:s + BATCH]
            opt.zero_grad()
            loss = F.cross_entropy(model(torch.tensor(ux[bi])), torch.tensor(uy[bi]))
            loss.backward()
            opt.step()
            steps += 1
    return float(np.mean([accuracy(model, t.test_x, t.test_y) for t in tasks]))
