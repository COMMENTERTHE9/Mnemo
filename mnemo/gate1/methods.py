"""Gate 1a continual-learning methods (torch here only).

Single shared 10-way head for ALL methods (domain-incremental; task id never
used at test). Same MLP 784-256-256-10, same SGD/epochs/batch. Methods:
naive, joint (oracle), replay, ewc, si, and the conservative `recipe`
(backbone EWC + reduced backbone LR + replay; full-LR penalty-free head).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_LR = 0.1
EPOCHS = 3
BATCH = 128
FISHER_SAMPLES = 1000
REPLAY_PER_TASK = 200
SI_DAMP = 0.1


class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU())
        self.head = nn.Linear(256, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))

    def backbone_parameters(self):
        return list(self.backbone.parameters())

    def head_parameters(self):
        return list(self.head.parameters())


# ── Regularizers ─────────────────────────────────────────────────────────────
class EWC:
    """Diagonal-Fisher EWC. Penalty = lambda * sum_tasks sum_i F_i (th_i - th*_i)^2.
    `params_fn(model)` selects which parameters are anchored (all, or backbone)."""

    def __init__(self, lam: float, params_fn) -> None:
        self.lam = lam
        self.params_fn = params_fn
        self.tasks: list[tuple[list[torch.Tensor], list[torch.Tensor]]] = []

    def penalty(self, model: nn.Module) -> torch.Tensor:
        if not self.tasks:
            return torch.zeros(())
        cur = self.params_fn(model)
        total = torch.zeros(())
        for theta_star, fisher in self.tasks:
            for p, ts, f in zip(cur, theta_star, fisher):
                total = total + (f * (p - ts) ** 2).sum()
        return self.lam * total

    def consolidate(self, model: nn.Module, x: np.ndarray, y: np.ndarray,
                    n_samples: int = FISHER_SAMPLES) -> None:
        params = self.params_fn(model)
        fisher = [torch.zeros_like(p) for p in params]
        idx = np.random.default_rng(0).choice(
            len(x), size=min(n_samples, len(x)), replace=False)
        model.eval()
        seen = 0
        for s in range(0, len(idx), BATCH):
            bi = idx[s:s + BATCH]
            xb = torch.tensor(x[bi])
            yb = torch.tensor(y[bi])
            model.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            for fi, p in zip(fisher, params):
                if p.grad is not None:
                    fi += p.grad.detach() ** 2 * len(bi)
            seen += len(bi)
        fisher = [fi / seen for fi in fisher]
        theta_star = [p.detach().clone() for p in params]
        self.tasks.append((theta_star, fisher))
        model.zero_grad()


class SI:
    """Zenke et al. Synaptic Intelligence (path-integral importance)."""

    def __init__(self, c: float, params_fn, xi: float = SI_DAMP) -> None:
        self.c = c
        self.params_fn = params_fn
        self.xi = xi
        self.omega = None
        self.w = None
        self.ref = None       # reference (last consolidated) params
        self.prev = None      # params at previous step

    def init(self, model: nn.Module) -> None:
        ps = self.params_fn(model)
        self.omega = [torch.zeros_like(p) for p in ps]
        self.w = [torch.zeros_like(p) for p in ps]
        self.ref = [p.detach().clone() for p in ps]
        self.prev = [p.detach().clone() for p in ps]

    def penalty(self, model: nn.Module) -> torch.Tensor:
        if self.omega is None:
            return torch.zeros(())
        ps = self.params_fn(model)
        total = torch.zeros(())
        for p, om, r in zip(ps, self.omega, self.ref):
            total = total + (om * (p - r) ** 2).sum()
        return self.c * total

    def accumulate(self, model: nn.Module, grads: list[torch.Tensor]) -> None:
        ps = self.params_fn(model)
        for i, p in enumerate(ps):
            if grads[i] is not None:
                self.w[i] += -grads[i] * (p.detach() - self.prev[i])
            self.prev[i] = p.detach().clone()

    def consolidate(self, model: nn.Module) -> None:
        ps = self.params_fn(model)
        for i, p in enumerate(ps):
            delta = p.detach() - self.ref[i]
            self.omega[i] += torch.clamp(self.w[i], min=0.0) / (delta ** 2 + self.xi)
            self.w[i] = torch.zeros_like(p)
            self.ref[i] = p.detach().clone()
            self.prev[i] = p.detach().clone()


class ReplayBuffer:
    """Ring buffer: each task contributes <= per_task samples (total bounded by
    per_task * n_tasks)."""

    def __init__(self, per_task: int = REPLAY_PER_TASK) -> None:
        self.per_task = per_task
        self.xs: list[np.ndarray] = []
        self.ys: list[np.ndarray] = []

    def add_task(self, x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> None:
        idx = rng.choice(len(x), size=min(self.per_task, len(x)), replace=False)
        self.xs.append(x[idx].copy())
        self.ys.append(y[idx].copy())

    def total(self) -> int:
        return sum(len(a) for a in self.xs)

    def sample(self, k: int, rng: np.random.Generator):
        if not self.xs:
            return None
        X = np.concatenate(self.xs)
        Y = np.concatenate(self.ys)
        idx = rng.choice(len(X), size=min(k, len(X)), replace=(len(X) < k))
        return X[idx], Y[idx]


# ── Training / evaluation ────────────────────────────────────────────────────
@torch.no_grad()
def accuracy(model: nn.Module, x: np.ndarray, y: np.ndarray) -> float:
    model.eval()
    preds = []
    for s in range(0, len(x), 512):
        preds.append(model(torch.tensor(x[s:s + 512])).argmax(1).numpy())
    return float((np.concatenate(preds) == y).mean())


def _opt_for(model: MLP, alpha: float):
    """SGD; backbone LR scaled by alpha (recipe), head always full LR."""
    if alpha == 1.0:
        return torch.optim.SGD(model.parameters(), lr=BASE_LR)
    return torch.optim.SGD([
        {"params": model.backbone_parameters(), "lr": alpha * BASE_LR},
        {"params": model.head_parameters(), "lr": BASE_LR},
    ])


def train_task(model, task, opt, *, reg=None, si=None, replay=None,
               replay_rng=None, buf_rng=None, shuffle_gen=None):
    x, y = task.train_x, task.train_y
    n = len(x)
    if si is not None and si.omega is None:
        si.init(model)
    for _ in range(EPOCHS):
        perm = torch.randperm(n, generator=shuffle_gen).numpy()
        for s in range(0, n, BATCH):
            bi = perm[s:s + BATCH]
            xb = torch.tensor(x[bi])
            yb = torch.tensor(y[bi])
            if replay is not None and replay.total() > 0:
                rs = replay.sample(len(bi), replay_rng)  # 50/50 current/replay
                if rs is not None:
                    xb = torch.cat([xb, torch.tensor(rs[0])])
                    yb = torch.cat([yb, torch.tensor(rs[1])])
            model.train()
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            if reg is not None:
                loss = loss + reg.penalty(model)
            if si is not None:
                loss = loss + si.penalty(model)
            loss.backward()
            grads = None
            if si is not None:
                grads = [p.grad.detach().clone() if p.grad is not None else None
                         for p in si.params_fn(model)]
            opt.step()
            if si is not None:
                si.accumulate(model, grads)
    if reg is not None:
        reg.consolidate(model, x, y)
    if si is not None:
        si.consolidate(model)
    if replay is not None:
        replay.add_task(x, y, buf_rng)


def run_sequential(method: str, tasks, seed: int, *, lam=100.0, c=1.0, alpha=1.0):
    """Train tasks in order; return R[i,j] = test acc on task j after task i."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MLP()
    T = len(tasks)
    R = np.zeros((T, T))
    shuffle_gen = torch.Generator().manual_seed(seed)
    replay_rng = np.random.default_rng(seed + 1)
    buf_rng = np.random.default_rng(seed + 2)

    reg = si = replay = None
    if method == "ewc":
        reg = EWC(lam, lambda m: list(m.parameters()))
        alpha = 1.0
    elif method == "si":
        si = SI(c, lambda m: list(m.parameters()))
        alpha = 1.0
    elif method == "replay":
        replay = ReplayBuffer()
        alpha = 1.0
    elif method == "recipe":
        reg = EWC(lam, lambda m: m.backbone_parameters())  # backbone only
        replay = ReplayBuffer()
        # alpha scales backbone LR (head full LR via _opt_for)
    # naive: no reg/replay, alpha=1

    opt = _opt_for(model, alpha)
    for i, task in enumerate(tasks):
        train_task(model, task, opt, reg=reg, si=si, replay=replay,
                   replay_rng=replay_rng, buf_rng=buf_rng, shuffle_gen=shuffle_gen)
        for j, tj in enumerate(tasks):
            R[i, j] = accuracy(model, tj.test_x, tj.test_y)
    return R


def run_joint(tasks, seed: int) -> float:
    """Oracle: train on the union with the SAME total gradient steps as the
    sequential methods get in sum; return final mean test acc over tasks."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MLP()
    ux = np.concatenate([t.train_x for t in tasks])
    uy = np.concatenate([t.train_y for t in tasks])
    n = len(ux)
    steps_per_epoch_seq = int(np.ceil(len(tasks[0].train_x) / BATCH))
    target_steps = len(tasks) * EPOCHS * steps_per_epoch_seq
    opt = torch.optim.SGD(model.parameters(), lr=BASE_LR)
    gen = torch.Generator().manual_seed(seed)
    steps = 0
    while steps < target_steps:
        perm = torch.randperm(n, generator=gen).numpy()
        for s in range(0, n, BATCH):
            if steps >= target_steps:
                break
            bi = perm[s:s + BATCH]
            model.train()
            opt.zero_grad()
            loss = F.cross_entropy(model(torch.tensor(ux[bi])), torch.tensor(uy[bi]))
            loss.backward()
            opt.step()
            steps += 1
    return float(np.mean([accuracy(model, t.test_x, t.test_y) for t in tasks]))
