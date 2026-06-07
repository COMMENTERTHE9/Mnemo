"""Single-episode consolidation screen — the entry exam for any revived
commitment mechanism (design plan §2c).

Question: does per-weight commitment have a future when it acts at the
sleep/consolidation step instead of during live (wake) learning? The candidate
gate is the SAME importance×stability signal that died in the wake vehicle
(Gate 1a'), so this isolates signal-vs-vehicle.

  CORE  : train on task A; compute c on A (importance×stability, per-layer
          rank-normalized; top-20% -> c=1, one-shot, no boundaries).
  WAKE  : teacher = full copy of core, fine-tuned on B with RESTRICTED A-replay
          (ring buffer 200 A samples, 50/50 mixing) — the deployment regime.
  DISTILL (one event): student init = core(A); match the TEACHER's outputs
          (function-space soft-CE) on a 50/50 mix of B + the same 200 A-replay;
          per-weight absorption lr_i = lr*(1-c_i). Arms vary c only.
torch here only.
"""
from __future__ import annotations
import copy

import numpy as np
import torch
import torch.nn.functional as F

from mnemo.gate1.data import build_tasks
from mnemo.gate1.methods import MLP, SI, ReplayBuffer, accuracy, BASE_LR, EPOCHS, BATCH
from mnemo.gate1.commitment import MOMENTUM
from mnemo.gate1.diagnostic import _train_task, _commit_score

SEEDS = [0, 1, 2]
TOP_FRAC = 0.20
REPLAY_N = 200


def _gate_c(model, score, top_frac=TOP_FRAC) -> dict:
    """Per-layer top-`top_frac` of commit_score -> c=1, else 0."""
    c = {}
    for nm, p in model.named_parameters():
        s = score[nm].view(-1)
        k = int(top_frac * s.numel())
        ci = torch.zeros_like(s)
        if k > 0:
            ci[torch.topk(s, k).indices] = 1.0
        c[nm] = ci.view_as(p)
    return c


def _zero_c(model) -> dict:
    return {nm: torch.zeros_like(p) for nm, p in model.named_parameters()}


def _one_c(model) -> dict:
    return {nm: torch.ones_like(p) for nm, p in model.named_parameters()}


def _shuffle_c(c, rng: np.random.Generator) -> dict:
    """Same per-layer mass (count of 1s), random targets."""
    out = {}
    for nm, ci in c.items():
        flat = ci.view(-1)
        k = int(flat.sum())
        new = torch.zeros_like(flat)
        if k > 0:
            idx = rng.choice(flat.numel(), size=k, replace=False)
            new[torch.tensor(idx)] = 1.0
        out[nm] = new.view_as(ci)
    return out


def train_core(task_a, seed):
    """Train the core on A; return (core, commit_score) — depends on A only."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MLP()
    opt = torch.optim.SGD(model.parameters(), lr=BASE_LR, momentum=MOMENTUM)
    gen = torch.Generator().manual_seed(seed)
    si = SI(0.0, lambda m: list(m.parameters()))
    si.init(model)
    vol = _train_task(model, task_a, opt, gen, si=si, track_vol=True)
    omega = {nm: si.omega[k] for k, (nm, _) in enumerate(model.named_parameters())}
    return model, _commit_score(model, omega, vol)


def _epoch_batches(x, y, n, gen):
    perm = torch.randperm(n, generator=gen).numpy()
    for s in range(0, n, BATCH):
        yield perm[s:s + BATCH]


def train_teacher(core, task_a, task_b, seed):
    teacher = copy.deepcopy(core)
    opt = torch.optim.SGD(teacher.parameters(), lr=BASE_LR, momentum=MOMENTUM)
    gen = torch.Generator().manual_seed(seed + 100)
    buf = ReplayBuffer(per_task=REPLAY_N)
    buf.add_task(task_a.train_x, task_a.train_y, np.random.default_rng(seed))
    rrng = np.random.default_rng(seed + 1)
    x, y = task_b.train_x, task_b.train_y
    n = len(x)
    for _ in range(EPOCHS):
        for bi in _epoch_batches(x, y, n, gen):
            xb = torch.tensor(x[bi])
            yb = torch.tensor(y[bi])
            rs = buf.sample(len(bi), rrng)
            if rs is not None:
                xb = torch.cat([xb, torch.tensor(rs[0])])
                yb = torch.cat([yb, torch.tensor(rs[1])])
            teacher.train()
            opt.zero_grad()
            F.cross_entropy(teacher(xb), yb).backward()
            opt.step()
    return teacher, buf


def consolidate(core, teacher, c, task_b, buf, seed):
    """One distillation event: student(core) matches teacher outputs on
    B + A-replay, with per-weight absorption lr_i = lr*(1-c_i)."""
    student = copy.deepcopy(core)
    opt = torch.optim.SGD(student.parameters(), lr=BASE_LR, momentum=MOMENTUM)
    gen = torch.Generator().manual_seed(seed + 200)
    rrng = np.random.default_rng(seed + 2)
    x, y = task_b.train_x, task_b.train_y
    n = len(x)
    for _ in range(EPOCHS):
        for bi in _epoch_batches(x, y, n, gen):
            xb = torch.tensor(x[bi])
            rs = buf.sample(len(bi), rrng)
            if rs is not None:
                xb = torch.cat([xb, torch.tensor(rs[0])])
            with torch.no_grad():
                t_prob = teacher(xb).softmax(-1)
            student.train()
            opt.zero_grad()
            logp = F.log_softmax(student(xb), -1)
            loss = -(t_prob * logp).sum(-1).mean()  # function-space distillation
            loss.backward()
            with torch.no_grad():
                for nm, p in student.named_parameters():
                    if p.grad is not None:
                        p.grad.mul_(1.0 - c[nm])  # gated absorption
            opt.step()
    return student


def run_screen(corpus_dir=None) -> dict:
    tasks = build_tasks("eval")
    task_a, task_b = tasks[0], tasks[1]
    arms = ("ungated", "candidate", "shuffled", "frozen")
    out = {a: {"A": [], "B": []} for a in arms}
    out["teacher"] = {"A": [], "B": []}
    for seed in SEEDS:
        core, score = train_core(task_a, seed)
        teacher, buf = train_teacher(core, task_a, task_b, seed)
        out["teacher"]["A"].append(accuracy(teacher, task_a.test_x, task_a.test_y))
        out["teacher"]["B"].append(accuracy(teacher, task_b.test_x, task_b.test_y))
        c_cand = _gate_c(core, score)
        c_shuf = _shuffle_c(c_cand, np.random.default_rng(seed + 3))
        cmap = {"ungated": _zero_c(core), "candidate": c_cand,
                "shuffled": c_shuf, "frozen": _one_c(core)}
        for a in arms:
            st = consolidate(core, teacher, cmap[a], task_b, buf, seed)
            out[a]["A"].append(accuracy(st, task_a.test_x, task_a.test_y))
            out[a]["B"].append(accuracy(st, task_b.test_x, task_b.test_y))
    return out


def _ms(xs):
    return float(np.mean(xs)), float(np.std(xs))


def print_report(out: dict) -> None:
    print("GATE 1 — single-episode consolidation screen (candidate gate #1, 3 seeds)")
    ta, tb = _ms(out["teacher"]["A"])[0], _ms(out["teacher"]["B"])[0]
    print(f"  teacher A/B = {ta:.3f}/{tb:.3f}")
    for a in ("ungated", "candidate", "shuffled", "frozen"):
        print(f"  {a:9s} A/B = {_ms(out[a]['A'])[0]:.3f}/{_ms(out[a]['B'])[0]:.3f}")
    ua, ub = _ms(out["ungated"]["A"])[0], _ms(out["ungated"]["B"])
    ca, cb = _ms(out["candidate"]["A"])[0], _ms(out["candidate"]["B"])[0]
    sa, sb = _ms(out["shuffled"]["A"])[0], _ms(out["shuffled"]["B"])[0]
    eps_b = 2 * ub[1]
    dominates = (ca > sa + 0.005) and (cb >= sb - 0.005)
    b_ok = cb >= ub[0] - eps_b
    near_shuffled = abs(ca - sa) <= 0.01 and abs(cb - sb) <= 0.01
    capacity_tax = (ca < ua) and (cb < ub[0])
    if dominates and b_ok:
        verdict = ("PASS — vehicle was the problem; commitment revives at sleep-time, "
                   "a true entropy gate is worth building")
    elif near_shuffled:
        verdict = "no-signal — gate carries no weight-specific info at consolidation either"
    elif capacity_tax:
        verdict = "capacity-tax — candidate worse than ungated on both axes"
    else:
        verdict = "no clear pass (candidate does not dominate shuffled and/or exceeds B-cost)"
    print(f"  eps_B={eps_b:.3f}  candidate vs shuffled A: {ca:.3f} vs {sa:.3f}; "
          f"B: {cb:.3f} vs {sb:.3f}; vs ungated B {ub[0]:.3f}")
    print(f"VERDICT (pre-registered): {verdict}")


if __name__ == "__main__":
    print_report(run_screen())
