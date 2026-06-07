"""Selection-movement retrodiction — calibrate the (falsified) 1a' commit_score
against a known-bad outcome.

Under plain naive training: compute the commit_score (importance x stability,
the exact 1a' machinery) on task 0, then measure how much each weight actually
MOVES while learning task 1. Spearman rho(commit_score, movement) tells us what
the score was selecting. The pre-registered read: the falsified score should
correlate POSITIVELY with task-1 movement (it froze the weights the next task
wanted to change) — the flunk that validates the diagnostic — while a random
score should correlate ~0.

torch imported here only.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from mnemo.gate1.methods import MLP, SI, BASE_LR, EPOCHS, BATCH
from mnemo.gate1.commitment import MOMENTUM, _rank_norm, EPS


def spearman(x: torch.Tensor, y: torch.Tensor) -> float:
    """Spearman rank correlation = Pearson on ordinal ranks (population-
    normalized; ordinal ties are fine for continuous weights)."""
    rx = x.argsort().argsort().to(torch.float64)
    ry = y.argsort().argsort().to(torch.float64)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = rx.norm() * ry.norm()
    return float((rx * ry).sum() / denom) if denom > 0 else 0.0


def _train_task(model, task, opt, gen, *, si=None, track_vol=False):
    x, y = task.train_x, task.train_y
    n = len(x)
    vol = ({nm: torch.zeros_like(p) for nm, p in model.named_parameters()}
           if track_vol else None)
    steps = 0
    for ep in range(EPOCHS):
        last = ep == EPOCHS - 1
        perm = torch.randperm(n, generator=gen).numpy()
        for s in range(0, n, BATCH):
            bi = perm[s:s + BATCH]
            opt.zero_grad()
            loss = F.cross_entropy(model(torch.tensor(x[bi])), torch.tensor(y[bi]))
            loss.backward()
            grads = ([p.grad.detach().clone() if p.grad is not None else None
                      for p in model.parameters()] if si is not None else None)
            wb = ({nm: p.detach().clone() for nm, p in model.named_parameters()}
                  if (track_vol and last) else None)
            opt.step()
            if si is not None:
                si.accumulate(model, grads)
            if track_vol and last:
                for nm, p in model.named_parameters():
                    vol[nm] += (p.detach() - wb[nm]).abs()
                steps += 1
    if si is not None:
        si.consolidate(model)
    if track_vol:
        vol = {nm: vol[nm] / max(1, steps) for nm in vol}
    return vol


def _commit_score(model, omega, vol):
    """The exact 1a' score: rank_norm(importance) * rank_norm(stability), per
    layer. (Here scored on all weights — naive has no committed weights.)"""
    score = {}
    for nm, _ in model.named_parameters():
        v = vol[nm].view(-1)
        stab = 1.0 / (1.0 + v / (v.median() + EPS))
        score[nm] = _rank_norm(omega[nm].view(-1)) * _rank_norm(stab)
    return score


def retrodiction(tasks, seeds=(0, 1, 2)) -> dict:
    per_layer: dict[str, list[float]] = {}
    global_rho: list[float] = []
    random_rho: list[float] = []
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = MLP()
        opt = torch.optim.SGD(model.parameters(), lr=BASE_LR, momentum=MOMENTUM)
        gen = torch.Generator().manual_seed(seed)
        si = SI(0.0, lambda m: list(m.parameters()))
        si.init(model)

        vol0 = _train_task(model, tasks[0], opt, gen, si=si, track_vol=True)
        omega = {nm: si.omega[k] for k, (nm, _) in enumerate(model.named_parameters())}
        score = _commit_score(model, omega, vol0)
        w_a = {nm: p.detach().clone() for nm, p in model.named_parameters()}

        _train_task(model, tasks[1], opt, gen, si=None, track_vol=False)

        gs, gm = [], []
        for nm, p in model.named_parameters():
            move = (p.detach() - w_a[nm]).abs().view(-1)
            per_layer.setdefault(nm, []).append(spearman(score[nm], move))
            gs.append(score[nm])
            gm.append(move)
        gscore = torch.cat(gs)
        gmove = torch.cat(gm)
        global_rho.append(spearman(gscore, gmove))
        rng = np.random.default_rng(seed)
        random_rho.append(spearman(torch.tensor(rng.random(gscore.numel())), gmove))

    def ms(xs):
        return float(np.mean(xs)), float(np.std(xs))

    return {
        "global": ms(global_rho),
        "random": ms(random_rho),
        "per_layer": {nm: ms(v) for nm, v in per_layer.items()},
    }


def print_report(res: dict) -> None:
    gm, gs = res["global"]
    rm, rs = res["random"]
    print("GATE 1 — selection-movement retrodiction (naive training, 3 seeds)")
    print(f"RETRODICTION: rho(score, movement) global = {gm:+.3f}±{gs:.3f}")
    print("  per-layer:")
    for nm, (m, s) in res["per_layer"].items():
        print(f"    {nm}: {m:+.3f}±{s:.3f}")
    print(f"  random control = {rm:+.3f}±{rs:.3f}")
    validated = gm > 0.05 and abs(rm) < 0.05
    print(f"DIAGNOSTIC VALIDATED (known-bad flunks positive, random ~0)? "
          f"{'Y' if validated else 'N'}")


if __name__ == "__main__":
    from mnemo.gate1.data import build_tasks
    print_report(retrodiction(build_tasks("eval")))
