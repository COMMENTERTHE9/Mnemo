"""Gate 0 train/eval loop + report. torch imported here only.

Baseline ladder for audio_dbfs_avg masked-node reconstruction:
  rung1 mean, rung2 linear, rung3 pooled-MLP, rung4 tiny-transformer.
Smoke test at 10 videos — debug the harness, do NOT treat the number as
evidence.
"""
from __future__ import annotations
import copy

import numpy as np
import torch
import torch.nn as nn

from mnemo.gate0.data import build_dataset, Example
from mnemo.gate0.models import (
    MeanPredictor, Linear, PooledMLP, TinyTransformer, count_params,
)

EPOCHS = 300
PATIENCE = 30
LR = 1e-3
SEEDS = [0, 1, 2]


def _labels(examples: list[Example]) -> np.ndarray:
    return np.array([e.label for e in examples], dtype=np.float64)


def _stack(examples: list[Example], attr: str) -> torch.Tensor:
    return torch.tensor(np.stack([getattr(e, attr) for e in examples]),
                        dtype=torch.float32)


def _pad_tokens(examples: list[Example]):
    n_max = max(e.tokens.shape[0] for e in examples)
    dim = examples[0].tokens.shape[1]
    toks = torch.zeros(len(examples), n_max, dim, dtype=torch.float32)
    pad_mask = torch.ones(len(examples), n_max, dtype=torch.bool)  # True == pad
    qidx = torch.zeros(len(examples), dtype=torch.long)
    for i, e in enumerate(examples):
        n = e.tokens.shape[0]
        toks[i, :n] = torch.tensor(e.tokens)
        pad_mask[i, :n] = False
        qidx[i] = e.query_idx
    return toks, qidx, pad_mask


def _forward(model, kind: str, packs) -> torch.Tensor:
    if kind == "linear":
        return model(packs["query_row"])
    if kind == "pooled":
        return model(packs["query_row"], packs["mean_pool"], packs["max_pool"])
    if kind == "transformer":
        return model(packs["tokens"], packs["query_idx"], packs["pad_mask"])
    raise ValueError(kind)


def _build_packs(examples: list[Example]) -> dict:
    toks, qidx, pad_mask = _pad_tokens(examples)
    return {
        "query_row": _stack(examples, "query_row"),
        "mean_pool": _stack(examples, "mean_pool"),
        "max_pool": _stack(examples, "max_pool"),
        "tokens": toks, "query_idx": qidx, "pad_mask": pad_mask,
    }


def _eval_mae(model, kind, packs, true_labels, lmean, lstd) -> tuple[np.ndarray, float]:
    model.eval()
    with torch.no_grad():
        pred_std = _forward(model, kind, packs).numpy()
    pred = pred_std * lstd + lmean
    mae = float(np.mean(np.abs(pred - true_labels)))
    return pred, mae


def _r2(pred: np.ndarray, true: np.ndarray) -> float:
    ss_res = float(np.sum((pred - true) ** 2))
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def _make_model(kind: str):
    return {"linear": Linear, "pooled": PooledMLP, "transformer": TinyTransformer}[kind]()


def train_rung(kind: str, data: dict, seed: int):
    """Train one torch rung; early-stop on val MAE (dBFS). Returns
    (test_pred, test_mae, params)."""
    torch.manual_seed(seed)
    model = _make_model(kind)
    params = count_params(model)

    train_y = _labels(data["train"])
    lmean, lstd = float(train_y.mean()), float(train_y.std() or 1.0)
    y_train = torch.tensor((train_y - lmean) / lstd, dtype=torch.float32)

    train_packs = _build_packs(data["train"])
    val_packs = _build_packs(data["val"])
    test_packs = _build_packs(data["test"])
    val_y = _labels(data["val"])
    test_y = _labels(data["test"])

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    bad = 0
    for _ in range(EPOCHS):
        model.train()
        opt.zero_grad()
        pred = _forward(model, kind, train_packs)
        loss = loss_fn(pred, y_train)
        loss.backward()
        opt.step()

        _, val_mae = _eval_mae(model, kind, val_packs, val_y, lmean, lstd)
        if val_mae < best_val - 1e-6:
            best_val = val_mae
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break

    model.load_state_dict(best_state)
    test_pred, test_mae = _eval_mae(model, kind, test_packs, test_y, lmean, lstd)
    return test_pred, test_mae, params


def run(corpus_dir: str = "corpus") -> dict:
    data = build_dataset(corpus_dir)
    test_y = _labels(data["test"])

    # Rung 1: mean predictor (no seeds — deterministic).
    mp = MeanPredictor().fit(_labels(data["train"]))
    r1_pred = mp.predict(len(test_y))
    r1_mae = float(np.mean(np.abs(r1_pred - test_y)))

    results = {"rung1": {"mae_mean": r1_mae, "mae_std": 0.0, "params": 0}}
    rung4_preds = []
    params_by_rung = {}

    for kind, name in [("linear", "rung2"), ("pooled", "rung3"),
                       ("transformer", "rung4")]:
        maes, preds = [], []
        params = 0
        for seed in SEEDS:
            pred, mae, params = train_rung(kind, data, seed)
            maes.append(mae)
            preds.append(pred)
        results[name] = {"mae_mean": float(np.mean(maes)),
                         "mae_std": float(np.std(maes)), "params": params}
        params_by_rung[name] = params
        if name == "rung4":
            rung4_preds = preds

    # Rung4 R^2 (avg prediction across seeds) + per-test-video MAE.
    r4_avg = np.mean(np.stack(rung4_preds), axis=0)
    r4_r2 = _r2(r4_avg, test_y)
    per_video: dict[str, float] = {}
    vids = [e.video_id for e in data["test"]]
    for vid in dict.fromkeys(vids):
        idx = [i for i, v in enumerate(vids) if v == vid]
        per_video[vid] = float(np.mean(np.abs(r4_avg[idx] - test_y[idx])))

    return {
        "split": (data["train_ids"], data["val_ids"], data["test_ids"]),
        "counts": (len(data["train"]), len(data["val"]), len(data["test"])),
        "results": results, "r4_r2": r4_r2, "per_video": per_video,
    }


def _ladder(r: dict) -> dict:
    m = {k: r["results"][k]["mae_mean"] for k in r["results"]}
    return {
        "r4<r1": m["rung4"] < m["rung1"],
        "r4<r2": m["rung4"] < m["rung2"],
        "r4<r3": m["rung4"] < m["rung3"],
        "r3<r2": m["rung3"] < m["rung2"],
    }


def print_report(r: dict) -> None:
    tr, va, te = r["split"]
    a, b, c = r["counts"]
    res = r["results"]
    disjoint = "Y" if not (set(tr) & set(va) or set(tr) & set(te) or set(va) & set(te)) else "N"
    print("GATE 0 — audio_dbfs_avg masked reconstruction (10 vids — harness validation, NOT evidence)")
    print(f"SPLIT: train={tr} val={va} test={te}  (disjoint: {disjoint})")
    print(f"EXAMPLES: train/val/test = {a}/{b}/{c}     "
          f"PARAMS: r2={res['rung2']['params']} r3={res['rung3']['params']} r4={res['rung4']['params']}")
    print("TEST MAE dBFS (mean±std over 3 seeds):")
    print(f"  rung1 mean       : {res['rung1']['mae_mean']:.3f}")
    print(f"  rung2 linear     : {res['rung2']['mae_mean']:.3f} ± {res['rung2']['mae_std']:.3f}")
    print(f"  rung3 pooledMLP  : {res['rung3']['mae_mean']:.3f} ± {res['rung3']['mae_std']:.3f}")
    print(f"  rung4 transformer: {res['rung4']['mae_mean']:.3f} ± {res['rung4']['mae_std']:.3f}")
    print(f"TEST R^2 rung4: {r['r4_r2']:.3f}")
    pv = " ".join(f"{k}={v:.3f}" for k, v in r["per_video"].items())
    print(f"PER-TEST-VIDEO MAE (rung4): {pv}")
    lad = _ladder(r)
    print(f"LADDER: rung4<rung1? {'Y' if lad['r4<r1'] else 'N'}  "
          f"rung4<rung2? {'Y' if lad['r4<r2'] else 'N'}  "
          f"rung4<rung3? {'Y' if lad['r4<r3'] else 'N'}  "
          f"rung3<rung2? {'Y' if lad['r3<r2'] else 'N'}")


if __name__ == "__main__":
    print_report(run())
