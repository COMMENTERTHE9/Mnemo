"""Gate 0 positive controls: identity floor + sibling-mean structure test.

Synthetic tasks that reuse the rung models and the train/eval/seed loop from
mnemo.gate0.run UNCHANGED — only the task/label/masking differ. The data
builders here are pure numpy; torch is pulled in lazily inside run_controls so
importing this module (e.g. from tests) stays torch-free.

Control A (identity floor): label = the node's own span_rel, input UNMASKED.
  Trivial signal — rungs 2-4 must crush the mean baseline. Plumbing check.
Control B (sibling-mean): append a per-node z ~ N(0,1); mask the query's own z;
  label = mean of the query's siblings' z. Global pooling sees ~the global
  mean (wrong set); only a topology-aware model wins. Structure discriminator.
"""
from __future__ import annotations

import numpy as np

from mnemo.model.featurize import load_corpus, FEATURE_NAMES, FeaturizedTree
from mnemo.gate0.data import (
    split_videos, fit_standardizer, Example, SPLIT_SEED, N_BASE,
)

SPAN_REL_IDX = FEATURE_NAMES.index("span_rel")
Z_COL = N_BASE              # z is appended right after the 19 base features
Z_RESAMPLES = 64


def _example(video_id: str, tokens: np.ndarray, qi: int,
             mask_flag: np.ndarray, is_q: np.ndarray, label: float) -> Example:
    tokens = tokens.astype(np.float32)
    return Example(
        video_id=video_id, tokens=tokens, query_idx=qi,
        audio_hidden=mask_flag.astype(np.float32), is_query=is_q.astype(np.float32),
        label=float(label), query_row=tokens[qi].copy(),
        mean_pool=tokens.mean(axis=0).astype(np.float32),
        max_pool=tokens.max(axis=0).astype(np.float32),
    )


# ── Control A ──────────────────────────────────────────────────────────────
def control_a_examples(ft: FeaturizedTree, mean: np.ndarray,
                       std: np.ndarray) -> list[Example]:
    """One example per node (any level), input unmasked, label = span_rel."""
    x_std = (ft.X - mean) / std
    n = len(ft.node_ids)
    out: list[Example] = []
    for i in range(n):
        is_q = np.zeros(n)
        is_q[i] = 1.0
        zero = np.zeros(n)
        tokens = np.concatenate([x_std, is_q[:, None], zero[:, None]], axis=1)
        out.append(_example(ft.video_id, tokens, i, zero, is_q,
                            float(ft.X[i, SPAN_REL_IDX])))
    return out


# ── Control B ──────────────────────────────────────────────────────────────
def siblings_of(parent_idx: list[int], i: int) -> list[int]:
    """Indices sharing i's parent (parent != root), excluding i."""
    p = parent_idx[i]
    if p == -1:
        return []
    return [j for j in range(len(parent_idx)) if j != i and parent_idx[j] == p]


def control_b_examples(ft: FeaturizedTree, mean: np.ndarray, std: np.ndarray,
                       z: np.ndarray) -> list[Example]:
    """For one z-draw: one example per eligible (has-sibling) query node.
    Token = [19 std features, z, is_query, z_hidden]; query's z masked to 0;
    label = mean of siblings' z (query excluded)."""
    x_std = (ft.X - mean) / std
    n = len(ft.node_ids)
    out: list[Example] = []
    for i in range(n):
        sib = siblings_of(ft.parent_idx, i)
        if not sib:
            continue
        z_masked = z.copy()
        z_masked[i] = 0.0
        is_q = np.zeros(n)
        is_q[i] = 1.0
        tokens = np.concatenate(
            [x_std, z_masked[:, None], is_q[:, None], is_q[:, None]], axis=1)
        out.append(_example(ft.video_id, tokens, i, is_q, is_q,
                            float(np.mean(z[sib]))))
    return out


# ── Dataset assembly ────────────────────────────────────────────────────────
def _split_and_standardize(corpus_dir: str, split_seed: int):
    trees = load_corpus(corpus_dir)
    ids = [t.video_id for t in trees]
    train_ids, val_ids, test_ids = split_videos(ids, seed=split_seed)
    mean, std = fit_standardizer(trees, train_ids)
    by_id = {t.video_id: t for t in trees}
    return by_id, train_ids, val_ids, test_ids, mean, std


def build_control_a(corpus_dir: str = "corpus", split_seed: int = SPLIT_SEED) -> dict:
    by_id, tr, va, te, mean, std = _split_and_standardize(corpus_dir, split_seed)

    def make(idset):
        out: list[Example] = []
        for vid in idset:
            out.extend(control_a_examples(by_id[vid], mean, std))
        return out

    return {"train": make(tr), "val": make(va), "test": make(te),
            "train_ids": tr, "val_ids": va, "test_ids": te, "in_dim": N_BASE + 2}


def build_control_b(corpus_dir: str = "corpus", split_seed: int = SPLIT_SEED,
                    k: int = Z_RESAMPLES, rng_seed: int = 0) -> dict:
    by_id, tr, va, te, mean, std = _split_and_standardize(corpus_dir, split_seed)
    rng = np.random.default_rng(rng_seed)

    def make(idset):
        out: list[Example] = []
        for vid in idset:
            ft = by_id[vid]
            n = len(ft.node_ids)
            for _ in range(k):
                z = rng.standard_normal(n)
                out.extend(control_b_examples(ft, mean, std, z))
        return out

    return {"train": make(tr), "val": make(va), "test": make(te),
            "train_ids": tr, "val_ids": va, "test_ids": te, "in_dim": N_BASE + 3}


# ── Run + report ────────────────────────────────────────────────────────────
def _evaluate(data: dict, in_dim: int, batch_size):
    from mnemo.gate0.run import train_rung, _labels  # lazy: pulls torch

    test_y = _labels(data["test"])
    r1 = float(np.mean(np.abs(float(np.mean(_labels(data["train"]))) - test_y)))
    res = {"rung1": {"mae": r1, "std": 0.0, "params": 0}}
    for kind, name in [("linear", "rung2"), ("pooled", "rung3"),
                       ("transformer", "rung4")]:
        maes, params = [], 0
        for seed in (0, 1, 2):
            _, mae, params = train_rung(kind, data, seed, in_dim=in_dim,
                                        batch_size=batch_size)
            maes.append(mae)
        res[name] = {"mae": float(np.mean(maes)), "std": float(np.std(maes)),
                     "params": params}
    return res, (len(data["train"]), len(data["test"]))


def run_controls(corpus_dir: str = "corpus") -> dict:
    a = build_control_a(corpus_dir)
    a_res, a_counts = _evaluate(a, a["in_dim"], batch_size=None)
    b = build_control_b(corpus_dir)
    b_res, b_counts = _evaluate(b, b["in_dim"], batch_size=256)
    return {"A": (a_res, a_counts), "B": (b_res, b_counts)}


def print_controls_report(r: dict) -> None:
    a_res, a_counts = r["A"]
    b_res, b_counts = r["B"]
    print("GATE 0 — POSITIVE CONTROLS (synthetic, 10 vids)")
    print("CONTROL A (identity span_rel, unmasked):")
    print("  TEST MAE: rung1=%.4f rung2=%.4f rung3=%.4f rung4=%.4f   examples train/test=%d/%d"
          % (a_res["rung1"]["mae"], a_res["rung2"]["mae"], a_res["rung3"]["mae"],
             a_res["rung4"]["mae"], a_counts[0], a_counts[1]))
    a_pass = all(a_res[k]["mae"] < 0.5 * a_res["rung1"]["mae"]
                 for k in ("rung2", "rung3", "rung4"))
    print("  VERDICT (learner fits trivial signal, rungs2-4 << mean): %s"
          % ("PASS" if a_pass else "FAIL"))
    print("CONTROL B (sibling-mean z, K=64 resamples, query z masked):")
    print("  TEST MAE: rung1=%.4f rung2=%.4f rung3=%.4f rung4=%.4f   examples train/test=%d/%d"
          % (b_res["rung1"]["mae"], b_res["rung2"]["mae"], b_res["rung3"]["mae"],
             b_res["rung4"]["mae"], b_counts[0], b_counts[1]))
    r4, r3, r1 = b_res["rung4"]["mae"], b_res["rung3"]["mae"], b_res["rung1"]["mae"]
    print("  LADDER: rung4<rung3? %s rung4<rung1? %s"
          % ("Y" if r4 < r3 else "N", "Y" if r4 < r1 else "N"))
    b_pass = (r4 < 0.7 * r3) and (r4 < 0.7 * r1)
    print("  VERDICT (structure-aware rung exploits topology, rung4 << rung3 & mean): %s"
          % ("PASS" if b_pass else "FAIL"))
    if not b_pass:
        print("  IF FAIL: tokens carry only timing/level proxies, no explicit "
              "parent/sibling edges; candidate fix = feed parent_idx.")


if __name__ == "__main__":
    print_controls_report(run_controls())
