"""Gate 2 rung 1.5 — masked-node verbalization.

Describe a segment whose own CONTENT is hidden, from tree context: the fusion of
Gate 0 (predict a masked feature) and rung 1 (verbalize). The query segment's 9
content columns are zeroed (post-standardization) along its whole ancestor path
(ancestors aggregate the query's seconds); structural columns stay visible, so
`position` is a gauge-reading CONTROL slot, while loudness/motion/action are
inferential. Per-slot ceilings are measured in-sprint (a direct classifier on
the same masked inputs) so the bars are relative to what is possible.

torch stays inside the gate packages.
"""
from __future__ import annotations
import copy
import random

import numpy as np
import torch
import torch.nn as nn

from mnemo.model.featurize import load_corpus, FEATURE_NAMES
from mnemo.gate0.models import TinyTransformer
from mnemo.gate0.data import ancestors_of
from mnemo.gate2.templates import (
    fit_quantile_edges, slots_from_row_quantile, sentence_tokens,
    LOUDNESS_WORDS, MOTION_WORDS, ACTION_VALUES, AUDIO_IDX, MOTION_IDX,
)
from mnemo.gate2 import run as R

CONTENT = [FEATURE_NAMES.index(c) for c in (
    "audio_dbfs_avg", "audio_dbfs_peak", "motion_avg", "motion_peak",
    "frame_imp_avg", "act_left_arm", "act_right_arm", "act_jump", "act_walk")]
IS_SEG = FEATURE_NAMES.index("is_segment")
N_BASE = 19
INPUT_DIM = 21
GATE0_MAE = 2.59
SLOT_CLASSES = {"loudness": LOUDNESS_WORDS, "motion": MOTION_WORDS,
                "action": ACTION_VALUES}
RANK = {"loudness": {w: i for i, w in enumerate(LOUDNESS_WORDS)},
        "motion": {w: i for i, w in enumerate(MOTION_WORDS)}}


def build_masked_dataset(corpus_dir="corpus"):
    trees = load_corpus(corpus_dir)
    by_id = {t.video_id: t for t in trees}
    tr, va, te = R._split([t.video_id for t in trees])
    train_X = np.vstack([by_id[v].X for v in tr])
    mean, std = train_X.mean(0), train_X.std(0)
    std = np.where(std < 1e-8, 1.0, std)
    # quantile edges fitted on TRAIN segment nodes only
    seg_audio, seg_motion = [], []
    for v in tr:
        ft = by_id[v]
        for i in range(len(ft.node_ids)):
            if ft.X[i, IS_SEG] == 1.0:
                seg_audio.append(ft.X[i, AUDIO_IDX])
                seg_motion.append(ft.X[i, MOTION_IDX])
    loud_edges = fit_quantile_edges(seg_audio)
    motion_edges = fit_quantile_edges(seg_motion)

    def make(ids):
        out, raw_dbfs = [], []
        for v in ids:
            ft = by_id[v]
            xstd = ((ft.X - mean) / std).astype(np.float32)
            n = len(ft.node_ids)
            for i in range(n):
                if ft.X[i, IS_SEG] != 1.0:
                    continue
                hidden = {i, *ancestors_of(ft, i)}
                masked = xstd.copy()
                for h in hidden:
                    masked[h, CONTENT] = 0.0  # blank content on query + path
                is_q = np.zeros(n, np.float32)
                mflag = np.zeros(n, np.float32)
                is_q[i] = 1.0
                for h in hidden:
                    mflag[h] = 1.0
                tokens = np.concatenate(
                    [masked, is_q[:, None], mflag[:, None]], axis=1).astype(np.float32)
                slots = slots_from_row_quantile(ft.X[i], loud_edges, motion_edges)
                target = [R.BOS] + [R.TOK2ID[t] for t in sentence_tokens(slots)] + [R.EOS]
                out.append(R.Example(tokens, i, target, slots, v))
                raw_dbfs.append(float(ft.X[i, AUDIO_IDX]))
        return out, raw_dbfs

    train, _ = make(tr)
    val, _ = make(va)
    test, test_raw_dbfs = make(te)
    return train, val, test, loud_edges, motion_edges, test_raw_dbfs


# ── Ceiling: direct slot classifier on the same masked inputs ────────────────
class SlotClassifier(nn.Module):
    def __init__(self, n_classes, in_dim=INPUT_DIM, d_model=32, nhead=2,
                 layers=1, ff=64):
        super().__init__()
        self.enc = TinyTransformer(in_dim=in_dim, d_model=d_model, nhead=nhead,
                                   layers=layers, ff=ff)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, tokens, query_idx, pad_mask):
        h = self.enc.proj(tokens)
        h = self.enc.encoder(h, src_key_padding_mask=pad_mask)
        return self.head(h[torch.arange(h.size(0)), query_idx])


def _labels(examples, slot):
    classes = SLOT_CLASSES[slot]
    idx = {w: i for i, w in enumerate(classes)}
    return torch.tensor([idx[e.slots[slot]] for e in examples], dtype=torch.long)


def _cls_eval(model, examples, slot):
    model.eval()
    y = _labels(examples, slot)
    preds = []
    with torch.no_grad():
        for s in range(0, len(examples), 128):
            b = examples[s:s + 128]
            tokens, qidx, pad_mask, _, _ = R._collate(b)
            preds.append(model(tokens, qidx, pad_mask).argmax(-1))
    return float((torch.cat(preds) == y).float().mean())


def train_classifier(train, val, test, slot, seed) -> float:
    """Ceiling = held-out accuracy of a direct classifier on masked inputs."""
    torch.manual_seed(seed)
    model = SlotClassifier(len(SLOT_CLASSES[slot]))
    opt = torch.optim.Adam(model.parameters(), lr=R.LR)
    loss_fn = nn.CrossEntropyLoss()
    y_tr = _labels(train, slot)
    order = list(range(len(train)))
    rng = random.Random(seed)
    best, best_state, bad = float("inf"), copy.deepcopy(model.state_dict()), 0
    for _ in range(R.EPOCHS):
        model.train()
        rng.shuffle(order)
        for s in range(0, len(order), R.BATCH):
            idx = order[s:s + R.BATCH]
            b = [train[i] for i in idx]
            tokens, qidx, pad_mask, _, _ = R._collate(b)
            opt.zero_grad()
            loss = loss_fn(model(tokens, qidx, pad_mask), y_tr[idx])
            loss.backward()
            opt.step()
        # val loss
        model.eval()
        yv = _labels(val, slot)
        with torch.no_grad():
            tokens, qidx, pad_mask, _, _ = R._collate(val)
            vl = float(loss_fn(model(tokens, qidx, pad_mask), yv))
        if vl < best - 1e-5:
            best, best_state, bad = vl, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= R.PATIENCE:
                break
    model.load_state_dict(best_state)
    return _cls_eval(model, test, slot)


def _adjacency(gens, slot) -> float:
    """Adjacent-bin accuracy for an ordinal slot (loudness/motion)."""
    from mnemo.gate2.templates import extract_slots
    rank = RANK[slot]
    ok = 0
    for e, gen in gens:
        pred = extract_slots(gen)[slot]
        true = e.slots[slot]
        if pred is not None and abs(rank[pred] - rank[true]) <= 1:
            ok += 1
    return ok / len(gens)


def _analytic_loudness_ceiling(raw_dbfs, loud_edges, seed=0) -> float:
    """MC bin-match: true_dbfs + Gate-0 noise N(0, 1.253*MAE) under quantile
    edges — how often the loudness bin survives Gate-0-grade prediction error.
    (MAE of a zero-mean Gaussian = sigma*sqrt(2/pi) -> sigma = MAE*1.253.)"""
    rng = np.random.default_rng(seed)
    sigma = 1.253 * GATE0_MAE

    def b(v):
        i = 0
        for e in loud_edges:
            if v < e:
                break
            i += 1
        return i

    ok, n = 0, 0
    for d in raw_dbfs:
        for _ in range(64):
            if b(d) == b(d + rng.normal(0, sigma)):
                ok += 1
            n += 1
    return ok / max(1, n)


def run_masked(corpus_dir="corpus") -> dict:
    train, val, test, loud_edges, motion_edges, test_raw_dbfs = build_masked_dataset(corpus_dir)
    slots_inf = ("loudness", "motion", "action")
    agg = {"ceiling": {s: [] for s in slots_inf},
           "context": {s: [] for s in ("loudness", "motion", "position", "action")},
           "priors": {s: [] for s in ("loudness", "motion", "position", "action")},
           "adj_ctx": {s: [] for s in ("loudness", "motion")}}
    samples = []
    for seed in R.SEEDS:
        for s in slots_inf:
            agg["ceiling"][s].append(train_classifier(train, val, test, s, seed))
        ctx = R.train_arm(train, val, seed, conditioned=True, in_dim=INPUT_DIM)
        pri = R.train_arm(train, val, seed, conditioned=False, in_dim=INPUT_DIM)
        cs, _, cg = R.evaluate(ctx, test)
        ps, _, _ = R.evaluate(pri, test)
        for s in ("loudness", "motion", "position", "action"):
            agg["context"][s].append(cs[s])
            agg["priors"][s].append(ps[s])
        agg["adj_ctx"]["loudness"].append(_adjacency(cg, "loudness"))
        agg["adj_ctx"]["motion"].append(_adjacency(cg, "motion"))
        if seed == R.SEEDS[0]:
            samples = cg[:3]
    analytic = _analytic_loudness_ceiling(test_raw_dbfs, loud_edges)
    return {"agg": agg, "samples": samples, "analytic_loudness": analytic,
            "counts": (len(train), len(val), len(test))}


def _ms(xs):
    return float(np.mean(xs)), float(np.std(xs))


def print_report(out: dict) -> None:
    agg = out["agg"]
    a, b, c = out["counts"]
    print("RUNG 1.5 — masked verbalization (40 vids, by-video, 3 seeds)")
    print(f"  examples train/val/test = {a}/{b}/{c}")
    print("SLOT TABLE: slot | ceiling | context | priors-only | adjacent(ctx)")
    for s in ("loudness", "motion", "action", "position"):
        ceil = f"{_ms(agg['ceiling'][s])[0]:.3f}" if s in agg["ceiling"] else "  -  "
        ctx = _ms(agg["context"][s])[0]
        pri = _ms(agg["priors"][s])[0]
        adj = f"{_ms(agg['adj_ctx'][s])[0]:.3f}" if s in agg["adj_ctx"] else "  -  "
        print(f"  {s:9s} | {ceil} | {ctx:.3f} | {pri:.3f} | {adj}")
    print(f"LOUDNESS CEILING CROSS-CHECK: classifier={_ms(agg['ceiling']['loudness'])[0]:.3f} "
          f"analytic={out['analytic_loudness']:.3f}")
    pos = _ms(agg["context"]["position"])[0]
    print(f"POSITION CONTROL: {pos:.3f}")
    for s in ("loudness", "motion", "action"):
        ceil = _ms(agg["ceiling"][s])[0]
        ctx = _ms(agg["context"][s])[0]
        pri = _ms(agg["priors"][s])[0]
        reaches = ctx >= 0.9 * ceil
        beats = ctx > pri + 0.05
        verdict = "PASS" if (reaches and beats) else "FAIL"
        reading = ("low ceiling = data property" if ceil < 0.6 else
                   "context below own ceiling" if not reaches else "")
        print(f"  VERDICT {s}: {verdict} (ctx {ctx:.3f} vs 0.9*ceiling {0.9 * ceil:.3f}, "
              f"vs priors {pri:.3f}) {reading}")
    print("SAMPLES (held-out, context):")
    for e, gen in out["samples"]:
        print(f"  TRUE: {' '.join(sentence_tokens(e.slots))}")
        print(f"  GEN : {' '.join(gen)}")


if __name__ == "__main__":
    print_report(run_masked())
