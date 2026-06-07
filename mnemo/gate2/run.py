"""Gate 2 rung-1 protocol: by-video split, 3 seeds, conditioned vs unconditioned.
torch via model.py.
"""
from __future__ import annotations
import copy
import random

import numpy as np
import torch
import torch.nn as nn

from mnemo.model.featurize import load_corpus
from mnemo.gate2.templates import (
    slots_from_row, sentence_tokens, extract_slots, build_vocab,
)
from mnemo.gate2.model import TreeToText

SEEDS = [0, 1, 2]
SPLIT_SEED = 0
EPOCHS = 150
PATIENCE = 20
BATCH = 64
LR = 1e-3
SLOTS = ("loudness", "motion", "position", "action")

VOCAB = build_vocab()
TOK2ID = {t: i for i, t in enumerate(VOCAB)}
PAD, BOS, EOS = TOK2ID["<pad>"], TOK2ID["<bos>"], TOK2ID["<eos>"]


class Example:
    __slots__ = ("std_tokens", "query_idx", "target_ids", "slots", "video_id")

    def __init__(self, std_tokens, query_idx, target_ids, slots, video_id):
        self.std_tokens = std_tokens
        self.query_idx = query_idx
        self.target_ids = target_ids
        self.slots = slots
        self.video_id = video_id


def _split(video_ids, seed=SPLIT_SEED, n_train=28, n_val=6, n_test=6):
    ids = sorted(video_ids)
    random.Random(seed).shuffle(ids)
    return ids[:n_train], ids[n_train:n_train + n_val], ids[n_train + n_val:n_train + n_val + n_test]


def build_dataset(corpus_dir="corpus"):
    trees = load_corpus(corpus_dir)
    by_id = {t.video_id: t for t in trees}
    tr, va, te = _split([t.video_id for t in trees])
    train_X = np.vstack([by_id[v].X for v in tr])
    mean, std = train_X.mean(0), train_X.std(0)
    std = np.where(std < 1e-8, 1.0, std)
    from mnemo.model.featurize import FEATURE_NAMES
    is_seg = FEATURE_NAMES.index("is_segment")

    def make(ids):
        out = []
        for v in ids:
            ft = by_id[v]
            xs = ((ft.X - mean) / std).astype(np.float32)
            for i in range(len(ft.node_ids)):
                if ft.X[i, is_seg] != 1.0:
                    continue
                slots = slots_from_row(ft.X[i])
                toks = ["<bos>"] + sentence_tokens(slots) + ["<eos>"]
                ids_ = [TOK2ID[t] for t in toks]
                out.append(Example(xs, i, ids_, slots, v))
        return out

    return make(tr), make(va), make(te)


def _collate(batch):
    n_max = max(e.std_tokens.shape[0] for e in batch)
    dim = batch[0].std_tokens.shape[1]
    l_max = max(len(e.target_ids) for e in batch)
    B = len(batch)
    tokens = torch.zeros(B, n_max, dim)
    pad_mask = torch.ones(B, n_max, dtype=torch.bool)
    qidx = torch.zeros(B, dtype=torch.long)
    tgt = torch.full((B, l_max), PAD, dtype=torch.long)
    for i, e in enumerate(batch):
        n = e.std_tokens.shape[0]
        tokens[i, :n] = torch.tensor(e.std_tokens)
        pad_mask[i, :n] = False
        qidx[i] = e.query_idx
        tgt[i, :len(e.target_ids)] = torch.tensor(e.target_ids)
    return tokens, qidx, pad_mask, tgt[:, :-1], tgt[:, 1:]


def _val_loss(model, examples) -> float:
    """Teacher-forced CE on val — a smooth early-stop signal (val exact-match
    plateaus and stops training while per-slot accuracy is still climbing)."""
    model.eval()
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD, reduction="sum")
    total, n_tok = 0.0, 0
    with torch.no_grad():
        for s in range(0, len(examples), 128):
            batch = examples[s:s + 128]
            tokens, qidx, pad_mask, tin, tout = _collate(batch)
            logits = model(tokens, qidx, pad_mask, tin)
            total += float(loss_fn(logits.reshape(-1, logits.size(-1)), tout.reshape(-1)))
            n_tok += int((tout != PAD).sum())
    return total / max(1, n_tok)


def train_arm(train, val, seed, conditioned) -> TreeToText:
    torch.manual_seed(seed)
    model = TreeToText(len(VOCAB), conditioned=conditioned)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)
    order = list(range(len(train)))
    rng = random.Random(seed)
    best, best_state, bad = float("inf"), copy.deepcopy(model.state_dict()), 0
    for _ in range(EPOCHS):
        model.train()
        rng.shuffle(order)
        for s in range(0, len(order), BATCH):
            batch = [train[i] for i in order[s:s + BATCH]]
            tokens, qidx, pad_mask, tin, tout = _collate(batch)
            opt.zero_grad()
            logits = model(tokens, qidx, pad_mask, tin)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), tout.reshape(-1))
            loss.backward()
            opt.step()
        vl = _val_loss(model, val)
        if vl < best - 1e-5:
            best, best_state, bad = vl, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model


def evaluate(model, examples):
    slot_ok = {s: 0 for s in SLOTS}
    exact = 0
    gens = []
    for s in range(0, len(examples), 128):
        batch = examples[s:s + 128]
        tokens, qidx, pad_mask, _, _ = _collate(batch)
        seqs = model.generate(tokens, qidx, pad_mask, BOS, EOS)
        for e, seq in zip(batch, seqs):
            gen_tokens = [VOCAB[t] for t in seq]
            pred = extract_slots(gen_tokens)
            for sl in SLOTS:
                if pred[sl] == e.slots[sl]:
                    slot_ok[sl] += 1
            if seq == e.target_ids[1:-1]:
                exact += 1
            gens.append((e, gen_tokens))
    n = len(examples)
    return ({s: slot_ok[s] / n for s in SLOTS}, exact / n, gens)


def run_gate2(corpus_dir="corpus") -> dict:
    train, val, test = build_dataset(corpus_dir)
    res = {"cond": {s: [] for s in SLOTS}, "uncond": {s: [] for s in SLOTS}}
    res["cond"]["exact"], res["uncond"]["exact"] = [], []
    samples = []
    for seed in SEEDS:
        cond = train_arm(train, val, seed, conditioned=True)
        unc = train_arm(train, val, seed, conditioned=False)
        cs, cx, cg = evaluate(cond, test)
        us, ux, _ = evaluate(unc, test)
        for s in SLOTS:
            res["cond"][s].append(cs[s])
            res["uncond"][s].append(us[s])
        res["cond"]["exact"].append(cx)
        res["uncond"]["exact"].append(ux)
        if seed == SEEDS[0]:
            samples = cg[:3]
    return {"res": res, "samples": samples,
            "counts": (len(train), len(val), len(test))}


def _ms(xs):
    return float(np.mean(xs)), float(np.std(xs))


def print_report(out: dict) -> None:
    r = out["res"]
    a, b, c = out["counts"]
    print("GATE 2 RUNG 1 — templated verbalization (40 vids, by-video split, 3 seeds)")
    print(f"  examples train/val/test = {a}/{b}/{c}")

    def line(slot):
        cm, _ = _ms(r["cond"][slot])
        um, _ = _ms(r["uncond"][slot])
        return f"{slot}={cm:.3f}/{um:.3f}"
    print("SLOT ACC (cond/uncond): " + "  ".join(line(s) for s in SLOTS))
    cx, _ = _ms(r["cond"]["exact"])
    ux, _ = _ms(r["uncond"]["exact"])
    print(f"EXACT MATCH (cond/uncond): {cx:.3f}/{ux:.3f}")
    print("SAMPLES (held-out, conditioned):")
    for e, gen in out["samples"]:
        true = sentence_tokens(e.slots)
        print(f"  TRUE: {' '.join(true)}")
        print(f"  GEN : {' '.join(gen)}")
    cond_high = all(_ms(r["cond"][s])[0] > 0.90 for s in SLOTS)
    beats = all(_ms(r["cond"][s])[0] > _ms(r["uncond"][s])[0] + 0.05 for s in SLOTS)
    print(f"VERDICT (pre-registered): {'PASS' if (cond_high and beats) else 'FAIL'}  "
          f"(cond>90% all slots? {'Y' if cond_high else 'N'}; cond>>uncond all slots? {'Y' if beats else 'N'})")


if __name__ == "__main__":
    print_report(run_gate2())
