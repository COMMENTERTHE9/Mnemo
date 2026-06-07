"""Gate 3 protocol: FP baseline, ternary iso-shape, ternary iso-footprint, on
the rung-1.5 masked-verbalization task. Paired across 3 seeds. torch via model.
"""
from __future__ import annotations
import copy
import random

import numpy as np
import torch
import torch.nn as nn

from mnemo.gate2 import run as R
from mnemo.gate2 import masked as M
from mnemo.gate3.model import Gate3Model, footprint

VERDICT_SLOTS = ("loudness", "motion")
LR_MULTS = (1.0, 3.0)


def _val_loss(model, examples) -> float:
    model.eval()
    loss_fn = nn.CrossEntropyLoss(ignore_index=R.PAD, reduction="sum")
    total, ntok = 0.0, 0
    with torch.no_grad():
        for s in range(0, len(examples), 128):
            tokens, qidx, pad_mask, tin, tout = R._collate(examples[s:s + 128])
            logits = model(tokens, qidx, pad_mask, tin)
            total += float(loss_fn(logits.reshape(-1, logits.size(-1)), tout.reshape(-1)))
            ntok += int((tout != R.PAD).sum())
    return total / max(1, ntok)


def train_gate3(train, val, seed, *, quantize, d_model, lr_mult=1.0,
                conditioned=True) -> Gate3Model:
    torch.manual_seed(seed)
    model = Gate3Model(len(R.VOCAB), in_dim=M.INPUT_DIM, d_model=d_model,
                       quantize=quantize, conditioned=conditioned)
    opt = torch.optim.Adam(model.parameters(), lr=R.LR * lr_mult)
    loss_fn = nn.CrossEntropyLoss(ignore_index=R.PAD)
    order = list(range(len(train)))
    rng = random.Random(seed)
    best, best_state, bad = float("inf"), copy.deepcopy(model.state_dict()), 0
    for _ in range(R.EPOCHS):
        model.train()
        rng.shuffle(order)
        for s in range(0, len(order), R.BATCH):
            b = [train[i] for i in order[s:s + R.BATCH]]
            tokens, qidx, pad_mask, tin, tout = R._collate(b)
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
            if bad >= R.PATIENCE:
                break
    model.load_state_dict(best_state)
    return model


def _eval(model, test):
    slot_acc, _, gens = R.evaluate(model, test)
    adj = {s: M._adjacency(gens, s) for s in ("loudness", "motion")}
    return slot_acc, adj


def _pick_d_for_footprint(fp_bits, vocab, in_dim, tol=0.10):
    """Smallest d_model whose ternary footprint is within ±tol of fp_bits."""
    best = None
    for d in range(32, 256, 2):
        m = Gate3Model(vocab, in_dim=in_dim, d_model=d, quantize=True)
        bits = footprint(m, quantized=True)["bits"]
        ratio = bits / fp_bits
        if abs(ratio - 1.0) <= tol:
            return d, bits
        if best is None or abs(ratio - 1.0) < abs(best[1] / fp_bits - 1.0):
            best = (d, bits)
    return best  # closest if none within tol


def _pick_lr(train, val, d_model):
    """Select LR multiplier on val (seed 0) for the ternary arm."""
    scores = []
    for lm in LR_MULTS:
        m = train_gate3(train, val, 0, quantize=True, d_model=d_model, lr_mult=lm)
        scores.append((lm, _val_loss(m, val)))
    return min(scores, key=lambda kv: kv[1])[0]


def run_gate3(corpus_dir="corpus") -> dict:
    train, val, test, *_ = M.build_masked_dataset(corpus_dir)
    vocab = len(R.VOCAB)

    fp_bits = footprint(Gate3Model(vocab, in_dim=M.INPUT_DIM, d_model=32,
                                   quantize=False), quantized=False)["bits"]
    shape_bits = footprint(Gate3Model(vocab, in_dim=M.INPUT_DIM, d_model=32,
                                      quantize=True), quantized=True)["bits"]
    foot_d, foot_bits = _pick_d_for_footprint(fp_bits, vocab, M.INPUT_DIM)

    lr_shape = _pick_lr(train, val, 32)
    lr_foot = _pick_lr(train, val, foot_d)

    arms = {"fp": {}, "shape": {}, "foot": {}}
    for slot in ("loudness", "motion", "position", "action"):
        for a in arms:
            arms[a][slot] = []
            arms[a].setdefault("adj_" + slot, [])
    for seed in R.SEEDS:
        fp = train_gate3(train, val, seed, quantize=False, d_model=32)
        sh = train_gate3(train, val, seed, quantize=True, d_model=32, lr_mult=lr_shape)
        ft = train_gate3(train, val, seed, quantize=True, d_model=foot_d, lr_mult=lr_foot)
        for name, model in (("fp", fp), ("shape", sh), ("foot", ft)):
            sa, adj = _eval(model, test)
            for slot in ("loudness", "motion", "position", "action"):
                arms[name][slot].append(sa[slot])
            for slot in ("loudness", "motion"):
                arms[name]["adj_" + slot].append(adj[slot])

    # ternary fraction from the iso-shape ternary model
    fp_acc = footprint(Gate3Model(vocab, in_dim=M.INPUT_DIM, d_model=32, quantize=True),
                       quantized=True)
    frac_params = fp_acc["ternary_params"] / fp_acc["total_params"]
    frac_bits = (fp_acc["ternary_params"] * 2) / fp_acc["bits"]
    return {"arms": arms,
            "footprints": {"fp": fp_bits, "shape": shape_bits,
                           "foot": foot_bits, "foot_d": foot_d},
            "lr": {"shape": lr_shape, "foot": lr_foot},
            "frac": {"params": frac_params, "bits": frac_bits},
            "counts": (len(train), len(val), len(test))}


def _ms(xs):
    return float(np.mean(xs)), float(np.std(xs))


def print_report(res: dict) -> None:
    arms = res["arms"]
    fp = res["footprints"]
    print("GATE 3 — ternarization (rung-1.5 task, 3 seeds, paired)")
    print(f"  examples train/val/test = {res['counts'][0]}/{res['counts'][1]}/{res['counts'][2]}")
    eps = {s: 2 * _ms(arms["fp"][s])[1] for s in VERDICT_SLOTS}
    print("TABLE: slot | FP | t-iso-shape | t-iso-footprint | epsilon")
    for slot in ("loudness", "motion", "position", "action"):
        f = _ms(arms["fp"][slot])
        s = _ms(arms["shape"][slot])
        t = _ms(arms["foot"][slot])
        e = f"{eps[slot]:.3f}" if slot in eps else "  -  "
        print(f"  {slot:9s} | {f[0]:.3f}±{f[1]:.3f} | {s[0]:.3f}±{s[1]:.3f} | "
              f"{t[0]:.3f}±{t[1]:.3f} | {e}")
    within10 = abs(fp["foot"] / fp["fp"] - 1.0) <= 0.10
    print(f"WIDTHS + FOOTPRINTS: fp={fp['fp']} bits  shape={fp['shape']} bits  "
          f"footprint-arm(d={fp['foot_d']})={fp['foot']} bits (±10%? {'Y' if within10 else 'N'})")
    print(f"TERNARY FRACTION: {100 * res['frac']['params']:.1f}% params, "
          f"{100 * res['frac']['bits']:.1f}% bits   "
          f"LR choice: shape={res['lr']['shape']:g}x foot={res['lr']['foot']:g}x")
    pos = _ms(arms["fp"]["position"])[0]
    pos_s = _ms(arms["shape"]["position"])[0]
    pos_t = _ms(arms["foot"]["position"])[0]
    print(f"SANITY: FP repro loud/motion/pos = {_ms(arms['fp']['loudness'])[0]:.3f}/"
          f"{_ms(arms['fp']['motion'])[0]:.3f}/{pos:.3f} (committed rung-1.5 ~0.740/0.733/0.986)")

    def arm_pass(name):
        ok_slots = all(abs(_ms(arms[name][s])[0] - _ms(arms["fp"][s])[0]) <= eps[s]
                       for s in VERDICT_SLOTS)
        ctrl = _ms(arms[name]["position"])[0] >= 0.90
        return ok_slots and ctrl
    shape_pass = arm_pass("shape")
    foot_pass = arm_pass("foot")
    print(f"VERDICTS: iso-shape {'PASS' if shape_pass else 'FAIL'} "
          f"(pos {pos_s:.3f})  iso-footprint {'PASS' if foot_pass else 'FAIL'} (pos {pos_t:.3f})")
    if shape_pass:
        reading = "clean pass — ternarization free at iso-shape"
    elif foot_pass:
        reading = "thesis holds — performance per byte (iso-footprint recovers it)"
    else:
        reading = "thesis killed at this scale"
    print(f"  -> THESIS READING: {reading}")
    print("  FP sits at the analytic ceiling, so a tie reads as no degradation "
          "detectable at this task difficulty, not as ternarization being free.")


if __name__ == "__main__":
    print_report(run_gate3())
