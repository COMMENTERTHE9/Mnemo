"""Reader engine for the Mnemo sidecar — corpus + learned reader behind the
wire protocol. **torch lives here** (model load + generation); the protocol
layer (server.py) never imports it.

The PRODUCTION reader is the Gate-2 rung-1 decoder (conditioned TreeToText over
the full visible tree, structure-blind) trained with rung-1.5 QUANTILE slot
edges, on the FULL corpus with NO held-out split — the generalization science
already happened (Gate 2). `train_and_save` bundles weights + bin edges +
feature normalization into one versioned file under weights/ (gitignored).

Granularity / source split (the wire contract):
  summary  -> SCENE nodes, deterministic gauge-template text   (source "gauge")
  describe -> SEGMENT nodes, learned-reader generation         (source "model")
  peaks    -> top-k SEGMENT nodes by a metric, gauge text      (source "gauge")
Every narration line carries a receipt: loud_db, mot, act, t0, t1.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from mnemo.model.featurize import FEATURE_NAMES, featurize_tree, FeaturizedTree
from mnemo.gate2.templates import (
    action_value, fit_quantile_edges, slots_from_row_quantile, sentence_tokens,
    AUDIO_IDX, MOTION_IDX,
)

# ── feature column indices (raw, un-standardized) ────────────────────────────
IMP = FEATURE_NAMES.index("importance")
IS_SEGMENT = FEATURE_NAMES.index("is_segment")
IS_SCENE = FEATURE_NAMES.index("is_scene")
JUMP = FEATURE_NAMES.index("act_jump")
WALK = FEATURE_NAMES.index("act_walk")
LARM = FEATURE_NAMES.index("act_left_arm")
RARM = FEATURE_NAMES.index("act_right_arm")

LEVEL_SEGMENT, LEVEL_SCENE = 1, 2

SUMMARY_CAP = 16          # scene lines
DESCRIBE_CAP = 12         # segment lines
PEAKS_CAP = 10            # top-k

WEIGHTS_FORMAT = "mnemo-reader-v1"
DEFAULT_WEIGHTS = "weights/reader_v1.pt"
DEFAULT_CORPUS = "corpus"

PEAK_METRICS = {
    "loudness": FEATURE_NAMES.index("audio_dbfs_avg"),
    "motion": FEATURE_NAMES.index("motion_avg"),
    "importance": IMP,
}


class UnknownVideo(Exception):
    """Requested video id is not in the loaded corpus."""


class BadParams(Exception):
    """Request parameters were missing or malformed."""


@dataclass
class Video:
    ft: FeaturizedTree
    starts: np.ndarray        # absolute seconds, aligned to ft node order
    ends: np.ndarray          # absolute seconds
    duration: float
    xstd: np.ndarray = field(default=None)  # standardized features for the model


# ── corpus loading (absolute times alongside the featurizer) ─────────────────
def _load_videos(corpus_dir: str | Path) -> dict[str, Video]:
    """Featurize every <id>.json and recover absolute node times in the SAME
    node order the featurizer emits (sort by node_level desc, start_time asc)."""
    videos: dict[str, Video] = {}
    for path in sorted(Path(corpus_dir).glob("*.json")):
        tj = json.loads(path.read_text(encoding="utf-8"))
        ft = featurize_tree(tj)
        nodes = sorted(tj.get("tree", []),
                       key=lambda n: (-int(n["node_level"]), float(n["start_time"])))
        starts = np.array([float(n["start_time"]) for n in nodes], dtype=np.float64)
        ends = np.array([float(n["end_time"]) for n in nodes], dtype=np.float64)
        dur = tj.get("duration_seconds")
        if not dur or dur <= 0:
            dur = float(ends.max()) if ends.size else 0.0
        dur = float(dur) if dur and dur > 0 else 1.0
        videos[ft.video_id] = Video(ft=ft, starts=starts, ends=ends, duration=dur)
    return videos


# ── offline training: the production reader ──────────────────────────────────
def train_and_save(corpus_dir: str = DEFAULT_CORPUS,
                   out_path: str = DEFAULT_WEIGHTS,
                   epochs: int = 120, seed: int = 0) -> dict:
    """Train the gauge reader on the full corpus (no split) with quantile slot
    edges; bundle weights + edges + normalization to a versioned file. Returns
    {weights_hash, path, n_examples, n_videos}."""
    import random
    import torch
    import torch.nn as nn

    from mnemo.model.featurize import load_corpus
    from mnemo.gate2 import run as R
    from mnemo.gate2.model import TreeToText

    trees = load_corpus(corpus_dir)
    if not trees:
        raise BadParams(f"no corpus json under {corpus_dir!r}")

    # full-corpus standardizer (production: every node, no held-out split)
    all_x = np.vstack([t.X for t in trees])
    mean = all_x.mean(0)
    std = all_x.std(0)
    std = np.where(std < 1e-8, 1.0, std)

    # quantile slot edges fitted on ALL segment nodes
    seg_audio, seg_motion = [], []
    for t in trees:
        for i in range(len(t.node_ids)):
            if t.X[i, IS_SEGMENT] == 1.0:
                seg_audio.append(t.X[i, AUDIO_IDX])
                seg_motion.append(t.X[i, MOTION_IDX])
    loud_edges = fit_quantile_edges(seg_audio)
    motion_edges = fit_quantile_edges(seg_motion)

    # one example per segment node (full visible tree, segment target sentence)
    examples = []
    for t in trees:
        xstd = ((t.X - mean) / std).astype(np.float32)
        for i in range(len(t.node_ids)):
            if t.X[i, IS_SEGMENT] != 1.0:
                continue
            slots = slots_from_row_quantile(t.X[i], loud_edges, motion_edges)
            target = [R.BOS] + [R.TOK2ID[w] for w in sentence_tokens(slots)] + [R.EOS]
            examples.append(R.Example(xstd, i, target, slots, t.video_id))

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    model = TreeToText(len(R.VOCAB), in_dim=len(FEATURE_NAMES), conditioned=True)
    opt = torch.optim.Adam(model.parameters(), lr=R.LR)
    loss_fn = nn.CrossEntropyLoss(ignore_index=R.PAD)
    order = list(range(len(examples)))
    rng = random.Random(seed)
    for _ in range(epochs):
        model.train()
        rng.shuffle(order)
        for s in range(0, len(order), R.BATCH):
            batch = [examples[i] for i in order[s:s + R.BATCH]]
            tokens, qidx, pad_mask, tin, tout = R._collate(batch)
            opt.zero_grad()
            logits = model(tokens, qidx, pad_mask, tin)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), tout.reshape(-1))
            loss.backward()
            opt.step()

    bundle = {
        "format": WEIGHTS_FORMAT,
        "in_dim": len(FEATURE_NAMES),
        "d_model": model.d_model,
        "vocab": list(R.VOCAB),
        "loud_edges": [float(e) for e in loud_edges],
        "motion_edges": [float(e) for e in motion_edges],
        "mean": [float(x) for x in mean],
        "std": [float(x) for x in std],
        "state_dict": model.state_dict(),
        "epochs": epochs,
        "seed": seed,
        "n_examples": len(examples),
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, out)
    weights_hash = hashlib.sha256(out.read_bytes()).hexdigest()[:12]
    return {"weights_hash": weights_hash, "path": str(out),
            "n_examples": len(examples), "n_videos": len(trees)}


# ── the engine the protocol layer talks to ───────────────────────────────────
class Engine:
    def __init__(self, videos, model, vocab, mean, std,
                 loud_edges, motion_edges, weights_hash):
        self.videos = videos
        self.model = model
        self.vocab = vocab
        self.mean = mean
        self.std = std
        self.loud_edges = loud_edges
        self.motion_edges = motion_edges
        self.weights_hash = weights_hash
        self.bos, self.eos = vocab.index("<bos>"), vocab.index("<eos>")

    @classmethod
    def load(cls, corpus_dir: str = DEFAULT_CORPUS,
             weights_path: str = DEFAULT_WEIGHTS) -> "Engine":
        """Load corpus + trained reader. NEVER trains. Fast (<1s typical)."""
        import torch
        from mnemo.gate2.model import TreeToText

        wp = Path(weights_path)
        if not wp.exists():
            raise FileNotFoundError(
                f"weights not found at {weights_path!r}; run "
                f"`python -m mnemo.sidecar train` first")
        weights_hash = hashlib.sha256(wp.read_bytes()).hexdigest()[:12]
        bundle = torch.load(wp, map_location="cpu", weights_only=False)
        vocab = list(bundle["vocab"])
        mean = np.array(bundle["mean"], dtype=np.float64)
        std = np.array(bundle["std"], dtype=np.float64)
        loud_edges = list(bundle["loud_edges"])
        motion_edges = list(bundle["motion_edges"])
        model = TreeToText(len(vocab), in_dim=int(bundle["in_dim"]),
                           d_model=int(bundle["d_model"]), conditioned=True)
        model.load_state_dict(bundle["state_dict"])
        model.eval()

        videos = _load_videos(corpus_dir)
        for v in videos.values():
            v.xstd = ((v.ft.X - mean) / std).astype(np.float32)
        return cls(videos, model, vocab, mean, std,
                   loud_edges, motion_edges, weights_hash)

    # -- helpers --------------------------------------------------------------
    def _video(self, vid) -> Video:
        if not isinstance(vid, str) or vid not in self.videos:
            raise UnknownVideo(f"no such video: {vid!r}")
        return self.videos[vid]

    def _receipt(self, v: Video, i: int) -> dict:
        row = v.ft.X[i]
        act = action_value(float(row[JUMP]), float(row[WALK]),
                           float(row[LARM]), float(row[RARM]))
        return {
            "loud_db": round(float(row[AUDIO_IDX]), 2),
            "mot": round(float(row[MOTION_IDX]), 4),
            "act": None if act == "none" else act,
            "t0": round(float(v.starts[i]), 2),
            "t1": round(float(v.ends[i]), 2),
        }

    def _gauge_line(self, v: Video, i: int, noun: str) -> dict:
        slots = slots_from_row_quantile(v.ft.X[i], self.loud_edges, self.motion_edges)
        text = " ".join(sentence_tokens(slots, noun=noun))
        return {"text": text, **self._receipt(v, i)}

    def _model_lines(self, v: Video, idxs: list[int]) -> list[dict]:
        if not idxs:
            return []
        import torch
        n = v.xstd.shape[0]
        tokens = torch.from_numpy(np.stack([v.xstd] * len(idxs)))
        qidx = torch.tensor(idxs, dtype=torch.long)
        pad = torch.zeros(len(idxs), n, dtype=torch.bool)
        seqs = self.model.generate(tokens, qidx, pad, self.bos, self.eos)
        lines = []
        for j, i in enumerate(idxs):
            text = " ".join(self.vocab[t] for t in seqs[j])
            lines.append({"text": text, **self._receipt(v, i)})
        return lines

    def _level_idxs(self, v: Video, level: int) -> list[int]:
        idxs = [i for i, lv in enumerate(v.ft.levels) if lv == level]
        idxs.sort(key=lambda i: float(v.starts[i]))
        return idxs

    # -- methods --------------------------------------------------------------
    def list_videos(self) -> dict:
        out = []
        for vid in sorted(self.videos):
            v = self.videos[vid]
            out.append({
                "id": vid,
                "duration": round(v.duration, 2),
                "n_scenes": sum(1 for lv in v.ft.levels if lv == LEVEL_SCENE),
                "n_segments": sum(1 for lv in v.ft.levels if lv == LEVEL_SEGMENT),
            })
        return {"videos": out, "n_videos": len(out)}

    def summary(self, vid) -> dict:
        v = self._video(vid)
        idxs = self._level_idxs(v, LEVEL_SCENE)
        truncated = len(idxs) > SUMMARY_CAP
        lines = [self._gauge_line(v, i, noun="scene") for i in idxs[:SUMMARY_CAP]]
        return {"video": vid, "source": "gauge", "lines": lines,
                "truncated": truncated}

    def describe(self, vid, t0=None, t1=None) -> dict:
        v = self._video(vid)
        idxs = self._level_idxs(v, LEVEL_SEGMENT)
        windowed = (t0 is not None) or (t1 is not None)
        if windowed:
            lo = float("-inf") if t0 is None else float(t0)
            hi = float("inf") if t1 is None else float(t1)
            idxs = [i for i in idxs if v.starts[i] < hi and v.ends[i] > lo]
        total = len(idxs)
        shown = idxs[:DESCRIBE_CAP]
        truncated = total > DESCRIBE_CAP
        lines = self._model_lines(v, shown)
        hint = None
        if truncated:
            where = " in window" if windowed else ""
            hint = (f"{total} segments{where}; showing first {DESCRIBE_CAP}. "
                    f"Pass a t0 t1 window to narrow.")
        return {"video": vid, "source": "model",
                "window": [t0, t1] if windowed else None,
                "lines": lines, "truncated": truncated, "hint": hint}

    def peaks(self, vid, metric, k=5) -> dict:
        v = self._video(vid)
        if metric not in PEAK_METRICS:
            raise BadParams(
                f"unknown metric {metric!r}; one of {sorted(PEAK_METRICS)}")
        try:
            k = int(k)
        except (TypeError, ValueError):
            raise BadParams(f"k must be an integer, got {k!r}")
        k = max(1, min(k, PEAKS_CAP))
        col = PEAK_METRICS[metric]
        idxs = [i for i, lv in enumerate(v.ft.levels) if lv == LEVEL_SEGMENT]
        idxs.sort(key=lambda i: float(v.ft.X[i, col]), reverse=True)
        lines = []
        for i in idxs[:k]:
            ln = self._gauge_line(v, i, noun="segment")
            ln["value"] = round(float(v.ft.X[i, col]), 4)
            lines.append(ln)
        return {"video": vid, "metric": metric, "k": k, "lines": lines}
