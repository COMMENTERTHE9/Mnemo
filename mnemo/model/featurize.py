"""Tree featurizer: node-as-token feature vectors.

Turns an exported memory tree (see mnemo.corpus.export) into one feature
vector per node. Deterministic, pure numpy — no ML framework. This is the
input-representation layer a model sits on.

Each node aggregates the per-second signals whose timestamp falls in its
[start_time, end_time) window. Scalars are left in natural units (no
standardization here — that's a probe-time step on the train split); only
bounded encodings (time-relative ratios, one-hots, multi-hots) are computed.
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

FEATURE_NAMES: list[str] = [
    "importance",       # 0  node importance (~0-1)
    "is_segment",       # 1  one-hot node_level == 1
    "is_scene",         # 2  one-hot node_level == 2
    "is_chapter",       # 3  one-hot node_level == 3
    "is_meta",          # 4  one-hot node_level == 4
    "start_rel",        # 5  start_time / duration  (0-1)
    "end_rel",          # 6  end_time / duration    (0-1)
    "span_rel",         # 7  (end-start) / duration (0-1)
    "n_children",       # 8  raw count of child nodes
    "motion_avg",       # 9  mean total_movement of in-range motion signals
    "motion_peak",      # 10 max  total_movement of in-range motion signals
    "audio_dbfs_avg",   # 11 mean dbfs of in-range audio signals (-80 if none)
    "audio_dbfs_peak",  # 12 max  dbfs of in-range audio signals (-80 if none)
    "audio_presence",   # 13 fraction of in-range audio signals with dbfs > -79
    "frame_imp_avg",    # 14 mean importance of in-range frame signals
    "act_left_arm",     # 15 1 if "left_arm_raised" in any in-range action_hints
    "act_right_arm",    # 16 1 if "right_arm_raised" ...
    "act_jump",         # 17 1 if "possible_jump" ...
    "act_walk",         # 18 1 if "walking_or_running" ...
]
FEATURE_DIM = len(FEATURE_NAMES)

_SILENCE_DBFS = -80.0


@dataclass
class FeaturizedTree:
    video_id: str
    node_ids: list[str]
    levels: list[int]
    parent_idx: list[int]   # index into node_ids; -1 for root
    X: np.ndarray           # [n_nodes, FEATURE_DIM]


def _signal_second(sig: dict[str, Any]) -> float:
    """Timestamp of a signal in seconds (stored as ms)."""
    return (sig.get("timestamp") or 0) / 1000.0


def featurize_tree(tree_json: dict[str, Any]) -> FeaturizedTree:
    """Featurize one exported tree dict into per-node vectors.

    Nodes are ordered by (node_level desc, start_time asc) — the same stable
    order the exporter emits.
    """
    video_id = tree_json.get("video_id", "")
    tree = list(tree_json.get("tree", []))
    signals = list(tree_json.get("signals", []))

    nodes = sorted(tree, key=lambda n: (-int(n["node_level"]), float(n["start_time"])))
    node_ids = [n["node_id"] for n in nodes]
    levels = [int(n["node_level"]) for n in nodes]
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

    # duration: corpus value, else fall back to max node end_time. Guard /0.
    duration = tree_json.get("duration_seconds")
    if not duration or duration <= 0:
        duration = max((float(n["end_time"]) for n in nodes), default=0.0)
    safe_dur = duration if duration and duration > 0 else 1.0

    # Pre-split signals by type with their second-timestamp.
    motion_sigs, audio_sigs, frame_sigs = [], [], []
    for s in signals:
        sec = _signal_second(s)
        gt = s.get("gapper_type")
        if gt == "motion":
            mf = s.get("features", {}).get("motion_features", {}) or {}
            motion_sigs.append((sec, float(mf.get("total_movement", 0.0) or 0.0),
                                list(mf.get("action_hints", []) or [])))
        elif gt == "audio":
            db = s.get("features", {}).get("dbfs")
            audio_sigs.append((sec, _SILENCE_DBFS if db is None else float(db)))
        elif gt == "frame":
            frame_sigs.append((sec, float(s.get("importance", 0.0) or 0.0)))

    # n_children per node_id.
    child_count: dict[str, int] = {}
    for n in nodes:
        pid = n.get("parent_id")
        if pid is not None:
            child_count[pid] = child_count.get(pid, 0) + 1

    X = np.zeros((len(nodes), FEATURE_DIM), dtype=np.float64)
    parent_idx: list[int] = []

    for i, n in enumerate(nodes):
        start = float(n["start_time"])
        end = float(n["end_time"])
        level = int(n["node_level"])

        pid = n.get("parent_id")
        parent_idx.append(id_to_idx.get(pid, -1) if pid is not None else -1)

        def in_range(sec: float) -> bool:
            return start <= sec < end

        m_vals = [mv for (sec, mv, _ah) in motion_sigs if in_range(sec)]
        m_acts = [a for (sec, _mv, ah) in motion_sigs if in_range(sec) for a in ah]
        a_vals = [db for (sec, db) in audio_sigs if in_range(sec)]
        f_vals = [imp for (sec, imp) in frame_sigs if in_range(sec)]

        X[i, 0] = float(n.get("importance", 0.0) or 0.0)
        X[i, 1] = 1.0 if level == 1 else 0.0
        X[i, 2] = 1.0 if level == 2 else 0.0
        X[i, 3] = 1.0 if level == 3 else 0.0
        X[i, 4] = 1.0 if level == 4 else 0.0
        X[i, 5] = start / safe_dur
        X[i, 6] = end / safe_dur
        X[i, 7] = (end - start) / safe_dur
        X[i, 8] = float(child_count.get(n["node_id"], 0))
        X[i, 9] = float(np.mean(m_vals)) if m_vals else 0.0
        X[i, 10] = float(np.max(m_vals)) if m_vals else 0.0
        X[i, 11] = float(np.mean(a_vals)) if a_vals else _SILENCE_DBFS
        X[i, 12] = float(np.max(a_vals)) if a_vals else _SILENCE_DBFS
        X[i, 13] = (sum(1 for db in a_vals if db > -79.0) / len(a_vals)) if a_vals else 0.0
        X[i, 14] = float(np.mean(f_vals)) if f_vals else 0.0
        X[i, 15] = 1.0 if "left_arm_raised" in m_acts else 0.0
        X[i, 16] = 1.0 if "right_arm_raised" in m_acts else 0.0
        X[i, 17] = 1.0 if "possible_jump" in m_acts else 0.0
        X[i, 18] = 1.0 if "walking_or_running" in m_acts else 0.0

    return FeaturizedTree(
        video_id=video_id, node_ids=node_ids, levels=levels,
        parent_idx=parent_idx, X=X,
    )


def load_corpus(corpus_dir: str | Path = "corpus") -> list[FeaturizedTree]:
    """Featurize every <video_id>.json under corpus_dir. Sorted by filename
    for determinism."""
    corpus_dir = Path(corpus_dir)
    out: list[FeaturizedTree] = []
    for path in sorted(corpus_dir.glob("*.json")):
        tree_json = json.loads(path.read_text(encoding="utf-8"))
        out.append(featurize_tree(tree_json))
    return out
