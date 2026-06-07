"""Deterministic templated verbalization of a segment node (pure numpy/stdlib).

Each SEGMENT node maps to ONE sentence fully determined by binned features:
  loudness  <- audio_dbfs_avg : silent(<-55) quiet(<-35) moderate(<-20) loud(>=-20)
  motion    <- motion_avg     : still(<0.02) calm(<0.10) active(<0.25) high-motion(>=0.25)
  position  <- start_rel      : early(<0.33) midway(<0.66) late(>=0.66)
  action    <- act_* flags    : jump > walk > arm priority, else none
Template: "a {loudness} , {motion} segment {position} in the video[ with ...] ."
"""
from __future__ import annotations

from mnemo.model.featurize import FEATURE_NAMES

_AUDIO = FEATURE_NAMES.index("audio_dbfs_avg")
_MOTION = FEATURE_NAMES.index("motion_avg")
_START = FEATURE_NAMES.index("start_rel")
_LARM = FEATURE_NAMES.index("act_left_arm")
_RARM = FEATURE_NAMES.index("act_right_arm")
_JUMP = FEATURE_NAMES.index("act_jump")
_WALK = FEATURE_NAMES.index("act_walk")

LOUDNESS_WORDS = ("silent", "quiet", "moderate", "loud")
MOTION_WORDS = ("still", "calm", "active", "high-motion")
POSITION_WORDS = ("early", "midway", "late")
ACTION_VALUES = ("none", "jumping", "walking", "arms")

# Closed sets for slot extraction from a generated/target token list.
_LOUD_SET = set(LOUDNESS_WORDS)
_MOT_SET = set(MOTION_WORDS)
_POS_SET = set(POSITION_WORDS)

SPECIALS = ("<pad>", "<bos>", "<eos>")


def loudness_word(dbfs: float) -> str:
    if dbfs < -55:
        return "silent"
    if dbfs < -35:
        return "quiet"
    if dbfs < -20:
        return "moderate"
    return "loud"


def motion_word(m: float) -> str:
    if m < 0.02:
        return "still"
    if m < 0.10:
        return "calm"
    if m < 0.25:
        return "active"
    return "high-motion"


def position_word(start_rel: float) -> str:
    if start_rel < 0.33:
        return "early"
    if start_rel < 0.66:
        return "midway"
    return "late"


def action_value(jump: float, walk: float, larm: float, rarm: float) -> str:
    if jump > 0.5:
        return "jumping"
    if walk > 0.5:
        return "walking"
    if larm > 0.5 or rarm > 0.5:
        return "arms"
    return "none"


def _action_clause(value: str) -> list[str]:
    return {"none": [], "jumping": ["with", "jumping"],
            "walking": ["with", "walking"],
            "arms": ["with", "raised", "arms"]}[value]


def slots_from_row(row) -> dict:
    """Slot dict from a raw (un-standardized) 19-dim feature row (rung-1
    hand-picked dBFS/motion edges)."""
    return {
        "loudness": loudness_word(float(row[_AUDIO])),
        "motion": motion_word(float(row[_MOTION])),
        "position": position_word(float(row[_START])),
        "action": action_value(float(row[_JUMP]), float(row[_WALK]),
                               float(row[_LARM]), float(row[_RARM])),
    }


# ── Quantile binning (rung 1.5: edges fitted on TRAIN videos only) ───────────
def fit_quantile_edges(values, n_bins: int = 4) -> list[float]:
    """Internal edges at the (n_bins-1) inner quantiles of `values`."""
    import numpy as np
    qs = [i / n_bins for i in range(1, n_bins)]
    return [float(np.quantile(np.asarray(values, dtype=float), q)) for q in qs]


def _bin_word(value: float, edges: list[float], words) -> str:
    i = 0
    for e in edges:
        if value < e:
            break
        i += 1
    return words[min(i, len(words) - 1)]


def slots_from_row_quantile(row, loud_edges: list[float],
                            motion_edges: list[float]) -> dict:
    """Slot dict using TRAIN-fitted quantile edges for loudness & motion;
    position (thirds) and action (priority flags) unchanged."""
    return {
        "loudness": _bin_word(float(row[_AUDIO]), loud_edges, LOUDNESS_WORDS),
        "motion": _bin_word(float(row[_MOTION]), motion_edges, MOTION_WORDS),
        "position": position_word(float(row[_START])),
        "action": action_value(float(row[_JUMP]), float(row[_WALK]),
                               float(row[_LARM]), float(row[_RARM])),
    }


# raw-feature column indices (for masking / quantile fitting in masked.py)
AUDIO_IDX, MOTION_IDX, START_IDX = _AUDIO, _MOTION, _START


def sentence_tokens(slots: dict) -> list[str]:
    return (["a", slots["loudness"], ",", slots["motion"], "segment",
             slots["position"], "in", "the", "video"]
            + _action_clause(slots["action"]) + ["."])


def extract_slots(tokens: list[str]) -> dict:
    """Recover slot values from a token list (generated or target). Slot
    vocabularies are disjoint, so membership is unambiguous; action is keyed
    off its content words."""
    loud = next((t for t in tokens if t in _LOUD_SET), None)
    mot = next((t for t in tokens if t in _MOT_SET), None)
    pos = next((t for t in tokens if t in _POS_SET), None)
    if "jumping" in tokens:
        act = "jumping"
    elif "walking" in tokens:
        act = "walking"
    elif "raised" in tokens or "arms" in tokens:
        act = "arms"
    else:
        act = "none"
    return {"loudness": loud, "motion": mot, "position": pos, "action": act}


def build_vocab() -> list[str]:
    """Stable vocab: specials + every token any template sentence can contain."""
    words = set()
    for loud in LOUDNESS_WORDS:
        for mot in MOTION_WORDS:
            for pos in POSITION_WORDS:
                for act in ACTION_VALUES:
                    words.update(sentence_tokens(
                        {"loudness": loud, "motion": mot, "position": pos,
                         "action": act}))
    return list(SPECIALS) + sorted(words)
