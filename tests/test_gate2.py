import numpy as np

from mnemo.model.featurize import FEATURE_NAMES
from mnemo.gate2.templates import (
    loudness_word, motion_word, position_word, action_value,
    slots_from_row, sentence_tokens, extract_slots, build_vocab,
)


def _row(audio=-25.0, motion=0.05, start=0.1, jump=0, walk=0, larm=0, rarm=0):
    r = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    r[FEATURE_NAMES.index("audio_dbfs_avg")] = audio
    r[FEATURE_NAMES.index("motion_avg")] = motion
    r[FEATURE_NAMES.index("start_rel")] = start
    r[FEATURE_NAMES.index("act_jump")] = jump
    r[FEATURE_NAMES.index("act_walk")] = walk
    r[FEATURE_NAMES.index("act_left_arm")] = larm
    r[FEATURE_NAMES.index("act_right_arm")] = rarm
    return r


def test_bin_edges():
    # loudness boundaries (half-open up)
    assert loudness_word(-56) == "silent"
    assert loudness_word(-55) == "quiet"
    assert loudness_word(-35) == "moderate"
    assert loudness_word(-20) == "loud"
    assert loudness_word(-21) == "moderate"
    # motion boundaries
    assert motion_word(0.0) == "still"
    assert motion_word(0.02) == "calm"
    assert motion_word(0.10) == "active"
    assert motion_word(0.25) == "high-motion"
    # position boundaries
    assert position_word(0.0) == "early"
    assert position_word(0.33) == "midway"
    assert position_word(0.66) == "late"


def test_action_priority():
    assert action_value(1, 1, 1, 1) == "jumping"   # jump wins
    assert action_value(0, 1, 1, 0) == "walking"   # walk over arm
    assert action_value(0, 0, 1, 0) == "arms"
    assert action_value(0, 0, 0, 1) == "arms"
    assert action_value(0, 0, 0, 0) == "none"


def test_template_determinism():
    r = _row(audio=-10, motion=0.3, start=0.8, jump=1)
    s1 = sentence_tokens(slots_from_row(r))
    s2 = sentence_tokens(slots_from_row(r.copy()))
    assert s1 == s2
    assert s1 == ["a", "loud", ",", "high-motion", "segment", "late",
                  "in", "the", "video", "with", "jumping", "."]


def test_slot_extraction_roundtrip():
    for audio, motion, start, act in [
        (-60, 0.0, 0.1, {}), (-30, 0.05, 0.5, {"walk": 1}),
        (-10, 0.4, 0.9, {"jump": 1}), (-25, 0.15, 0.4, {"larm": 1}),
    ]:
        r = _row(audio=audio, motion=motion, start=start, **act)
        slots = slots_from_row(r)
        toks = sentence_tokens(slots)
        assert extract_slots(toks) == slots  # generated->slots recovers exactly


def test_vocab_covers_all_templates():
    vocab = set(build_vocab())
    r = _row(audio=-10, motion=0.3, start=0.8, larm=1)
    assert all(t in vocab for t in sentence_tokens(slots_from_row(r)))
    assert {"<pad>", "<bos>", "<eos>"} <= vocab


def test_unconditioned_arm_sees_no_tree():
    import torch
    from mnemo.gate2.model import TreeToText
    torch.manual_seed(0)
    model = TreeToText(len(build_vocab()), conditioned=False)
    # Two totally different trees -> condition vector must be identical
    # (the unconditioned arm replaces the encoder output with a constant).
    t1 = torch.rand(1, 6, 19)
    t2 = torch.rand(1, 9, 19)
    pad1 = torch.zeros(1, 6, dtype=torch.bool)
    pad2 = torch.zeros(1, 9, dtype=torch.bool)
    q = torch.zeros(1, dtype=torch.long)
    c1 = model.condition(t1, q, pad1)
    c2 = model.condition(t2, q, pad2)
    assert torch.equal(c1, c2)  # no tree-derived input
