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


# ── rung 1.5: masked-node verbalization ──────────────────────────────────────
def test_quantile_edges_deterministic_and_train_only():
    from mnemo.gate2.templates import fit_quantile_edges, _bin_word, LOUDNESS_WORDS
    vals = list(range(100))
    e1 = fit_quantile_edges(vals, 4)
    e2 = fit_quantile_edges(vals, 4)
    assert e1 == e2                       # deterministic
    assert len(e1) == 3                   # 3 inner edges for 4 bins
    # different data -> different edges (fitted on what you pass, i.e. train-only)
    assert fit_quantile_edges([v * 10 for v in vals], 4) != e1
    # binning respects edges (quartiles of 0..99 ~ 24.75/49.5/74.25)
    assert _bin_word(0, e1, LOUDNESS_WORDS) == "silent"
    assert _bin_word(99, e1, LOUDNESS_WORDS) == "loud"


def test_mask_covers_exactly_nine_content_cols_on_path():
    from mnemo.model.featurize import featurize_tree
    from mnemo.gate2.masked import CONTENT, N_BASE
    from mnemo.gate0.data import ancestors_of
    # known tree: meta -> scene -> {seg1, seg2}
    tree = {
        "video_id": "m", "duration_seconds": 10.0,
        "tree": [
            {"node_id": "meta", "node_level": 4, "parent_id": None,
             "start_time": 0.0, "end_time": 10.0, "importance": 0.5,
             "summary": "", "narrative_tags": []},
            {"node_id": "scene", "node_level": 2, "parent_id": "meta",
             "start_time": 0.0, "end_time": 10.0, "importance": 0.5,
             "summary": "", "narrative_tags": []},
            {"node_id": "seg1", "node_level": 1, "parent_id": "scene",
             "start_time": 0.0, "end_time": 5.0, "importance": 0.5,
             "summary": "", "narrative_tags": []},
            {"node_id": "seg2", "node_level": 1, "parent_id": "scene",
             "start_time": 5.0, "end_time": 10.0, "importance": 0.5,
             "summary": "", "narrative_tags": []},
        ],
        "signals": [{"gapper_type": "audio", "timestamp": t * 1000,
                     "importance": 0.5, "features": {"dbfs": -20.0}} for t in range(10)]
              + [{"gapper_type": "frame", "timestamp": t * 1000, "importance": 0.5,
                  "features": {"blur_variance": 150.0}} for t in range(10)],
    }
    ft = featurize_tree(tree)
    i1 = ft.node_ids.index("seg1")
    hidden = {i1, *ancestors_of(ft, i1)}
    assert hidden == {i1, ft.node_ids.index("scene"), ft.node_ids.index("meta")}
    # simulate the mask (standardize-free here: just zero content cols on path)
    masked = ft.X.copy()
    for h in hidden:
        masked[h, CONTENT] = 0.0
    structural = [j for j in range(N_BASE) if j not in set(CONTENT)]
    # exactly the 9 content cols are zeroed on hidden nodes; structural untouched
    assert len(CONTENT) == 9
    for h in hidden:
        assert bool((masked[h, CONTENT] == 0.0).all())
        assert bool((masked[h, structural] == ft.X[h, structural]).all())
    # the visible sibling seg2 keeps its content
    i2 = ft.node_ids.index("seg2")
    assert bool((masked[i2, CONTENT] == ft.X[i2, CONTENT]).all())


def test_masked_priors_twin_sees_no_tree():
    import torch
    from mnemo.gate2.model import TreeToText
    from mnemo.gate2.templates import build_vocab
    from mnemo.gate2.masked import INPUT_DIM
    torch.manual_seed(0)
    model = TreeToText(len(build_vocab()), in_dim=INPUT_DIM, conditioned=False)
    c1 = model.condition(torch.rand(1, 6, INPUT_DIM), torch.zeros(1, dtype=torch.long),
                         torch.zeros(1, 6, dtype=torch.bool))
    c2 = model.condition(torch.rand(1, 9, INPUT_DIM), torch.zeros(1, dtype=torch.long),
                         torch.zeros(1, 9, dtype=torch.bool))
    assert torch.equal(c1, c2)  # learned constant, independent of the tree


def test_ceiling_and_text_arms_share_input_dim():
    from mnemo.gate2.masked import INPUT_DIM, SlotClassifier, SLOT_CLASSES
    from mnemo.gate2.model import TreeToText
    from mnemo.gate2.templates import build_vocab
    clf = SlotClassifier(len(SLOT_CLASSES["loudness"]))
    txt = TreeToText(len(build_vocab()), in_dim=INPUT_DIM, conditioned=True)
    # both encoders project from the same masked token width
    assert clf.enc.proj.in_features == INPUT_DIM
    assert txt.encoder.proj.in_features == INPUT_DIM
