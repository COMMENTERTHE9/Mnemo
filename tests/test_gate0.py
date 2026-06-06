import numpy as np

from mnemo.model.featurize import featurize_tree
from mnemo.gate0.data import (
    split_videos, build_examples_for_tree, ancestors_of,
    IS_SEGMENT_IDX, AUDIO_AVG_IDX, AUDIO_PEAK_IDX, N_BASE,
)
from mnemo.gate0.controls import (
    control_a_examples, control_b_examples, siblings_of, SPAN_REL_IDX, Z_COL,
)
from mnemo.gate0.structure import (
    relation_matrix, REL_SELF, REL_PARENT, REL_CHILD, REL_SIBLING, REL_OTHER,
)


def test_split_disjoint():
    ids = [f"v{i}" for i in range(10)]
    train, val, test = split_videos(ids, seed=0)
    assert len(train) == 6 and len(val) == 2 and len(test) == 2
    s_tr, s_va, s_te = set(train), set(val), set(test)
    assert not (s_tr & s_va)
    assert not (s_tr & s_te)
    assert not (s_va & s_te)
    # Covers exactly the input ids, no dupes.
    assert s_tr | s_va | s_te == set(ids)


def _known_tree():
    """meta(L4) -> scene(L2) -> {seg1(L1,0-5), seg2(L1,5-10)}, with audio so
    audio_dbfs_avg is non-zero on every node."""
    signals = []
    for t in range(10):
        signals.append({"gapper_type": "audio", "timestamp": t * 1000,
                        "importance": 0.5, "features": {"dbfs": -20.0 - t, "rms": 0.1}})
        signals.append({"gapper_type": "frame", "timestamp": t * 1000,
                        "importance": 0.5, "features": {"blur_variance": 150.0}})
    return {
        "video_id": "known",
        "duration_seconds": 10.0,
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
        "signals": signals,
    }


def test_mask_blanks_ancestor_path():
    ft = featurize_tree(_known_tree())
    # node order: meta(4), scene(2), seg1(1,start0), seg2(1,start5)
    assert ft.node_ids == ["meta", "scene", "seg1", "seg2"]
    seg1_idx = ft.node_ids.index("seg1")
    # identity standardizer so we can read raw audio values back
    mean = np.zeros(N_BASE)
    std = np.ones(N_BASE)
    examples = build_examples_for_tree(ft, mean, std)
    ex = next(e for e in examples if e.query_idx == seg1_idx)

    # ancestor chain seg1 -> scene -> meta
    assert ancestors_of(ft, seg1_idx) == [ft.node_ids.index("scene"),
                                          ft.node_ids.index("meta")]

    # audio_hidden == 1 exactly on {seg1, scene, meta}, 0 on seg2
    expected_hidden = np.zeros(4)
    for nid in ("seg1", "scene", "meta"):
        expected_hidden[ft.node_ids.index(nid)] = 1.0
    assert np.array_equal(ex.audio_hidden, expected_hidden)

    # audio columns zeroed exactly where hidden...
    for j in range(4):
        if expected_hidden[j] == 1.0:
            assert ex.tokens[j, AUDIO_AVG_IDX] == 0.0
            assert ex.tokens[j, AUDIO_PEAK_IDX] == 0.0
    # ...and NOT zeroed on the visible sibling (seg2 has real audio)
    seg2_idx = ft.node_ids.index("seg2")
    assert ex.tokens[seg2_idx, AUDIO_AVG_IDX] != 0.0

    # is_query one-hot on seg1 only
    expected_q = np.zeros(4)
    expected_q[seg1_idx] = 1.0
    assert np.array_equal(ex.is_query, expected_q)


def test_query_nodes_are_segments():
    ft = featurize_tree(_known_tree())
    mean = np.zeros(N_BASE)
    std = np.ones(N_BASE)
    examples = build_examples_for_tree(ft, mean, std)
    assert examples  # non-empty
    for e in examples:
        assert ft.X[e.query_idx, IS_SEGMENT_IDX] == 1.0


# ── Positive controls ───────────────────────────────────────────────────────
def test_control_a_label_is_span_rel():
    ft = featurize_tree(_known_tree())
    mean = np.zeros(N_BASE)
    std = np.ones(N_BASE)  # identity standardizer
    examples = control_a_examples(ft, mean, std)
    assert len(examples) == len(ft.node_ids)  # one per node, any level
    for e in examples:
        # label is the node's true span_rel
        assert abs(e.label - ft.X[e.query_idx, SPAN_REL_IDX]) < 1e-9
        # input is UNMASKED: the node's own span_rel sits in its token row
        assert abs(e.tokens[e.query_idx, SPAN_REL_IDX] - e.label) < 1e-9


def test_control_b_label_is_sibling_mean():
    ft = featurize_tree(_known_tree())  # meta -> scene -> {seg1, seg2}
    i1 = ft.node_ids.index("seg1")
    i2 = ft.node_ids.index("seg2")
    assert siblings_of(ft.parent_idx, i1) == [i2]  # seg1 & seg2 are siblings

    z = np.zeros(len(ft.node_ids))
    z[i1] = 5.0
    z[i2] = 7.0
    examples = control_b_examples(ft, np.zeros(N_BASE), np.ones(N_BASE), z)
    ex1 = next(e for e in examples if e.query_idx == i1)
    # label = mean of siblings' z (just seg2 here) and EXCLUDES the query's own z
    assert abs(ex1.label - 7.0) < 1e-9


def test_control_b_query_z_masked():
    ft = featurize_tree(_known_tree())
    i1 = ft.node_ids.index("seg1")
    i2 = ft.node_ids.index("seg2")
    z = np.zeros(len(ft.node_ids))
    z[i1] = 5.0
    z[i2] = 7.0
    examples = control_b_examples(ft, np.zeros(N_BASE), np.ones(N_BASE), z)
    ex1 = next(e for e in examples if e.query_idx == i1)
    # query's own z column is masked to 0...
    assert ex1.tokens[i1, Z_COL] == 0.0
    # ...while the visible sibling keeps its z
    assert ex1.tokens[i2, Z_COL] == 7.0


def test_audio_examples_carry_correct_rel():
    ft = featurize_tree(_known_tree())  # meta -> scene -> {seg1, seg2}
    examples = build_examples_for_tree(ft, np.zeros(N_BASE), np.ones(N_BASE))
    expected = relation_matrix(ft.parent_idx)
    assert examples  # segments are queries
    for e in examples:
        assert e.rel is not None
        assert np.array_equal(e.rel, expected)
    # spot-check the actual relations against the tree
    i1 = ft.node_ids.index("seg1")
    i2 = ft.node_ids.index("seg2")
    sc = ft.node_ids.index("scene")
    assert expected[i1, i2] == REL_SIBLING       # seg1 & seg2 share scene
    assert expected[i1, sc] == REL_PARENT         # scene is seg1's parent
    assert expected[sc, i1] == REL_CHILD          # seg1 is scene's child


def test_relation_matrix():
    # tree: root(0) -> {A(1), B(2)};  A -> {A1(3), A2(4)}
    #   parent_idx: root=-1, A=0, B=0, A1=1, A2=1
    parent_idx = [-1, 0, 0, 1, 1]
    rel = relation_matrix(parent_idx)
    assert rel.shape == (5, 5)
    # self
    assert rel[0, 0] == REL_SELF
    assert rel[3, 3] == REL_SELF
    # parent: A's parent is root -> rel[A, root] = parent
    assert rel[1, 0] == REL_PARENT
    assert rel[3, 1] == REL_PARENT          # A1's parent is A
    # child: root's child is A -> rel[root, A] = child
    assert rel[0, 1] == REL_CHILD
    assert rel[1, 3] == REL_CHILD           # A's child is A1
    # sibling: A & B share root; A1 & A2 share A
    assert rel[1, 2] == REL_SIBLING
    assert rel[3, 4] == REL_SIBLING
    # other: A1 vs B (different parents, not parent/child)
    assert rel[3, 2] == REL_OTHER
    # other: root vs A1 (A1's parent is A, not root)
    assert rel[0, 3] == REL_OTHER
