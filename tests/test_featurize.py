import numpy as np

from mnemo.model.featurize import featurize_tree, FEATURE_NAMES, FEATURE_DIM


def _f(name):
    """Index of a feature by name."""
    return FEATURE_NAMES.index(name)


def _tree():
    """Synthetic tree: 1 meta (root) + 2 segments.

    Segment A (0-5s): 3 motion signals (tm 0.2/0.4/0.6, one with possible_jump),
    2 audio signals (-20/-30 dBFS), 1 frame (imp 0.5).
    Segment B (5-10s): NO signals at all (tests empty-window fallbacks).
    """
    return {
        "video_id": "synthetic",
        "duration_seconds": 10.0,
        "tree": [
            {"node_id": "meta", "node_level": 4, "parent_id": None,
             "start_time": 0.0, "end_time": 10.0, "importance": 0.5,
             "summary": "", "narrative_tags": []},
            {"node_id": "segA", "node_level": 1, "parent_id": "meta",
             "start_time": 0.0, "end_time": 5.0, "importance": 0.6,
             "summary": "", "narrative_tags": []},
            {"node_id": "segB", "node_level": 1, "parent_id": "meta",
             "start_time": 5.0, "end_time": 10.0, "importance": 0.3,
             "summary": "", "narrative_tags": []},
        ],
        "signals": [
            {"gapper_type": "motion", "timestamp": 0, "importance": 0.3,
             "features": {"motion_features": {"motion_detected": True,
                          "total_movement": 0.2, "action_hints": []}}},
            {"gapper_type": "motion", "timestamp": 1000, "importance": 0.5,
             "features": {"motion_features": {"motion_detected": True,
                          "total_movement": 0.4, "action_hints": ["possible_jump"]}}},
            {"gapper_type": "motion", "timestamp": 2000, "importance": 0.7,
             "features": {"motion_features": {"motion_detected": True,
                          "total_movement": 0.6, "action_hints": []}}},
            {"gapper_type": "audio", "timestamp": 0, "importance": 0.5,
             "features": {"dbfs": -20.0, "rms": 0.1}},
            {"gapper_type": "audio", "timestamp": 1000, "importance": 0.5,
             "features": {"dbfs": -30.0, "rms": 0.03}},
            {"gapper_type": "frame", "timestamp": 0, "importance": 0.5,
             "features": {"blur_variance": 150.0}},
        ],
    }


def test_shape_and_feature_names():
    ft = featurize_tree(_tree())
    assert ft.X.shape == (3, 19)
    assert len(FEATURE_NAMES) == 19 == FEATURE_DIM


def test_node_order_and_parent_idx():
    ft = featurize_tree(_tree())
    # Order: level desc, start asc -> meta, segA, segB.
    assert ft.node_ids == ["meta", "segA", "segB"]
    assert ft.levels == [4, 1, 1]
    # meta is root -> -1; both segments point at meta (index 0).
    assert ft.parent_idx == [-1, 0, 0]


def test_n_children():
    ft = featurize_tree(_tree())
    meta = ft.node_ids.index("meta")
    assert ft.X[meta, _f("n_children")] == 2.0
    segA = ft.node_ids.index("segA")
    assert ft.X[segA, _f("n_children")] == 0.0


def test_motion_avg_peak_in_range():
    ft = featurize_tree(_tree())
    segA = ft.node_ids.index("segA")
    # tm values 0.2, 0.4, 0.6 -> mean 0.4, max 0.6
    assert abs(ft.X[segA, _f("motion_avg")] - 0.4) < 1e-9
    assert abs(ft.X[segA, _f("motion_peak")] - 0.6) < 1e-9


def test_audio_avg_and_action_hit():
    ft = featurize_tree(_tree())
    segA = ft.node_ids.index("segA")
    # dbfs -20, -30 -> mean -25, peak -20
    assert abs(ft.X[segA, _f("audio_dbfs_avg")] - (-25.0)) < 1e-9
    assert abs(ft.X[segA, _f("audio_dbfs_peak")] - (-20.0)) < 1e-9
    assert ft.X[segA, _f("audio_presence")] == 1.0
    # possible_jump fired in this window
    assert ft.X[segA, _f("act_jump")] == 1.0
    # arm/walk did not
    assert ft.X[segA, _f("act_left_arm")] == 0.0
    assert ft.X[segA, _f("act_walk")] == 0.0


def test_empty_window_fallbacks():
    ft = featurize_tree(_tree())
    segB = ft.node_ids.index("segB")  # 5-10s, no signals
    assert ft.X[segB, _f("motion_avg")] == 0.0
    assert ft.X[segB, _f("motion_peak")] == 0.0
    assert ft.X[segB, _f("audio_dbfs_avg")] == -80.0
    assert ft.X[segB, _f("audio_dbfs_peak")] == -80.0
    assert ft.X[segB, _f("audio_presence")] == 0.0
    assert ft.X[segB, _f("frame_imp_avg")] == 0.0


def test_time_features_bounded():
    ft = featurize_tree(_tree())
    for col in ("start_rel", "end_rel", "span_rel"):
        vals = ft.X[:, _f(col)]
        assert np.all(vals >= 0.0) and np.all(vals <= 1.0)
    # one-hots are 0/1
    for col in ("is_segment", "is_scene", "is_chapter", "is_meta"):
        vals = ft.X[:, _f(col)]
        assert np.all((vals == 0.0) | (vals == 1.0))


def test_duration_fallback_to_max_end_time():
    t = _tree()
    t["duration_seconds"] = None  # force fallback to max end_time (10.0)
    ft = featurize_tree(t)
    meta = ft.node_ids.index("meta")
    assert abs(ft.X[meta, _f("end_rel")] - 1.0) < 1e-9
    assert np.all(ft.X[:, _f("end_rel")] <= 1.0)
