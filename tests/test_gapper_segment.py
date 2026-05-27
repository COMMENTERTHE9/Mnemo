from mnemo.gappers.base import UnifiedFrame, GapperLevel
from mnemo.gappers.segment import build_segment_nodes


def _frame(fn, ts, imp, motion=0.0, audio=False, actions=()):
    return UnifiedFrame(frame_number=fn, timestamp_seconds=ts,
                        importance=imp, blur_variance=100.0,
                        motion_magnitude=motion, has_audio=audio,
                        actions=list(actions))


def test_segment_empty_inputs():
    assert build_segment_nodes([], "v1", 19.0) == []
    assert build_segment_nodes([_frame(0, 0.0, 0.5)], "v1", 0.0) == []


def test_segment_19_second_video_yields_4_segments():
    # 19 frames at 1s intervals, simulating a 19s video sampled at 1fps
    frames = [_frame(fn, float(fn), 0.5) for fn in range(19)]
    nodes = build_segment_nodes(frames, "v1", 19.0)
    # Windows: 0-5, 5-10, 10-15, 15-19 → 4 segments
    assert len(nodes) == 4
    assert nodes[0].start_time == 0.0 and nodes[0].end_time == 5.0
    assert nodes[3].start_time == 15.0 and nodes[3].end_time == 19.0
    for n in nodes:
        assert n.level == GapperLevel.SEGMENT
        assert n.video_id == "v1"
        assert n.parent_id is None  # set by tree builder later


def test_segment_peak_importance_kept():
    frames = [
        _frame(0, 0.5, 0.2),
        _frame(1, 1.5, 0.9),  # peak
        _frame(2, 2.5, 0.3),
    ]
    nodes = build_segment_nodes(frames, "v1", 5.0)
    assert len(nodes) == 1
    assert nodes[0].importance == 0.9


def test_segment_collects_actions_and_audio():
    frames = [
        _frame(0, 0.5, 0.5, audio=True, actions=["walking"]),
        _frame(1, 1.5, 0.5, motion=0.2, actions=["walking", "left_arm_raised"]),
    ]
    nodes = build_segment_nodes(frames, "v1", 5.0)
    assert "walking" in nodes[0].narrative_tags
    assert "left_arm_raised" in nodes[0].narrative_tags
    assert nodes[0].metadata["has_audio"] is True
