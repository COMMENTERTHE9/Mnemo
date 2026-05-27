from mnemo.gappers.base import GapperLevel, GapperNode
from mnemo.gappers.scene import build_scene_nodes


def _segment(start, importance, tags=()):
    return GapperNode(
        level=GapperLevel.SEGMENT,
        node_id=f"v1_segment_{int(start):04d}",
        video_id="v1",
        start_time=start,
        end_time=start + 5.0,
        importance=importance,
        summary="",
        narrative_tags=list(tags),
    )


def test_scene_empty():
    assert build_scene_nodes([], "v1") == []


def test_scene_short_video_one_scene():
    # 4 segments (19s video equivalent) → 1 scene
    segs = [_segment(i * 5.0, 0.5) for i in range(4)]
    scenes = build_scene_nodes(segs, "v1")
    assert len(scenes) == 1
    assert scenes[0].start_time == 0.0
    assert scenes[0].end_time == 20.0  # last segment's end
    assert scenes[0].level == GapperLevel.SCENE


def test_scene_long_video_multiple_scenes():
    # 13 segments → 3 scenes of (6, 6, 1) segments
    segs = [_segment(i * 5.0, 0.5) for i in range(13)]
    scenes = build_scene_nodes(segs, "v1")
    assert len(scenes) == 3
    assert scenes[0].metadata["segment_count"] == 6
    assert scenes[1].metadata["segment_count"] == 6
    assert scenes[2].metadata["segment_count"] == 1


def test_scene_aggregates_importance_and_tags():
    segs = [
        _segment(0.0, 0.3, tags=["walking"]),
        _segment(5.0, 0.9, tags=["jumping"]),
        _segment(10.0, 0.6, tags=["walking"]),
    ]
    scenes = build_scene_nodes(segs, "v1")
    assert len(scenes) == 1
    assert abs(scenes[0].importance - 0.6) < 0.01  # mean
    assert "walking" in scenes[0].narrative_tags
    assert "jumping" in scenes[0].narrative_tags
    assert scenes[0].metadata["peak_importance"] == 0.9
