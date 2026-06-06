import json
from mnemo.db import init_for_settings, enqueue_video, insert_gapper_report
from mnemo.memory.builder import build_memory_tree
from mnemo.models import GapperReport


def _seed_per_second_frame_data(conn, video_id, seconds: int, fps: float = 1.0):
    """Insert frame + motion + audio gapper_reports for `seconds` seconds
    of synthetic video at `fps`."""
    for i in range(seconds):
        ts_ms = int(i * 1000)
        insert_gapper_report(conn, GapperReport(
            video_id=video_id, gapper_type="frame", timestamp_ms=ts_ms,
            gapper_id=f"f_{i}", start_frame=i, end_frame=i,
            summary="", importance=0.5,
            features={"blur_variance": 150.0},
        ))
        insert_gapper_report(conn, GapperReport(
            video_id=video_id, gapper_type="motion", timestamp_ms=ts_ms,
            gapper_id=f"m_{i}", start_frame=i, end_frame=i,
            summary="", importance=0.7,
            features={"motion_features": {
                "total_movement": 0.1,
                "action_hints": ["walking"] if i % 3 == 0 else [],
            }},
        ))
        insert_gapper_report(conn, GapperReport(
            video_id=video_id, gapper_type="audio", timestamp_ms=ts_ms,
            gapper_id=f"a_{i}", start_frame=0, end_frame=0,
            summary="", importance=0.5, features={},
        ))


def test_build_tree_19_second_video(settings):
    conn = init_for_settings(settings)
    vid = enqueue_video(conn, "test://x")
    _seed_per_second_frame_data(conn, vid, seconds=19)
    count = build_memory_tree(conn, vid, video_duration=19.0)

    # 1 meta + 1 chapter + 1 scene + 4 segments = 7
    assert count == 7

    # Verify structure
    rows = conn.execute(
        "SELECT node_level, COUNT(*) AS n FROM memory_nodes "
        "WHERE video_id=? GROUP BY node_level ORDER BY node_level",
        (vid,),
    ).fetchall()
    by_level = {r["node_level"]: r["n"] for r in rows}
    assert by_level == {1: 4, 2: 1, 3: 1, 4: 1}  # segment/scene/chapter/meta


def test_build_tree_parent_ids_correct(settings):
    conn = init_for_settings(settings)
    vid = enqueue_video(conn, "test://x")
    _seed_per_second_frame_data(conn, vid, seconds=19)
    build_memory_tree(conn, vid, video_duration=19.0)

    # The meta node has no parent
    meta = conn.execute(
        "SELECT node_id, parent_id FROM memory_nodes WHERE node_level=4 AND video_id=?",
        (vid,),
    ).fetchone()
    assert meta["parent_id"] is None

    # All chapters point at meta
    chapters = conn.execute(
        "SELECT node_id, parent_id FROM memory_nodes WHERE node_level=3 AND video_id=?",
        (vid,),
    ).fetchall()
    for ch in chapters:
        assert ch["parent_id"] == meta["node_id"]

    # All scenes point at a real chapter
    scene_parents = {r["parent_id"] for r in conn.execute(
        "SELECT parent_id FROM memory_nodes WHERE node_level=2 AND video_id=?",
        (vid,),
    )}
    chapter_ids = {ch["node_id"] for ch in chapters}
    assert scene_parents.issubset(chapter_ids)

    # All segments point at a real scene
    segment_parents = {r["parent_id"] for r in conn.execute(
        "SELECT parent_id FROM memory_nodes WHERE node_level=1 AND video_id=?",
        (vid,),
    )}
    scene_ids = {r["node_id"] for r in conn.execute(
        "SELECT node_id FROM memory_nodes WHERE node_level=2 AND video_id=?",
        (vid,),
    )}
    assert segment_parents.issubset(scene_ids)


def test_build_tree_empty_video(settings):
    conn = init_for_settings(settings)
    vid = enqueue_video(conn, "test://x")
    # No frames seeded
    count = build_memory_tree(conn, vid, 0.0)
    assert count == 0


def test_build_tree_long_video_correct_counts(settings):
    """Synthetic 7-minute (420s) video: 1 meta + 2 chapters + 14 scenes + 84 segments = 101"""
    conn = init_for_settings(settings)
    vid = enqueue_video(conn, "test://x")
    _seed_per_second_frame_data(conn, vid, seconds=420)
    count = build_memory_tree(conn, vid, video_duration=420.0)
    assert count == 101


def test_build_tree_segment_at_scene_boundary_parents_to_containing_scene(settings):
    """Regression: a segment starting exactly at a scene boundary (e.g. 30.0s,
    where the first scene ends) must be parented to the scene that contains
    [30, ...), not the scene that ends at 30. (Guards the half-open _find_parent.)"""
    conn = init_for_settings(settings)
    vid = enqueue_video(conn, "test://x")
    _seed_per_second_frame_data(conn, vid, seconds=35)  # 7 segments -> 2 scenes
    build_memory_tree(conn, vid, video_duration=35.0)

    seg = conn.execute(
        "SELECT node_id, parent_id FROM memory_nodes "
        "WHERE node_level=1 AND video_id=? AND start_time=30.0",
        (vid,),
    ).fetchone()
    assert seg is not None
    parent = conn.execute(
        "SELECT start_time, end_time FROM memory_nodes WHERE node_id=? AND video_id=?",
        (seg["parent_id"], vid),
    ).fetchone()
    # The parent scene must actually contain the boundary start_time.
    assert parent["start_time"] <= 30.0 < parent["end_time"]


def test_build_tree_actions_propagate(settings):
    conn = init_for_settings(settings)
    vid = enqueue_video(conn, "test://x")
    _seed_per_second_frame_data(conn, vid, seconds=19)
    build_memory_tree(conn, vid, 19.0)
    # The meta node should have 'walking' in its narrative_tags JSON
    row = conn.execute(
        "SELECT narrative_tags FROM memory_nodes WHERE node_level=4 AND video_id=?",
        (vid,),
    ).fetchone()
    tags = json.loads(row["narrative_tags"])
    assert "walking" in tags
