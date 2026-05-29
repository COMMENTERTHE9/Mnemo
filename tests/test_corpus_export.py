import json

from mnemo.db import init_for_settings, enqueue_video, insert_gapper_report
from mnemo.corpus.export import export_tree, write_export
from mnemo.memory.builder import build_memory_tree
from mnemo.models import GapperReport


def _seed(conn, video_id, seconds=19):
    for i in range(seconds):
        ts_ms = int(i * 1000)
        insert_gapper_report(conn, GapperReport(
            video_id=video_id, gapper_type="frame", timestamp_ms=ts_ms,
            gapper_id=f"f_{i}", start_frame=i, end_frame=i,
            summary="", importance=0.5, features={"blur_variance": 150.0},
        ))
        insert_gapper_report(conn, GapperReport(
            video_id=video_id, gapper_type="motion", timestamp_ms=ts_ms,
            gapper_id=f"m_{i}", start_frame=i, end_frame=i,
            summary="", importance=0.7,
            features={"motion_features": {
                "total_movement": 0.1234,
                "action_hints": ["walking"] if i % 3 == 0 else [],
            }},
        ))
        insert_gapper_report(conn, GapperReport(
            video_id=video_id, gapper_type="audio", timestamp_ms=ts_ms,
            gapper_id=f"a_{i}", start_frame=0, end_frame=0,
            summary="Audio segment", importance=0.42,
            features={"rms": 0.013579, "dbfs": -37.34},
        ))


def test_export_tree_structure_and_links(settings):
    conn = init_for_settings(settings)
    vid = enqueue_video(conn, "test://x")
    _seed(conn, vid, seconds=19)
    build_memory_tree(conn, vid, video_duration=19.0)

    export = export_tree(conn, vid)

    assert export["video_id"] == vid
    # All four tree levels present (segment/scene/chapter/meta).
    levels = {n["node_level"] for n in export["tree"]}
    assert levels == {1, 2, 3, 4}

    # Tree sorted by node_level desc then start_time asc.
    keys = [(n["node_level"], n["start_time"]) for n in export["tree"]]
    assert keys == sorted(keys, key=lambda k: (-k[0], k[1]))

    # parent_id links are intact: meta has no parent; every other node's
    # parent_id resolves to a node that exists in the export.
    ids = {n["node_id"] for n in export["tree"]}
    meta = [n for n in export["tree"] if n["node_level"] == 4]
    assert len(meta) == 1 and meta[0]["parent_id"] is None
    for n in export["tree"]:
        if n["node_level"] != 4:
            assert n["parent_id"] in ids


def test_export_features_are_dicts_not_strings(settings):
    conn = init_for_settings(settings)
    vid = enqueue_video(conn, "test://x")
    _seed(conn, vid, seconds=19)
    build_memory_tree(conn, vid, video_duration=19.0)

    export = export_tree(conn, vid)
    assert export["signals"], "expected signal rows"
    for s in export["signals"]:
        assert isinstance(s["features"], dict)  # parsed, not raw JSON string

    # Signals sorted by timestamp ascending.
    ts = [s["timestamp"] for s in export["signals"]]
    assert ts == sorted(ts)

    # Raw numerics preserved exactly (no rounding/reformatting).
    audio = [s for s in export["signals"] if s["gapper_type"] == "audio"]
    assert audio and audio[0]["features"]["rms"] == 0.013579
    assert audio[0]["features"]["dbfs"] == -37.34
    motion = [s for s in export["signals"] if s["gapper_type"] == "motion"]
    assert motion and motion[0]["features"]["motion_features"]["total_movement"] == 0.1234

    # narrative_tags on the tree come back as a list, not a string.
    for n in export["tree"]:
        assert isinstance(n["narrative_tags"], list)


def test_write_export_roundtrips_to_file(settings, tmp_path):
    conn = init_for_settings(settings)
    vid = enqueue_video(conn, "test://x")
    _seed(conn, vid, seconds=19)
    build_memory_tree(conn, vid, video_duration=19.0)

    out_path = write_export(conn, vid, out_dir=tmp_path / "corpus")
    assert out_path.exists()

    loaded = json.loads(out_path.read_text(encoding="utf-8"))
    assert loaded == export_tree(conn, vid)
    assert loaded["video_id"] == vid
    assert {n["node_level"] for n in loaded["tree"]} == {1, 2, 3, 4}
