from pathlib import Path

from mnemo.db import init_for_settings, enqueue_video, get_video_status
from mnemo.models import GapperReport, MemoryNode
from mnemo.db import insert_gapper_report, insert_memory_node, top_memory_nodes


def test_schema_initializes(settings):
    conn = init_for_settings(settings)
    tables = {r["name"] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    )}
    assert {"video_metadata", "processing_queue", "gapper_reports",
             "memory_nodes", "peripheral_detections"} <= tables


def test_enqueue_creates_metadata_and_queue_row(settings):
    conn = init_for_settings(settings)
    vid = enqueue_video(conn, "https://youtube.com/watch?v=test")
    assert vid.startswith("video_")
    assert get_video_status(conn, vid) == "pending"
    q = conn.execute("SELECT task_type, status FROM processing_queue").fetchone()
    assert q["task_type"] == "download" and q["status"] == "pending"


def test_gapper_and_memory_roundtrip(settings):
    conn = init_for_settings(settings)
    vid = enqueue_video(conn, "https://example.com/x.mp4")
    insert_gapper_report(conn, GapperReport(
        video_id=vid, gapper_type="frame", timestamp_ms=1000,
        gapper_id="g1", start_frame=0, end_frame=0, importance=0.7,
    ))
    insert_memory_node(conn, MemoryNode(
        video_id=vid, node_level=4, node_id="root",
        start_time=0.0, end_time=10.0, summary="test", importance=0.9,
    ))
    top = top_memory_nodes(conn, vid)
    assert len(top) == 1 and top[0]["node_id"] == "root"
