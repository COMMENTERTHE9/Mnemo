from pathlib import Path
import pytest

from mnemo.db import init_for_settings, enqueue_video
from mnemo.pipeline.frames import read_metadata, extract_frames


def test_read_metadata(synthetic_video):
    md = read_metadata(synthetic_video)
    assert md.fps > 0
    assert md.frame_count == 90
    assert md.width == 320 and md.height == 240
    assert 2.5 < md.duration_seconds < 3.5


def test_extract_frames_writes_jpgs_and_gapper_reports(settings, synthetic_video, tmp_path):
    conn = init_for_settings(settings)
    video_id = enqueue_video(conn, "test://synthetic")
    frames_dir = tmp_path / "frames"

    records = extract_frames(
        video_path=synthetic_video, video_id=video_id,
        frames_dir=frames_dir, sample_rate_fps=1.0, conn=conn,
    )

    # 3-second video at 1 fps target should yield ~3 frames
    assert 2 <= len(records) <= 4
    for r in records:
        assert r.frame_path.exists()
        assert r.blur_variance >= 0.0

    rows = conn.execute(
        "SELECT COUNT(*) AS n FROM gapper_reports WHERE video_id=? AND gapper_type='frame'",
        (video_id,),
    ).fetchone()
    assert rows["n"] == len(records)


def test_extract_frames_creates_missing_dir(settings, synthetic_video, tmp_path):
    conn = init_for_settings(settings)
    video_id = enqueue_video(conn, "test://x")
    frames_dir = tmp_path / "deeply" / "nested" / "frames"
    assert not frames_dir.exists()
    records = extract_frames(synthetic_video, video_id, frames_dir, 1.0, conn)
    assert frames_dir.exists()
    assert len(records) > 0
