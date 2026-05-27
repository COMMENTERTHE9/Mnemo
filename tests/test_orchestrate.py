import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from mnemo.db import init_for_settings, enqueue_video
from mnemo.pipeline.orchestrate import run_pipeline


class FakeBackend:
    def process(self, rgb):
        return SimpleNamespace(
            pose_landmarks=None, face_landmarks=None,
            left_hand_landmarks=None, right_hand_landmarks=None,
        )
    def close(self):
        pass


@pytest.fixture
def fake_motion():
    from mnemo.pipeline.motion import MotionExtractor
    return MotionExtractor(_backend=FakeBackend())


def _stub_downloader(synthetic_video: Path):
    """Returns a download() replacement that copies synthetic_video to the
    target out_path."""
    def stub(url, out_path, cookies_path=None, ytdlp_bin="yt-dlp"):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(synthetic_video, out_path)
        from mnemo.pipeline.download import DownloadResult
        return DownloadResult(video_path=out_path, source_url=url)
    return stub


def test_pipeline_happy_path(settings, synthetic_video, fake_motion):
    conn = init_for_settings(settings)
    video_id = enqueue_video(conn, "https://example.com/x.mp4")

    result = run_pipeline(
        video_id, "https://example.com/x.mp4", conn, settings,
        motion_extractor=fake_motion,
        downloader=_stub_downloader(synthetic_video),
    )
    assert result.status == "completed"
    assert result.error is None
    assert result.frame_count > 0
    assert result.motion_frame_count == result.frame_count

    # video_metadata transitioned correctly
    row = conn.execute(
        "SELECT status, motion_status, fps, duration_seconds FROM video_metadata WHERE video_id=?",
        (video_id,),
    ).fetchone()
    assert row["status"] == "completed"
    assert row["motion_status"] == "completed"
    assert row["fps"] > 0
    assert row["duration_seconds"] > 0

    # Source MP4 deleted, frames preserved
    assert not (settings.work_dir / video_id / "source.mp4").exists()
    assert (settings.work_dir / video_id / "frames").exists()

    # Sprint 2: memory tree was built
    rows = conn.execute(
        "SELECT node_level, COUNT(*) AS n FROM memory_nodes "
        "WHERE video_id=? GROUP BY node_level",
        (video_id,),
    ).fetchall()
    by_level = {r["node_level"]: r["n"] for r in rows}
    assert by_level.get(4, 0) == 1   # exactly one meta node
    assert by_level.get(3, 0) >= 1   # at least one chapter
    g = conn.execute(
        "SELECT gapper_status FROM video_metadata WHERE video_id=?",
        (video_id,),
    ).fetchone()
    assert g["gapper_status"] == "completed"


def test_pipeline_download_failure_marks_failed_and_cleans(settings, fake_motion):
    from mnemo.pipeline.download import DownloadError
    conn = init_for_settings(settings)
    video_id = enqueue_video(conn, "https://example.com/x.mp4")

    def broken_downloader(url, out_path, cookies_path=None, ytdlp_bin="yt-dlp"):
        raise DownloadError("simulated network failure")

    result = run_pipeline(
        video_id, "https://example.com/x.mp4", conn, settings,
        motion_extractor=fake_motion, downloader=broken_downloader,
    )
    assert result.status == "failed"
    assert "simulated network failure" in result.error

    row = conn.execute(
        "SELECT status FROM video_metadata WHERE video_id=?", (video_id,),
    ).fetchone()
    assert row["status"] == "failed"

    # work_dir cleaned up on failure
    assert not (settings.work_dir / video_id).exists()


def test_pipeline_no_audio_track_still_completes(settings, synthetic_video, fake_motion):
    # Synthetic video has no audio. Pipeline should still complete with 0 segments.
    conn = init_for_settings(settings)
    video_id = enqueue_video(conn, "https://example.com/x.mp4")
    result = run_pipeline(
        video_id, "https://example.com/x.mp4", conn, settings,
        motion_extractor=fake_motion,
        downloader=_stub_downloader(synthetic_video),
    )
    assert result.status == "completed"
    assert result.audio_segment_count == 0
