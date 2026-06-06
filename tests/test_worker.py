import shutil
from types import SimpleNamespace
from unittest.mock import patch

from mnemo.db import init_for_settings, enqueue_video
from mnemo.pipeline.motion import MotionExtractor
from mnemo.worker.runner import (
    claim_one_task, mark_task_complete, mark_task_failed, run_one_iteration,
)


class FakeBackend:
    def process(self, rgb):
        return SimpleNamespace(
            pose_landmarks=None, face_landmarks=None,
            left_hand_landmarks=None, right_hand_landmarks=None,
        )
    def close(self):
        pass


def test_claim_returns_none_when_queue_empty(settings):
    conn = init_for_settings(settings)
    assert claim_one_task(conn, "w1") is None


def test_claim_returns_task_and_marks_assigned(settings):
    conn = init_for_settings(settings)
    video_id = enqueue_video(conn, "https://example.com/x")
    task = claim_one_task(conn, "w1")
    assert task is not None and task.video_id == video_id

    row = conn.execute(
        "SELECT status, assigned_to FROM processing_queue WHERE id=?",
        (task.task_id,),
    ).fetchone()
    assert row["status"] == "assigned" and row["assigned_to"] == "w1"


def test_claim_does_not_double_claim(settings):
    conn = init_for_settings(settings)
    enqueue_video(conn, "https://example.com/x")
    first = claim_one_task(conn, "w1")
    second = claim_one_task(conn, "w2")
    assert first is not None
    assert second is None  # already assigned


def test_mark_complete_and_failed(settings):
    conn = init_for_settings(settings)
    enqueue_video(conn, "https://example.com/x")
    task = claim_one_task(conn, "w1")
    mark_task_complete(conn, task.task_id)
    row = conn.execute(
        "SELECT status FROM processing_queue WHERE id=?", (task.task_id,),
    ).fetchone()
    assert row["status"] == "completed"

    enqueue_video(conn, "https://example.com/y")
    t2 = claim_one_task(conn, "w1")
    mark_task_failed(conn, t2.task_id, "kaboom")
    row = conn.execute(
        "SELECT status, error_message FROM processing_queue WHERE id=?",
        (t2.task_id,),
    ).fetchone()
    assert row["status"] == "failed" and row["error_message"] == "kaboom"


def test_run_one_iteration_empty_returns_false(settings):
    conn = init_for_settings(settings)
    motion = MotionExtractor(_backend=FakeBackend())
    assert run_one_iteration(conn, settings, motion) is False


def test_run_one_iteration_processes_synthetic(settings, synthetic_video):
    """Full e2e: enqueue -> worker claims -> pipeline runs -> DB state is final."""
    from mnemo.pipeline.download import DownloadResult
    conn = init_for_settings(settings)
    video_id = enqueue_video(conn, "https://example.com/x.mp4")

    def stub_download(url, out_path, cookies_path=None, ytdlp_bin="yt-dlp"):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(synthetic_video, out_path)
        return DownloadResult(video_path=out_path, source_url=url)

    motion = MotionExtractor(_backend=FakeBackend())
    with patch("mnemo.pipeline.orchestrate.download", side_effect=stub_download):
        did = run_one_iteration(conn, settings, motion)
    assert did is True

    # Task marked completed
    q = conn.execute(
        "SELECT status FROM processing_queue WHERE video_id=?", (video_id,),
    ).fetchone()
    assert q["status"] == "completed"

    # Video reached completed
    v = conn.execute(
        "SELECT status, motion_status FROM video_metadata WHERE video_id=?",
        (video_id,),
    ).fetchone()
    assert v["status"] == "completed"
    assert v["motion_status"] == "completed"

    # Gapper reports written: frame + motion (audio may be 0)
    types = {r["gapper_type"] for r in conn.execute(
        "SELECT DISTINCT gapper_type FROM gapper_reports WHERE video_id=?",
        (video_id,),
    )}
    assert "frame" in types and "motion" in types

    # Sprint 2: memory_nodes populated
    mn = conn.execute(
        "SELECT COUNT(*) AS n FROM memory_nodes WHERE video_id=?", (video_id,),
    ).fetchone()
    assert mn["n"] >= 4  # at least segment + scene + chapter + meta
