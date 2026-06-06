"""Worker process. Polls processing_queue, claims tasks atomically,
drives the pipeline, marks tasks completed/failed.
"""
from __future__ import annotations
import logging
import sqlite3
import time
from dataclasses import dataclass
from typing import Callable

from mnemo.config import Settings, get_settings
from mnemo.db import init_for_settings, now_ms
from mnemo.pipeline.motion import MotionExtractor
from mnemo.pipeline.orchestrate import run_pipeline

log = logging.getLogger(__name__)


@dataclass
class ClaimedTask:
    task_id: int
    video_id: str
    video_url: str


def claim_one_task(conn: sqlite3.Connection, worker_id: str) -> ClaimedTask | None:
    """Atomically claim the highest-priority pending task. Returns None if empty."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        row = conn.execute(
            "SELECT q.id, q.video_id, vm.filename AS url "
            "FROM processing_queue q "
            "JOIN video_metadata vm ON vm.video_id = q.video_id "
            "WHERE q.status = 'pending' "
            "ORDER BY q.priority DESC, q.created_at ASC LIMIT 1"
        ).fetchone()
        if row is None:
            conn.execute("ROLLBACK")
            return None
        conn.execute(
            "UPDATE processing_queue SET status='assigned', assigned_to=?, "
            "started_at=? WHERE id=?",
            (worker_id, now_ms(), row["id"]),
        )
        conn.execute("COMMIT")
        return ClaimedTask(task_id=row["id"], video_id=row["video_id"],
                           video_url=row["url"])
    except Exception:
        conn.execute("ROLLBACK")
        raise


def mark_task_complete(conn: sqlite3.Connection, task_id: int) -> None:
    conn.execute(
        "UPDATE processing_queue SET status='completed', completed_at=? WHERE id=?",
        (now_ms(), task_id),
    )


def mark_task_failed(conn: sqlite3.Connection, task_id: int, error: str) -> None:
    conn.execute(
        "UPDATE processing_queue SET status='failed', completed_at=?, "
        "error_message=? WHERE id=?",
        (now_ms(), error[:1000], task_id),
    )


def run_one_iteration(
    conn: sqlite3.Connection, settings: Settings,
    motion_extractor: MotionExtractor,
) -> bool:
    """Try to process exactly one task. Returns True if work was done."""
    task = claim_one_task(conn, settings.worker_id)
    if task is None:
        return False
    log.info("worker: claimed task %d (video %s)", task.task_id, task.video_id)
    try:
        result = run_pipeline(
            task.video_id, task.video_url, conn, settings,
            motion_extractor=motion_extractor,
        )
        if result.status == "completed":
            mark_task_complete(conn, task.task_id)
            log.info("worker: task %d completed", task.task_id)
        else:
            mark_task_failed(conn, task.task_id, result.error or "unknown")
            log.warning("worker: task %d failed: %s", task.task_id, result.error)
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
        log.exception("worker: unexpected error on task %d", task.task_id)
        mark_task_failed(conn, task.task_id, err)
    return True


def run_worker(
    settings: Settings | None = None,
    motion_extractor: MotionExtractor | None = None,
    idle_callback: Callable[[], None] | None = None,
) -> None:
    settings = settings or get_settings()
    conn = init_for_settings(settings)
    motion = motion_extractor or MotionExtractor()
    backoff = settings.worker_poll_interval_seconds
    max_backoff = 60.0
    log.info("worker %s starting", settings.worker_id)
    try:
        while True:
            try:
                did_work = run_one_iteration(conn, settings, motion)
                if did_work:
                    backoff = settings.worker_poll_interval_seconds  # reset
                else:
                    if idle_callback is not None:
                        idle_callback()
                    time.sleep(settings.worker_poll_interval_seconds)
            except Exception:
                log.exception("worker: outer loop error, backing off %.1fs", backoff)
                time.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)
    except KeyboardInterrupt:
        log.info("worker: shutdown requested")
    finally:
        motion.close()
        conn.close()
