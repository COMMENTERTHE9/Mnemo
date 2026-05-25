"""SQLite access layer. Single connection per process for Sprint 0;
upgrade to a pool when worker concurrency demands it."""
from __future__ import annotations
import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from mnemo.config import Settings
from mnemo.models import GapperReport, MemoryNode

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "database" / "schema.sql"


def now_ms() -> int:
    return time.time_ns() // 1_000_000


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), isolation_level=None)  # autocommit
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    sql = SCHEMA_PATH.read_text()
    conn.executescript(sql)


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    conn.execute("BEGIN")
    try:
        yield conn
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise


def enqueue_video(conn: sqlite3.Connection, video_url: str) -> str:
    video_id = f"video_{time.time_ns()}"
    created = now_ms()
    with transaction(conn):
        conn.execute(
            "INSERT INTO video_metadata (video_id, filename, status, created_at) "
            "VALUES (?, ?, 'pending', ?)",
            (video_id, video_url, created),
        )
        conn.execute(
            "INSERT INTO processing_queue (video_id, task_type, priority, status, created_at) "
            "VALUES (?, 'download', 10, 'pending', ?)",
            (video_id, created),
        )
    return video_id


def get_video_status(conn: sqlite3.Connection, video_id: str) -> str | None:
    row = conn.execute(
        "SELECT status FROM video_metadata WHERE video_id = ?", (video_id,)
    ).fetchone()
    return row["status"] if row else None


def insert_gapper_report(conn: sqlite3.Connection, report: GapperReport) -> None:
    conn.execute(
        "INSERT INTO gapper_reports (video_id, gapper_type, timestamp, gapper_id, "
        "start_frame, end_frame, summary, importance, features) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            report.video_id, report.gapper_type, report.timestamp_ms,
            report.gapper_id, report.start_frame, report.end_frame,
            report.summary, report.importance, json.dumps(report.features),
        ),
    )


def insert_memory_node(conn: sqlite3.Connection, node: MemoryNode) -> None:
    conn.execute(
        "INSERT INTO memory_nodes (video_id, node_level, node_id, parent_id, "
        "start_time, end_time, summary, importance, narrative_tags, "
        "deleted_by_ai, compression_data) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            node.video_id, node.node_level, node.node_id, node.parent_id,
            node.start_time, node.end_time, node.summary, node.importance,
            json.dumps(node.narrative_tags), node.deleted_by_ai, node.compression_data,
        ),
    )


def top_memory_nodes(
    conn: sqlite3.Connection, video_id: str, limit: int = 10, min_importance: float = 0.3
) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT node_id, summary, importance, start_time FROM memory_nodes "
        "WHERE video_id = ? AND importance > ? ORDER BY importance DESC LIMIT ?",
        (video_id, min_importance, limit),
    ).fetchall()


def init_for_settings(settings: Settings) -> sqlite3.Connection:
    settings.ensure_dirs()
    conn = connect(settings.db_path)
    init_schema(conn)
    return conn
