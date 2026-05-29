"""Memory-tree export to portable JSON.

Produces a single self-contained dict per video capturing the full memory
tree (every memory_nodes row, all levels) plus the raw per-frame signals
(every gapper_reports row, features parsed from JSON). This is the dataset
container for downstream model work, so RAW NUMERIC fields are preserved
exactly — importance, and every numeric inside `features` (motion
total_movement, audio rms/dbfs, landmark visibility, etc.) — never rounded,
reformatted, or dropped in favor of the human-readable summary strings.
"""
from __future__ import annotations
import json
import sqlite3
from pathlib import Path
from typing import Any


def _parse_json(blob: str | None, fallback: Any) -> Any:
    if not blob:
        return fallback
    try:
        return json.loads(blob)
    except (json.JSONDecodeError, TypeError):
        return fallback


def export_tree(conn: sqlite3.Connection, video_id: str) -> dict[str, Any]:
    """Build a self-contained dict for `video_id`: full tree + raw signals.

    Numeric values are passed through untouched. `narrative_tags` (tree) and
    `features` (signals) are parsed from their stored JSON text into real
    Python structures so the export round-trips as native JSON, not strings.
    """
    meta = conn.execute(
        "SELECT duration_seconds FROM video_metadata WHERE video_id = ?",
        (video_id,),
    ).fetchone()
    duration = meta["duration_seconds"] if meta is not None else None

    tree: list[dict[str, Any]] = []
    for row in conn.execute(
        "SELECT node_id, node_level, parent_id, start_time, end_time, "
        "importance, summary, narrative_tags FROM memory_nodes "
        "WHERE video_id = ? ORDER BY node_level DESC, start_time ASC",
        (video_id,),
    ):
        tree.append({
            "node_id": row["node_id"],
            "node_level": row["node_level"],
            "parent_id": row["parent_id"],
            "start_time": row["start_time"],
            "end_time": row["end_time"],
            "importance": row["importance"],
            "summary": row["summary"],
            "narrative_tags": _parse_json(row["narrative_tags"], []),
        })

    signals: list[dict[str, Any]] = []
    for row in conn.execute(
        "SELECT gapper_type, timestamp, gapper_id, start_frame, end_frame, "
        "importance, summary, features FROM gapper_reports "
        "WHERE video_id = ? ORDER BY timestamp ASC",
        (video_id,),
    ):
        signals.append({
            "gapper_type": row["gapper_type"],
            "timestamp": row["timestamp"],
            "gapper_id": row["gapper_id"],
            "start_frame": row["start_frame"],
            "end_frame": row["end_frame"],
            "importance": row["importance"],
            "summary": row["summary"],
            "features": _parse_json(row["features"], {}),
        })

    return {
        "video_id": video_id,
        "duration_seconds": duration,
        "tree": tree,
        "signals": signals,
    }


def write_export(
    conn: sqlite3.Connection, video_id: str,
    out_dir: Path = Path("data/corpus"),
) -> Path:
    """Export `video_id` and write it pretty-printed UTF-8 JSON to
    out_dir/<video_id>.json. Returns the written path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{video_id}.json"
    payload = export_tree(conn, video_id)
    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    return out_path
