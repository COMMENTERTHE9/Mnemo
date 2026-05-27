"""Frame gapper: load and combine per-frame data from frame, motion,
and audio gapper_reports into UnifiedFrame records."""
from __future__ import annotations
import json
import sqlite3

from mnemo.gappers.base import (
    UnifiedFrame, FRAME_WEIGHT, MOTION_WEIGHT, AUDIO_WEIGHT,
)


def load_unified_frames(conn: sqlite3.Connection, video_id: str) -> list[UnifiedFrame]:
    """Pull all per-frame gapper_reports for a video and merge into a
    sorted list of UnifiedFrame records.

    - frame gapper: keyed by start_frame, gives blur_variance + frame importance
    - motion gapper: keyed by start_frame, gives total_movement + actions
    - audio gapper: keyed by floor(timestamp_ms/1000), gives audio importance
    """
    frames: dict[int, dict] = {}
    for row in conn.execute(
        "SELECT start_frame, timestamp, importance, features FROM gapper_reports "
        "WHERE video_id=? AND gapper_type='frame'",
        (video_id,),
    ):
        feats = json.loads(row["features"] or "{}")
        frames[row["start_frame"]] = {
            "timestamp_seconds": (row["timestamp"] or 0) / 1000.0,
            "frame_importance": row["importance"] or 0.0,
            "blur_variance": float(feats.get("blur_variance", 0.0)),
        }

    motions: dict[int, dict] = {}
    for row in conn.execute(
        "SELECT start_frame, importance, features FROM gapper_reports "
        "WHERE video_id=? AND gapper_type='motion'",
        (video_id,),
    ):
        feats = json.loads(row["features"] or "{}")
        mf = feats.get("motion_features") if isinstance(feats, dict) else None
        mf = mf if isinstance(mf, dict) else {}
        motions[row["start_frame"]] = {
            "importance": row["importance"] or 0.0,
            "total_movement": float(mf.get("total_movement", 0.0) or 0.0),
            "actions": list(mf.get("action_hints", []) or []),
        }

    audio_by_sec: dict[int, float] = {}
    for row in conn.execute(
        "SELECT timestamp, importance FROM gapper_reports "
        "WHERE video_id=? AND gapper_type='audio'",
        (video_id,),
    ):
        sec = int((row["timestamp"] or 0) / 1000)
        audio_by_sec[sec] = max(audio_by_sec.get(sec, 0.0), row["importance"] or 0.0)

    unified: list[UnifiedFrame] = []
    for fn in sorted(frames):
        f = frames[fn]
        m = motions.get(fn, {})
        ts_sec = int(f["timestamp_seconds"])
        aud_imp = audio_by_sec.get(ts_sec, 0.0)

        combined = (
            FRAME_WEIGHT * f["frame_importance"]
            + MOTION_WEIGHT * m.get("importance", 0.0)
            + AUDIO_WEIGHT * aud_imp
        )
        unified.append(UnifiedFrame(
            frame_number=fn,
            timestamp_seconds=f["timestamp_seconds"],
            importance=min(max(combined, 0.0), 1.0),
            blur_variance=f["blur_variance"],
            motion_magnitude=m.get("total_movement", 0.0),
            has_audio=ts_sec in audio_by_sec,
            actions=list(m.get("actions", [])),
        ))
    return unified
