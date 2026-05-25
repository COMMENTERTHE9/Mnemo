"""Frame extraction.

Replaces worker/video_processor.py:101-156. Differences:
  - Caller supplies the SQLite connection (no connect-per-row N+1)
  - Configurable output directory
  - Returns a list of FrameRecord dataclasses for downstream gappers
  - Importance is normalized to [0, 1] via 1000-cap (matches legacy line 324)
  - Writes gapper_reports in a single transaction
"""
from __future__ import annotations
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import cv2

from mnemo.db import insert_gapper_report, transaction
from mnemo.models import GapperReport

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class VideoMetadata:
    fps: float
    frame_count: int
    width: int
    height: int
    duration_seconds: float


@dataclass(frozen=True)
class FrameRecord:
    frame_number: int
    timestamp_seconds: float
    frame_path: Path
    blur_variance: float  # Raw Laplacian variance (pre-normalization)


def read_metadata(video_path: Path) -> VideoMetadata:
    cap = cv2.VideoCapture(str(video_path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration = (frames / fps) if fps > 0 else 0.0
        return VideoMetadata(fps=fps, frame_count=frames, width=width,
                             height=height, duration_seconds=duration)
    finally:
        cap.release()


def extract_frames(
    video_path: Path,
    video_id: str,
    frames_dir: Path,
    sample_rate_fps: float,
    conn: sqlite3.Connection,
) -> list[FrameRecord]:
    """Sample frames at `sample_rate_fps`, save to disk, write gapper_reports.

    Args:
        video_path: Source video file.
        video_id: Video identifier (for gapper_reports rows).
        frames_dir: Directory where JPGs are written. Created if missing.
        sample_rate_fps: How many frames per source second to keep.
        conn: SQLite connection to write reports through.

    Returns:
        List of FrameRecord, one per saved frame.
    """
    frames_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    try:
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        interval = max(1, int(src_fps / sample_rate_fps))
        log.info("frames: sampling every %d source frames (src_fps=%.2f, target=%.2f)",
                 interval, src_fps, sample_rate_fps)

        records: list[FrameRecord] = []
        reports: list[GapperReport] = []
        frame_number = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_number % interval == 0:
                timestamp = frame_number / src_fps if src_fps > 0 else 0.0
                frame_path = frames_dir / f"frame_{frame_number:06d}.jpg"
                cv2.imwrite(str(frame_path), frame)

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                importance = min(variance / 1000.0, 1.0)

                records.append(FrameRecord(
                    frame_number=frame_number,
                    timestamp_seconds=timestamp,
                    frame_path=frame_path,
                    blur_variance=variance,
                ))
                reports.append(GapperReport(
                    video_id=video_id,
                    gapper_type="frame",
                    timestamp_ms=int(timestamp * 1000),
                    gapper_id=f"frame_gapper_{frame_number}",
                    start_frame=frame_number,
                    end_frame=frame_number,
                    summary=f"Frame at {timestamp:.2f}s",
                    importance=importance,
                    features={
                        "blur_variance": variance,
                        "has_content": variance > 100.0,
                    },
                ))
            frame_number += 1
    finally:
        cap.release()

    with transaction(conn):
        for r in reports:
            insert_gapper_report(conn, r)

    log.info("frames: extracted %d frames from %d total", len(records), frame_number)
    return records
