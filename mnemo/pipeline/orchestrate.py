"""Pipeline coordinator. Runs download -> metadata -> frames -> audio -> motion
against a single video and transitions status columns along the way.
"""
from __future__ import annotations
import logging
import shutil
import sqlite3
from dataclasses import dataclass

from mnemo.config import Settings
from mnemo.db import now_ms
from mnemo.memory.builder import build_memory_tree
from mnemo.pipeline.audio import extract_audio, segment_audio, AudioError
from mnemo.pipeline.download import download
from mnemo.pipeline.frames import extract_frames, read_metadata, VideoMetadata
from mnemo.pipeline.motion import extract_motion, MotionExtractor

log = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    video_id: str
    status: str             # 'completed' | 'failed'
    error: str | None
    metadata: VideoMetadata | None
    frame_count: int
    audio_segment_count: int
    motion_frame_count: int


def _set_status(conn: sqlite3.Connection, video_id: str,
                status: str | None = None,
                motion_status: str | None = None,
                gapper_status: str | None = None) -> None:
    sets, params = [], []
    if status is not None:
        sets.append("status = ?")
        params.append(status)
    if motion_status is not None:
        sets.append("motion_status = ?")
        params.append(motion_status)
    if gapper_status is not None:
        sets.append("gapper_status = ?")
        params.append(gapper_status)
    if not sets:
        return
    params.append(video_id)
    conn.execute(
        f"UPDATE video_metadata SET {', '.join(sets)} WHERE video_id = ?", params
    )


def _update_metadata(conn: sqlite3.Connection, video_id: str,
                     md: VideoMetadata) -> None:
    conn.execute(
        "UPDATE video_metadata SET duration_seconds = ?, fps = ?, "
        "width = ?, height = ?, processed_at = ? WHERE video_id = ?",
        (md.duration_seconds, md.fps, md.width, md.height, now_ms(), video_id),
    )


def run_pipeline(
    video_id: str,
    video_url: str,
    conn: sqlite3.Connection,
    settings: Settings,
    motion_extractor: MotionExtractor | None = None,
    downloader=None,  # injectable for tests; resolved at call time
) -> PipelineResult:
    work_dir = settings.work_dir / video_id
    work_dir.mkdir(parents=True, exist_ok=True)
    video_path = work_dir / "source.mp4"
    frames_dir = work_dir / "frames"
    audio_dir = work_dir / "audio"
    segments_dir = audio_dir / "segments"

    if downloader is None:
        downloader = download

    metadata: VideoMetadata | None = None
    frame_count = 0
    audio_count = 0
    motion_count = 0

    try:
        # 1. Download
        _set_status(conn, video_id, status="downloading")
        cookies = settings.cookies_path if settings.cookies_path.exists() else None
        downloader(video_url, video_path, cookies_path=cookies)

        # 2. Metadata
        metadata = read_metadata(video_path)
        _update_metadata(conn, video_id, metadata)

        # 3. Frames
        _set_status(conn, video_id, status="processing")
        records = extract_frames(
            video_path, video_id, frames_dir,
            settings.frame_sample_rate_fps, conn,
        )
        frame_count = len(records)

        # 4. Audio (soft-fail: no audio track shouldn't kill the run)
        try:
            audio_info = extract_audio(video_path, audio_dir)
            if audio_info is not None:
                segments = segment_audio(
                    audio_info, video_id, segments_dir,
                    settings.audio_segment_seconds, conn,
                )
                audio_count = len(segments)
            else:
                log.info("orchestrate: no audio track in %s", video_id)
        except AudioError as exc:
            log.warning("orchestrate: audio failed for %s: %s", video_id, exc)
            # Continue — audio is non-fatal

        # 5. Motion
        _set_status(conn, video_id, motion_status="processing")
        if metadata.fps > 0 and frame_count > 0:
            motion_count = extract_motion(
                video_id, frames_dir, metadata.fps, conn,
                extractor=motion_extractor,
            )
            _set_status(conn, video_id, motion_status="completed")
        else:
            _set_status(conn, video_id, motion_status="skipped")

        # 6. Memory tree
        _set_status(conn, video_id, gapper_status="processing")
        try:
            node_count = build_memory_tree(conn, video_id, metadata.duration_seconds)
            if node_count > 0:
                _set_status(conn, video_id, gapper_status="completed")
                log.info("orchestrate: built %d memory nodes for %s",
                         node_count, video_id)
            else:
                _set_status(conn, video_id, gapper_status="skipped")
        except Exception as exc:
            log.warning("orchestrate: memory tree build failed for %s: %s",
                        video_id, exc)
            _set_status(conn, video_id, gapper_status="failed")

        # 7. Cleanup source MP4 (keep frames + audio)
        if video_path.exists():
            video_path.unlink()

        _set_status(conn, video_id, status="completed")
        return PipelineResult(
            video_id=video_id, status="completed", error=None,
            metadata=metadata, frame_count=frame_count,
            audio_segment_count=audio_count, motion_frame_count=motion_count,
        )

    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
        log.error("orchestrate: pipeline failed for %s: %s", video_id, err)
        conn.execute(
            "UPDATE video_metadata SET status='failed', "
            "motion_status = CASE WHEN motion_status IN ('pending','processing') "
            "THEN 'failed' ELSE motion_status END, "
            "gapper_status = CASE WHEN gapper_status IN ('pending','processing') "
            "THEN 'failed' ELSE gapper_status END "
            "WHERE video_id = ?",
            (video_id,),
        )
        # Clean up the whole work_dir on failure
        if work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)
        return PipelineResult(
            video_id=video_id, status="failed", error=err,
            metadata=metadata, frame_count=frame_count,
            audio_segment_count=audio_count, motion_frame_count=motion_count,
        )
