"""Audio extraction and segmentation via ffmpeg.

Replaces worker/video_processor.py:158-264. Differences:
  - Caller supplies the DB connection (no connect-per-segment N+1)
  - Configurable output directory
  - Two clean functions (extract, segment) instead of intertwined logic
  - Returns dataclasses for downstream consumption
  - Raises typed AudioError on ffmpeg failure
"""
from __future__ import annotations
import logging
import sqlite3
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mnemo.db import insert_gapper_report, transaction
from mnemo.models import GapperReport

log = logging.getLogger(__name__)


def _segment_loudness(wav_path: Path) -> tuple[float, float]:
    """RMS and dBFS for a 16kHz mono pcm_s16le WAV.

    Returns (rms, dbfs): rms normalized to [0,1]; dbfs = 20*log10(rms)
    floored at -80.0. Silence or unreadable -> (0.0, -80.0).
    """
    try:
        with wave.open(str(wav_path), "rb") as w:
            raw = w.readframes(w.getnframes())
    except (wave.Error, OSError):
        return 0.0, -80.0
    if not raw:
        return 0.0, -80.0
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0
    if samples.size == 0:
        return 0.0, -80.0
    rms = float(np.sqrt(np.mean(samples ** 2)))
    if rms <= 0.0:
        return 0.0, -80.0
    return rms, max(-80.0, 20.0 * float(np.log10(rms)))


def _loudness_to_importance(dbfs: float) -> float:
    """Map dBFS to [0,1]: -60 dBFS (near silence) -> 0, 0 dBFS -> 1."""
    return min(1.0, max(0.0, (dbfs + 60.0) / 60.0))


class AudioError(RuntimeError):
    """Raised when ffmpeg fails or output is unusable."""


@dataclass(frozen=True)
class AudioInfo:
    path: Path
    duration_seconds: float
    sample_rate: int
    channels: int


@dataclass(frozen=True)
class AudioSegment:
    segment_number: int
    timestamp_seconds: float
    duration_seconds: float
    path: Path


def extract_audio(
    video_path: Path,
    audio_dir: Path,
    ffmpeg_bin: str = "ffmpeg",
) -> AudioInfo | None:
    """Extract a 16kHz mono WAV from `video_path`.

    Returns None if the source has no audio track. Raises AudioError on
    hard failures (binary missing, write error).
    """
    audio_dir.mkdir(parents=True, exist_ok=True)
    audio_path = audio_dir / "full_audio.wav"
    cmd = [
        ffmpeg_bin, "-i", str(video_path),
        "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        "-y", str(audio_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise AudioError(f"ffmpeg binary not found: {ffmpeg_bin}") from exc

    if result.returncode != 0:
        # No audio stream is a soft failure — ffmpeg reports it via stderr
        stderr = result.stderr or ""
        if "does not contain any stream" in stderr or "Stream map" in stderr:
            log.warning("audio: source has no audio track")
            return None
        raise AudioError(f"ffmpeg failed: {stderr.strip()[:500]}")

    if not audio_path.exists() or audio_path.stat().st_size == 0:
        log.warning("audio: ffmpeg succeeded but output is empty (no audio track?)")
        return None

    try:
        with wave.open(str(audio_path), "rb") as w:
            frames = w.getnframes()
            rate = w.getframerate()
            channels = w.getnchannels()
            duration = frames / float(rate) if rate else 0.0
    except wave.Error as exc:
        raise AudioError(f"failed to read wav: {exc}") from exc

    return AudioInfo(path=audio_path, duration_seconds=duration,
                     sample_rate=rate, channels=channels)


def segment_audio(
    audio: AudioInfo,
    video_id: str,
    segments_dir: Path,
    segment_seconds: float,
    conn: sqlite3.Connection,
    ffmpeg_bin: str = "ffmpeg",
) -> list[AudioSegment]:
    """Slice `audio` into fixed-length chunks, write a gapper_report per segment."""
    segments_dir.mkdir(parents=True, exist_ok=True)
    segments: list[AudioSegment] = []
    reports: list[GapperReport] = []
    timestamp = 0.0
    segment_number = 0

    while timestamp < audio.duration_seconds:
        seg_path = segments_dir / f"audio_segment_{int(timestamp):06d}.wav"
        cmd = [
            ffmpeg_bin, "-i", str(audio.path),
            "-ss", f"{timestamp:.3f}", "-t", f"{segment_seconds:.3f}",
            "-acodec", "copy", "-y", str(seg_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            log.warning("audio: segment %d failed (skip): %s",
                        segment_number, (proc.stderr or "").strip()[:200])
            timestamp += segment_seconds
            segment_number += 1
            continue

        segments.append(AudioSegment(
            segment_number=segment_number,
            timestamp_seconds=timestamp,
            duration_seconds=segment_seconds,
            path=seg_path,
        ))
        rms, dbfs = _segment_loudness(seg_path)
        audio_importance = _loudness_to_importance(dbfs)
        reports.append(GapperReport(
            video_id=video_id,
            gapper_type="audio",
            timestamp_ms=int(timestamp * 1000),
            gapper_id=f"audio_gapper_{segment_number}",
            start_frame=0,  # frame-correlation happens in Sprint 2
            end_frame=0,
            summary=f"Audio segment at {timestamp:.2f}s ({dbfs:.0f} dBFS)",
            importance=audio_importance,
            features={
                "segment_duration": segment_seconds,
                "has_audio": True,
                "timestamp": timestamp,
                "rms": round(rms, 6),
                "dbfs": round(dbfs, 2),
            },
        ))
        timestamp += segment_seconds
        segment_number += 1

    with transaction(conn):
        for r in reports:
            insert_gapper_report(conn, r)

    log.info("audio: produced %d segments", len(segments))
    return segments
