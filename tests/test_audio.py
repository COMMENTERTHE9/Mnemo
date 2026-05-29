import os
from pathlib import Path
from unittest.mock import patch, MagicMock
import wave
import numpy as np
import pytest

from mnemo.db import init_for_settings, enqueue_video
from mnemo.pipeline.audio import (
    extract_audio, segment_audio, AudioError, AudioInfo,
    _segment_loudness, _loudness_to_importance, _resolve_ffmpeg,
)


def _write_silent_wav(path: Path, seconds: float = 3.0, rate: int = 16000):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(rate * seconds))


def _mock_ffmpeg_extract(audio_dir: Path):
    def runner(cmd, **kw):
        out = Path(cmd[-1])
        _write_silent_wav(out, seconds=3.0)
        return MagicMock(returncode=0, stdout="", stderr="")
    return runner


def _mock_ffmpeg_segment_success():
    def runner(cmd, **kw):
        out = Path(cmd[-1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"RIFF....WAVEfmt ")  # not a real wav, fine for this test
        return MagicMock(returncode=0, stdout="", stderr="")
    return runner


def test_extract_audio_reads_wav_metadata(tmp_path):
    with patch("subprocess.run", side_effect=_mock_ffmpeg_extract(tmp_path)):
        info = extract_audio(tmp_path / "fake.mp4", tmp_path / "audio")
    assert info is not None
    assert info.sample_rate == 16000
    assert info.channels == 1
    assert 2.5 < info.duration_seconds < 3.5


def test_extract_audio_no_track_returns_none(tmp_path):
    def runner(cmd, **kw):
        return MagicMock(returncode=1, stdout="",
                         stderr="Output file does not contain any stream")
    with patch("subprocess.run", side_effect=runner):
        info = extract_audio(tmp_path / "fake.mp4", tmp_path / "audio")
    assert info is None


def test_extract_audio_hard_failure_raises(tmp_path):
    def runner(cmd, **kw):
        return MagicMock(returncode=1, stdout="", stderr="Permission denied")
    with patch("subprocess.run", side_effect=runner):
        with pytest.raises(AudioError):
            extract_audio(tmp_path / "fake.mp4", tmp_path / "audio")


def test_extract_audio_missing_binary_raises(tmp_path):
    with patch("subprocess.run", side_effect=FileNotFoundError("ffmpeg")):
        with pytest.raises(AudioError):
            extract_audio(tmp_path / "fake.mp4", tmp_path / "audio",
                          ffmpeg_bin="ffmpeg-nope")


def test_segment_audio_writes_reports(settings, tmp_path):
    conn = init_for_settings(settings)
    video_id = enqueue_video(conn, "test://x")
    audio_path = tmp_path / "audio" / "full.wav"
    _write_silent_wav(audio_path, seconds=3.0)
    info = AudioInfo(path=audio_path, duration_seconds=3.0,
                     sample_rate=16000, channels=1)

    with patch("subprocess.run", side_effect=_mock_ffmpeg_segment_success()):
        segs = segment_audio(info, video_id, tmp_path / "audio" / "segments",
                             segment_seconds=1.0, conn=conn)

    assert len(segs) == 3
    rows = conn.execute(
        "SELECT COUNT(*) AS n FROM gapper_reports WHERE video_id=? AND gapper_type='audio'",
        (video_id,),
    ).fetchone()
    assert rows["n"] == 3


def _write_wav(path, amplitude, seconds=1.0, rate=16000):
    n = int(rate * seconds)
    samples = np.full(n, amplitude, dtype=np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(samples.tobytes())


def test_loudness_to_importance_endpoints():
    assert _loudness_to_importance(-80.0) == 0.0
    assert _loudness_to_importance(-60.0) == 0.0
    assert _loudness_to_importance(0.0) == 1.0
    assert 0.4 < _loudness_to_importance(-30.0) < 0.6


def test_segment_loudness_orders_silence_quiet_loud(tmp_path):
    silent = tmp_path / "silent.wav"
    _write_wav(silent, 0)
    quiet = tmp_path / "quiet.wav"
    _write_wav(quiet, 1000)
    loud = tmp_path / "loud.wav"
    _write_wav(loud, 20000)
    _, db_s = _segment_loudness(silent)
    _, db_q = _segment_loudness(quiet)
    _, db_l = _segment_loudness(loud)
    assert db_s == -80.0
    assert db_l > db_q > db_s
    assert _loudness_to_importance(db_l) > _loudness_to_importance(db_q)


def test_segment_loudness_missing_file_is_safe(tmp_path):
    rms, dbfs = _segment_loudness(tmp_path / "nope.wav")
    assert rms == 0.0 and dbfs == -80.0


def test_resolve_ffmpeg_returns_usable_executable():
    # imageio-ffmpeg is a project dependency, so even without system ffmpeg
    # the resolver must hand back a real, executable binary path.
    path = _resolve_ffmpeg()
    assert os.path.exists(path), f"ffmpeg path does not exist: {path}"
    assert os.access(path, os.X_OK), f"ffmpeg not executable: {path}"
