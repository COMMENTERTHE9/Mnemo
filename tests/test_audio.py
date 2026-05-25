from pathlib import Path
from unittest.mock import patch, MagicMock
import wave
import pytest

from mnemo.db import init_for_settings, enqueue_video
from mnemo.pipeline.audio import (
    extract_audio, segment_audio, AudioError, AudioInfo,
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
