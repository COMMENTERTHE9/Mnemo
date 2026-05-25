from pathlib import Path
from unittest.mock import patch, MagicMock
import subprocess
import pytest

from mnemo.pipeline.download import (
    download, DownloadError, AuthRequiredError, DownloadResult,
)


def _fake_subprocess_run_success(out_path: Path):
    """Patch target: simulate yt-dlp writing the file and exiting 0."""
    def _runner(cmd, **kwargs):
        # Last arg is the URL; -o flag's value is the output path
        idx = cmd.index("-o")
        Path(cmd[idx + 1]).write_bytes(b"\x00\x00\x00\x18ftypmp42fake")
        return MagicMock(returncode=0, stdout="", stderr="")
    return _runner


def test_download_success(tmp_path):
    out = tmp_path / "vid.mp4"
    with patch("subprocess.run", side_effect=_fake_subprocess_run_success(out)):
        result = download("https://example.com/x", out)
    assert isinstance(result, DownloadResult)
    assert result.video_path == out
    assert out.exists()


def test_download_includes_cookies_when_present(tmp_path):
    out = tmp_path / "vid.mp4"
    cookies = tmp_path / "cookies.txt"
    cookies.write_text("# Netscape\n")
    with patch("subprocess.run", side_effect=_fake_subprocess_run_success(out)) as m:
        download("https://example.com/x", out, cookies_path=cookies)
    cmd = m.call_args.args[0]
    assert "--cookies" in cmd
    assert str(cookies) in cmd


def test_download_skips_cookies_when_missing(tmp_path):
    out = tmp_path / "vid.mp4"
    cookies = tmp_path / "nope.txt"  # does not exist
    with patch("subprocess.run", side_effect=_fake_subprocess_run_success(out)) as m:
        download("https://example.com/x", out, cookies_path=cookies)
    cmd = m.call_args.args[0]
    assert "--cookies" not in cmd


def test_download_auth_required_raises_typed(tmp_path):
    out = tmp_path / "vid.mp4"
    def runner(cmd, **kw):
        raise subprocess.CalledProcessError(
            1, cmd, output="", stderr="ERROR: Sign in to confirm you're not a bot",
        )
    with patch("subprocess.run", side_effect=runner):
        with pytest.raises(AuthRequiredError):
            download("https://youtube.com/x", out)


def test_download_generic_failure_raises_DownloadError(tmp_path):
    out = tmp_path / "vid.mp4"
    def runner(cmd, **kw):
        raise subprocess.CalledProcessError(1, cmd, output="", stderr="ERROR: 404")
    with patch("subprocess.run", side_effect=runner):
        with pytest.raises(DownloadError) as exc_info:
            download("https://example.com/x", out)
    assert not isinstance(exc_info.value, AuthRequiredError)


def test_download_missing_binary_raises_DownloadError(tmp_path):
    out = tmp_path / "vid.mp4"
    with patch("subprocess.run", side_effect=FileNotFoundError("yt-dlp")):
        with pytest.raises(DownloadError):
            download("https://example.com/x", out, ytdlp_bin="yt-dlp-nope")
