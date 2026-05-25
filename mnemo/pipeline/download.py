"""Video download via yt-dlp.

Replaces the download portion of the legacy worker/video_processor.py:67-99.
Differences from the original:
  - Configurable cookies path (was hardcoded /data/youtube_cookies.txt)
  - Configurable output path (was self.work_dir / f"{video_id}.mp4")
  - Raises a typed DownloadError so callers can distinguish failure modes
  - No global yt-dlp install assumption — caller provides the binary name
"""
from __future__ import annotations
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


class DownloadError(RuntimeError):
    """Raised when yt-dlp fails to retrieve the video."""


class AuthRequiredError(DownloadError):
    """Raised when the source requires authentication (e.g. age-gated YouTube)."""


@dataclass(frozen=True)
class DownloadResult:
    video_path: Path
    source_url: str


def download(
    url: str,
    out_path: Path,
    cookies_path: Path | None = None,
    ytdlp_bin: str = "yt-dlp",
) -> DownloadResult:
    """Download `url` to `out_path` using yt-dlp.

    Args:
        url: Source URL (YouTube or any yt-dlp-supported site).
        out_path: Destination file path. Parent must exist.
        cookies_path: Optional Netscape-format cookies file. If provided
            and the file exists, passed via --cookies.
        ytdlp_bin: yt-dlp binary name/path (override for tests).

    Returns:
        DownloadResult with the actual on-disk path.

    Raises:
        AuthRequiredError: If yt-dlp reports the source needs sign-in.
        DownloadError: For any other yt-dlp failure.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        ytdlp_bin,
        "-f", "best[ext=mp4]/best",
        "-o", str(out_path),
        "--no-playlist",
    ]
    if cookies_path is not None and cookies_path.exists():
        log.info("download: using cookies from %s", cookies_path)
        cmd.extend(["--cookies", str(cookies_path)])
    cmd.append(url)

    log.info("download: starting %s", url)
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr or ""
        if "Sign in to confirm" in stderr or "Sign in" in stderr:
            raise AuthRequiredError(
                f"yt-dlp requires authentication for {url}. "
                f"Provide a cookies file."
            ) from exc
        raise DownloadError(f"yt-dlp failed: {stderr.strip()}") from exc
    except FileNotFoundError as exc:
        raise DownloadError(f"yt-dlp binary not found: {ytdlp_bin}") from exc

    if not out_path.exists():
        raise DownloadError(f"yt-dlp reported success but {out_path} is missing")
    log.info("download: complete %s (%d bytes)", out_path, out_path.stat().st_size)
    return DownloadResult(video_path=out_path, source_url=url)
