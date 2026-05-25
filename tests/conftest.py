"""Shared pytest fixtures."""
from pathlib import Path
import cv2
import numpy as np
import pytest

from mnemo.config import Settings


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        db_path=tmp_path / "test.db",
        work_dir=tmp_path / "work",
        cookies_path=tmp_path / "cookies.txt",
    )


@pytest.fixture
def synthetic_video(tmp_path: Path) -> Path:
    """Generate a 3-second, 30fps, 320x240 test video using OpenCV only.
    Each frame is a solid color that drifts so motion detection has signal.
    No ffmpeg dependency — pure cv2.VideoWriter with mp4v fourcc.
    """
    out = tmp_path / "synthetic.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out), fourcc, 30.0, (320, 240))
    assert writer.isOpened(), "VideoWriter failed to open — check mp4v codec"
    for i in range(90):  # 3 seconds @ 30fps
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        frame[:, :] = (i * 2 % 256, 100, 200)
        cv2.rectangle(frame, (i * 3 % 280, 50), (i * 3 % 280 + 40, 90),
                      (255, 255, 255), -1)
        writer.write(frame)
    writer.release()
    assert out.exists() and out.stat().st_size > 1000
    return out
