from types import SimpleNamespace

import cv2
import numpy as np

from mnemo.db import init_for_settings, enqueue_video
from mnemo.pipeline.motion import (
    MotionExtractor, calculate_motion_features, detect_actions,
    find_motion_segments, extract_motion, MotionFrame,
)


def _fake_landmark(x, y, z=0.0, vis=1.0):
    return SimpleNamespace(x=x, y=y, z=z, visibility=vis)


def _fake_holistic_result(has_pose=True):
    pose_lm = None
    if has_pose:
        # 33 landmarks; we only need a handful indexed correctly
        lms = [_fake_landmark(0.5, 0.5) for _ in range(33)]
        # Make left_wrist (15) higher (smaller y) than left_shoulder (11)
        lms[15] = _fake_landmark(0.5, 0.2)
        lms[11] = _fake_landmark(0.5, 0.4)
        pose_lm = SimpleNamespace(landmark=lms)
    return SimpleNamespace(
        pose_landmarks=pose_lm,
        face_landmarks=None,
        left_hand_landmarks=None,
        right_hand_landmarks=None,
    )


class FakeBackend:
    def __init__(self, result=None):
        self._result = result or _fake_holistic_result()
        self.calls = 0
    def process(self, rgb):
        self.calls += 1
        return self._result
    def close(self):
        pass


def test_extract_holistic_returns_pose_dict():
    ext = MotionExtractor(_backend=FakeBackend())
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    result = ext.extract_holistic(frame)
    assert result is not None
    assert "pose" in result
    assert "left_wrist" in result["pose"]


def test_extract_holistic_returns_none_when_nothing_detected():
    ext = MotionExtractor(_backend=FakeBackend(_fake_holistic_result(has_pose=False)))
    result = ext.extract_holistic(np.zeros((100, 100, 3), dtype=np.uint8))
    assert result is None


def test_calculate_motion_features_zero_movement():
    pose = {"pose": {"left_wrist": {"x": 0.5, "y": 0.5, "z": 0.0}}}
    f = calculate_motion_features(pose, pose)
    assert f["motion_detected"] is True
    assert f["total_movement"] == 0.0


def test_calculate_motion_features_with_movement():
    prev = {"pose": {"left_wrist": {"x": 0.5, "y": 0.5, "z": 0.0}}}
    curr = {"pose": {"left_wrist": {"x": 0.6, "y": 0.6, "z": 0.0}}}
    f = calculate_motion_features(curr, prev)
    assert f["total_movement"] > 0


def test_detect_actions_arm_raised():
    pose = {"pose": {"left_wrist": {"y": 0.2}, "left_shoulder": {"y": 0.5}}}
    actions = detect_actions(pose, {})
    assert "left_arm_raised" in actions


def test_find_motion_segments_groups_continuous_motion():
    seq = [
        MotionFrame(0, None, {"total_movement": 0.001}),
        MotionFrame(1, None, {"total_movement": 0.05}),
        MotionFrame(2, None, {"total_movement": 0.05}),
        MotionFrame(3, None, {"total_movement": 0.05}),
        MotionFrame(4, None, {"total_movement": 0.05}),
        MotionFrame(5, None, {"total_movement": 0.001}),
    ]
    segs = find_motion_segments(seq, min_movement=0.02, min_frames=3)
    assert len(segs) == 1
    assert segs[0].start_frame == 1 and segs[0].end_frame == 4
    assert segs[0].frame_count == 4


def test_extract_motion_writes_reports(settings, tmp_path):
    """End-to-end: synthetic frames + fake backend -> reports in DB."""
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    # Write 5 stub jpgs to the frames dir
    for i in range(0, 5):
        img = np.full((240, 320, 3), i * 40, dtype=np.uint8)
        cv2.imwrite(str(frames_dir / f"frame_{i:06d}.jpg"), img)

    conn = init_for_settings(settings)
    video_id = enqueue_video(conn, "test://synthetic")

    ext = MotionExtractor(_backend=FakeBackend())
    count = extract_motion(video_id, frames_dir, fps=30.0, conn=conn, extractor=ext)
    assert count == 5

    rows = conn.execute(
        "SELECT COUNT(*) AS n FROM gapper_reports WHERE video_id=? AND gapper_type='motion'",
        (video_id,),
    ).fetchone()
    assert rows["n"] == 5
