"""Motion + pose extraction via MediaPipe Holistic.

Replaces motion-extractor/motion_extractor.py with the same heuristics
but driven by the orchestrator (no own queue, no sentinel rows).
"""
from __future__ import annotations
import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

from mnemo.db import insert_gapper_report, transaction
from mnemo.models import GapperReport

log = logging.getLogger(__name__)

_POSE_LANDMARK_NAMES = {
    0: "nose", 1: "left_eye_inner", 2: "left_eye", 3: "left_eye_outer",
    4: "right_eye_inner", 5: "right_eye", 6: "right_eye_outer",
    7: "left_ear", 8: "right_ear", 9: "mouth_left", 10: "mouth_right",
    11: "left_shoulder", 12: "right_shoulder", 13: "left_elbow",
    14: "right_elbow", 15: "left_wrist", 16: "right_wrist",
    17: "left_pinky", 18: "right_pinky", 19: "left_index",
    20: "right_index", 21: "left_thumb", 22: "right_thumb",
    23: "left_hip", 24: "right_hip", 25: "left_knee",
    26: "right_knee", 27: "left_ankle", 28: "right_ankle",
    29: "left_heel", 30: "right_heel", 31: "left_foot_index",
    32: "right_foot_index",
}


class HolisticBackend(Protocol):
    """Minimal interface a holistic backend must satisfy. Lets tests inject fakes."""
    def process(self, rgb_frame: np.ndarray) -> Any: ...
    def close(self) -> None: ...


def _build_default_holistic() -> HolisticBackend:
    import mediapipe as mp
    return mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=2,
        enable_segmentation=True,
        refine_face_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


@dataclass
class MotionExtractor:
    """Encapsulates the holistic backend so it can be reused across videos
    (MediaPipe construction is expensive) and swapped out in tests.
    """
    _backend: HolisticBackend | None = None

    def backend(self) -> HolisticBackend:
        if self._backend is None:
            self._backend = _build_default_holistic()
        return self._backend

    def close(self) -> None:
        if self._backend is not None:
            self._backend.close()
            self._backend = None

    def extract_holistic(self, frame_bgr: np.ndarray) -> dict[str, Any] | None:
        """Return a dict with 'pose'/'face'/'left_hand'/'right_hand' keys, or None
        if nothing was detected."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.backend().process(rgb)
        out: dict[str, Any] = {}

        pose_lm = getattr(results, "pose_landmarks", None)
        if pose_lm:
            out["pose"] = {
                _POSE_LANDMARK_NAMES.get(i, f"point_{i}"): {
                    "x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility,
                }
                for i, lm in enumerate(pose_lm.landmark)
            }

        face_lm = getattr(results, "face_landmarks", None)
        if face_lm:
            out["face"] = {"landmark_count": len(face_lm.landmark), "detected": True}

        left_lm = getattr(results, "left_hand_landmarks", None)
        if left_lm:
            out["left_hand"] = {
                f"point_{i}": {"x": lm.x, "y": lm.y, "z": lm.z}
                for i, lm in enumerate(left_lm.landmark)
            }

        right_lm = getattr(results, "right_hand_landmarks", None)
        if right_lm:
            out["right_hand"] = {
                f"point_{i}": {"x": lm.x, "y": lm.y, "z": lm.z}
                for i, lm in enumerate(right_lm.landmark)
            }

        return out or None


def calculate_motion_features(
    current: dict[str, Any] | None, previous: dict[str, Any] | None
) -> dict[str, Any]:
    if not current or not previous:
        return {"motion_detected": False}
    curr_pose = current.get("pose", {})
    prev_pose = previous.get("pose", {})
    velocities: dict[str, float] = {}
    total = 0.0
    count = 0
    for joint, c in curr_pose.items():
        p = prev_pose.get(joint)
        if not p:
            continue
        d = float(np.sqrt(
            (c["x"] - p["x"]) ** 2 + (c["y"] - p["y"]) ** 2 + (c["z"] - p["z"]) ** 2
        ))
        velocities[joint] = d
        total += d
        count += 1
    # Mean per-joint displacement, not the raw sum. The sum scaled with how
    # many landmarks were visible and pinned importance at the 0.9 cap; the
    # mean stays in a sane per-joint range that varies across segments.
    mean_movement = (total / count) if count else 0.0
    return {
        "motion_detected": True,
        "joint_velocities": velocities,
        "total_movement": mean_movement,   # now mean per-joint (was raw sum)
        "raw_total_movement": total,        # retained for debugging
        "action_hints": detect_actions(current, velocities),
    }


def detect_actions(pose: dict[str, Any], velocities: dict[str, float]) -> list[str]:
    actions: list[str] = []
    p = pose.get("pose", {})
    # Arm-raised stays body-relative positional (wrist above shoulder).
    if "left_wrist" in p and "left_shoulder" in p:
        if p["left_wrist"]["y"] < p["left_shoulder"]["y"]:
            actions.append("left_arm_raised")
    if "right_wrist" in p and "right_shoulder" in p:
        if p["right_wrist"]["y"] < p["right_shoulder"]["y"]:
            actions.append("right_arm_raised")
    # Velocity actions judged RELATIVE to the frame's own motion.
    # Uniform camera shake/pan moves every joint by a similar amount, so
    # no joint is an outlier and these stay quiet. A real action drives
    # specific joints well above the body's average displacement.
    moving = [v for v in velocities.values() if v > 0.0]
    if moving:
        mean_v = sum(moving) / len(moving)
        if mean_v > 0.02:  # absolute floor: ignore near-static jitter
            la = velocities.get("left_ankle", 0.0)
            ra = velocities.get("right_ankle", 0.0)
            lk = velocities.get("left_knee", 0.0)
            rk = velocities.get("right_knee", 0.0)
            if la > mean_v * 1.8 or ra > mean_v * 1.8:
                actions.append("possible_jump")
            if lk > mean_v * 1.4 and rk > mean_v * 1.4:
                actions.append("walking_or_running")
    return actions


@dataclass
class MotionFrame:
    frame_number: int
    pose: dict[str, Any] | None
    features: dict[str, Any]


@dataclass
class MotionSegment:
    start_frame: int
    end_frame: int
    actions: list[str] = field(default_factory=list)
    frame_count: int = 0


def find_motion_segments(sequence: list[MotionFrame],
                         min_movement: float = 0.02,
                         min_frames: int = 3) -> list[MotionSegment]:
    segments: list[MotionSegment] = []
    current: list[MotionFrame] | None = None
    for item in sequence:
        moving = (item.features.get("total_movement", 0) > min_movement)
        if moving:
            if current is None:
                current = [item]
            else:
                current.append(item)
        else:
            if current and len(current) > min_frames:
                all_actions = [a for f in current
                               for a in f.features.get("action_hints", [])]
                segments.append(MotionSegment(
                    start_frame=current[0].frame_number,
                    end_frame=current[-1].frame_number,
                    actions=sorted(set(all_actions)),
                    frame_count=len(current),
                ))
            current = None
    # Tail segment
    if current and len(current) > min_frames:
        all_actions = [a for f in current
                       for a in f.features.get("action_hints", [])]
        segments.append(MotionSegment(
            start_frame=current[0].frame_number,
            end_frame=current[-1].frame_number,
            actions=sorted(set(all_actions)),
            frame_count=len(current),
        ))
    return segments


def extract_motion(
    video_id: str,
    frames_dir: Path,
    fps: float,
    conn: sqlite3.Connection,
    extractor: MotionExtractor | None = None,
) -> int:
    """Run motion analysis over the JPGs in `frames_dir`.

    Writes one gapper_reports row of type 'motion' per frame, plus one row
    of type 'motion_segment' per detected motion segment. Returns the number
    of frames processed. Does NOT update video_metadata.motion_status —
    that's the orchestrator's responsibility.
    """
    if not frames_dir.exists():
        log.warning("motion: frames dir missing: %s", frames_dir)
        return 0

    ext = extractor or MotionExtractor()
    ms_per_frame = (1000.0 / fps) if fps > 0 else 33.33

    frame_files = sorted(frames_dir.glob("frame_*.jpg"))
    log.info("motion: processing %d frames at %.2f fps", len(frame_files), fps)

    motion_reports: list[GapperReport] = []
    sequence: list[MotionFrame] = []
    previous: dict[str, Any] | None = None

    for path in frame_files:
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        frame_number = int(path.stem.split("_")[1])

        pose = ext.extract_holistic(frame)
        features = calculate_motion_features(pose, previous) if previous else {
            "motion_detected": False,
        }

        summary = "No person detected"
        if pose:
            summary = "Person detected"
            hints = features.get("action_hints", [])
            if hints:
                summary = f"Action: {', '.join(hints)}"

        importance = 0.3
        mean_mv = features.get("total_movement", 0)  # now mean per-joint
        if mean_mv > 0.02:
            importance = min(0.9, 0.3 + mean_mv)

        motion_reports.append(GapperReport(
            video_id=video_id,
            gapper_type="motion",
            timestamp_ms=int(frame_number * ms_per_frame),
            gapper_id=f"motion_{frame_number}",
            start_frame=frame_number,
            end_frame=frame_number,
            summary=summary,
            importance=importance,
            features={
                "has_pose": pose is not None,
                "pose_data": pose,
                "motion_features": features,
            },
        ))
        sequence.append(MotionFrame(frame_number, pose, features))
        previous = pose

    segments = find_motion_segments(sequence)
    segment_reports: list[GapperReport] = []
    for seg in segments:
        if seg.actions:
            summary = f"Motion: {seg.actions[0]} ({seg.frame_count} frames)"
        else:
            summary = f"Motion segment: {seg.frame_count} frames"
        segment_reports.append(GapperReport(
            video_id=video_id,
            gapper_type="motion_segment",
            timestamp_ms=int(seg.start_frame * ms_per_frame),
            gapper_id=f"motion_seg_{seg.start_frame}",
            start_frame=seg.start_frame,
            end_frame=seg.end_frame,
            summary=summary,
            importance=0.8,
            features={"actions": seg.actions},
        ))

    with transaction(conn):
        for r in motion_reports:
            insert_gapper_report(conn, r)
        for r in segment_reports:
            insert_gapper_report(conn, r)

    log.info("motion: wrote %d frame reports, %d segment reports",
             len(motion_reports), len(segment_reports))
    return len(motion_reports)
