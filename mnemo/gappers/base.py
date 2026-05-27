"""Base types and constants for the gapper hierarchy.

The hierarchy mirrors the original Cap'n Proto design:
  FRAME (0) -> SEGMENT (1) -> SCENE (2) -> CHAPTER (3) -> META (4)

Frame-level data stays as gapper_reports rows (one per source frame).
Segment and above are persisted as memory_nodes rows with parent_id.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any


class GapperLevel(IntEnum):
    FRAME = 0
    SEGMENT = 1
    SCENE = 2
    CHAPTER = 3
    META = 4


# Window sizes in seconds. Fixed for Sprint 2; smart boundary detection later.
SEGMENT_SECONDS = 5.0
SCENE_SECONDS = 30.0
CHAPTER_SECONDS = 300.0  # 5 minutes

# Weights for combining per-frame importance from three sources.
# Sum to 1.0. Tune in a later sprint when we have ground truth.
FRAME_WEIGHT = 0.3
MOTION_WEIGHT = 0.5
AUDIO_WEIGHT = 0.2


@dataclass(frozen=True)
class UnifiedFrame:
    """Per-frame view that merges frame/audio/motion gapper_reports."""
    frame_number: int
    timestamp_seconds: float
    importance: float                # Combined [0, 1]
    blur_variance: float
    motion_magnitude: float          # total_movement from motion features
    has_audio: bool
    actions: list[str] = field(default_factory=list)


@dataclass
class GapperNode:
    """A node in the memory hierarchy. Mutable: parent_id is assigned
    after construction during tree wiring."""
    level: GapperLevel
    node_id: str
    video_id: str
    start_time: float
    end_time: float
    importance: float
    summary: str
    parent_id: str | None = None
    narrative_tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
