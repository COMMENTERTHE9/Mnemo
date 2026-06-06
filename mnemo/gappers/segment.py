"""Segment gapper: aggregate UnifiedFrame records into 5-second windows."""
from __future__ import annotations

from mnemo.gappers.base import (
    GapperLevel, GapperNode, UnifiedFrame, SEGMENT_SECONDS,
)


def build_segment_nodes(
    frames: list[UnifiedFrame], video_id: str, video_duration: float,
) -> list[GapperNode]:
    """Partition frames into 5-second windows. One GapperNode per window
    that contains at least one frame."""
    if not frames or video_duration <= 0:
        return []

    nodes: list[GapperNode] = []
    window_start = 0.0
    while window_start < video_duration:
        window_end = min(window_start + SEGMENT_SECONDS, video_duration)
        in_window = [f for f in frames
                     if window_start <= f.timestamp_seconds < window_end]
        if not in_window:
            window_start = window_end
            continue

        peak = max(f.importance for f in in_window)
        avg_motion = sum(f.motion_magnitude for f in in_window) / len(in_window)
        all_actions = sorted({a for f in in_window for a in f.actions})
        has_audio = any(f.has_audio for f in in_window)

        parts = []
        if all_actions:
            parts.append(f"actions: {','.join(all_actions[:3])}")
        if avg_motion > 0.05:
            parts.append(f"motion {avg_motion:.2f}")
        if has_audio:
            parts.append("audio")
        summary = (
            f"{window_start:.0f}-{window_end:.0f}s: "
            + (" | ".join(parts) if parts else "static")
        )

        nodes.append(GapperNode(
            level=GapperLevel.SEGMENT,
            node_id=f"{video_id}_segment_{int(window_start):04d}",
            video_id=video_id,
            start_time=window_start,
            end_time=window_end,
            importance=peak,
            summary=summary,
            narrative_tags=all_actions,
            metadata={
                "frame_count": len(in_window),
                "peak_importance": peak,
                "avg_motion": avg_motion,
                "has_audio": has_audio,
            },
        ))
        window_start = window_end
    return nodes
