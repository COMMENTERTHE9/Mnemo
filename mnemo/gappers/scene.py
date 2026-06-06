"""Scene gapper: group every 6 segments (30s) into a scene."""
from __future__ import annotations

from mnemo.gappers.base import (
    GapperLevel, GapperNode, SEGMENT_SECONDS, SCENE_SECONDS,
)

SEGMENTS_PER_SCENE = int(SCENE_SECONDS / SEGMENT_SECONDS)  # 6


def build_scene_nodes(
    segments: list[GapperNode], video_id: str,
) -> list[GapperNode]:
    """Group consecutive segments into scenes of up to SEGMENTS_PER_SCENE each."""
    if not segments:
        return []

    nodes: list[GapperNode] = []
    for i in range(0, len(segments), SEGMENTS_PER_SCENE):
        group = segments[i:i + SEGMENTS_PER_SCENE]
        start = group[0].start_time
        end = group[-1].end_time
        avg_imp = sum(s.importance for s in group) / len(group)
        peak_seg = max(group, key=lambda s: s.importance)
        all_tags = sorted({t for s in group for t in s.narrative_tags})

        summary = (
            f"{start:.0f}-{end:.0f}s scene: {len(group)} segments, "
            f"peak {peak_seg.importance:.2f} at {peak_seg.start_time:.0f}s"
        )
        if all_tags:
            summary += f", actions: {','.join(all_tags[:3])}"

        nodes.append(GapperNode(
            level=GapperLevel.SCENE,
            node_id=f"{video_id}_scene_{int(start):04d}",
            video_id=video_id,
            start_time=start,
            end_time=end,
            importance=avg_imp,
            summary=summary,
            narrative_tags=all_tags,
            metadata={
                "segment_count": len(group),
                "peak_segment_id": peak_seg.node_id,
                "peak_importance": peak_seg.importance,
            },
        ))
    return nodes
