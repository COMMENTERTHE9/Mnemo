"""Chapter gapper: group every 10 scenes (5 minutes) into a chapter."""
from __future__ import annotations

from mnemo.gappers.base import (
    GapperLevel, GapperNode, SCENE_SECONDS, CHAPTER_SECONDS,
)

SCENES_PER_CHAPTER = int(CHAPTER_SECONDS / SCENE_SECONDS)  # 10


def build_chapter_nodes(
    scenes: list[GapperNode], video_id: str,
) -> list[GapperNode]:
    if not scenes:
        return []
    nodes: list[GapperNode] = []
    for i in range(0, len(scenes), SCENES_PER_CHAPTER):
        group = scenes[i:i + SCENES_PER_CHAPTER]
        start = group[0].start_time
        end = group[-1].end_time
        avg_imp = sum(s.importance for s in group) / len(group)
        all_tags = sorted({t for s in group for t in s.narrative_tags})
        peak_scene = max(group, key=lambda s: s.importance)

        summary = (
            f"Chapter {start:.0f}-{end:.0f}s: {len(group)} scenes, "
            f"avg importance {avg_imp:.2f}"
        )
        if all_tags:
            summary += f", themes: {','.join(all_tags[:5])}"

        nodes.append(GapperNode(
            level=GapperLevel.CHAPTER,
            node_id=f"{video_id}_chapter_{int(start):04d}",
            video_id=video_id,
            start_time=start,
            end_time=end,
            importance=avg_imp,
            summary=summary,
            narrative_tags=all_tags,
            metadata={
                "scene_count": len(group),
                "peak_scene_id": peak_scene.node_id,
                "peak_importance": peak_scene.importance,
            },
        ))
    return nodes
