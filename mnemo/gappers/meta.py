"""Meta gapper: a single root node spanning the entire video."""
from __future__ import annotations

from mnemo.gappers.base import GapperLevel, GapperNode


def build_meta_node(
    chapters: list[GapperNode], video_id: str, video_duration: float,
) -> GapperNode | None:
    """Single root summarising the whole video. None if there are no chapters."""
    if not chapters:
        return None
    avg_imp = sum(c.importance for c in chapters) / len(chapters)
    all_tags = sorted({t for c in chapters for t in c.narrative_tags})
    peak_chapter = max(chapters, key=lambda c: c.importance)

    summary = (
        f"Video: {video_duration:.1f}s, {len(chapters)} chapter(s), "
        f"avg importance {avg_imp:.2f}"
    )
    if all_tags:
        summary += f", themes: {','.join(all_tags[:5])}"

    return GapperNode(
        level=GapperLevel.META,
        node_id=f"{video_id}_meta",
        video_id=video_id,
        start_time=0.0,
        end_time=video_duration,
        importance=avg_imp,
        summary=summary,
        narrative_tags=all_tags,
        metadata={
            "chapter_count": len(chapters),
            "peak_chapter_id": peak_chapter.node_id,
            "duration_seconds": video_duration,
        },
    )
