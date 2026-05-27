"""Memory tree builder: bottom-up assembly + top-down parent linking.

Pulls unified frames from gapper_reports, runs the four aggregation
levels (segment → scene → chapter → meta), wires parent_id pointers
by time-range containment, and writes everything to memory_nodes.
"""
from __future__ import annotations
import logging
import sqlite3

from mnemo.db import insert_memory_node, transaction
from mnemo.gappers.base import GapperNode
from mnemo.gappers.chapter import build_chapter_nodes
from mnemo.gappers.frame import load_unified_frames
from mnemo.gappers.meta import build_meta_node
from mnemo.gappers.scene import build_scene_nodes
from mnemo.gappers.segment import build_segment_nodes
from mnemo.models import MemoryNode

log = logging.getLogger(__name__)


def _find_parent(child: GapperNode, candidates: list[GapperNode]) -> str | None:
    """Return the node_id of the candidate whose time range contains
    child.start_time. None if none match."""
    for c in candidates:
        if c.start_time <= child.start_time < c.end_time + 1e-6:
            return c.node_id
    return None


def _to_memory_node(node: GapperNode) -> MemoryNode:
    return MemoryNode(
        video_id=node.video_id,
        node_level=int(node.level),
        node_id=node.node_id,
        parent_id=node.parent_id,
        start_time=node.start_time,
        end_time=node.end_time,
        summary=node.summary,
        importance=node.importance,
        narrative_tags=node.narrative_tags,
        deleted_by_ai=None,
        compression_data=None,
    )


def build_memory_tree(
    conn: sqlite3.Connection, video_id: str, video_duration: float,
) -> int:
    """Build and persist the full memory tree for a video. Returns the
    total number of memory_nodes rows written."""
    frames = load_unified_frames(conn, video_id)
    if not frames or video_duration <= 0:
        log.info("memory: no frames or zero duration for %s, skipping", video_id)
        return 0

    segments = build_segment_nodes(frames, video_id, video_duration)
    scenes = build_scene_nodes(segments, video_id)
    chapters = build_chapter_nodes(scenes, video_id)
    meta = build_meta_node(chapters, video_id, video_duration)

    if meta is None:
        log.info("memory: no chapters built for %s, skipping", video_id)
        return 0

    # Wire parent_ids top-down
    for ch in chapters:
        ch.parent_id = meta.node_id
    for sc in scenes:
        sc.parent_id = _find_parent(sc, chapters)
    for seg in segments:
        seg.parent_id = _find_parent(seg, scenes)

    all_nodes = [meta] + chapters + scenes + segments

    with transaction(conn):
        for n in all_nodes:
            insert_memory_node(conn, _to_memory_node(n))

    log.info(
        "memory: wrote %d nodes for %s (1 meta + %d chapters + %d scenes + %d segments)",
        len(all_nodes), video_id, len(chapters), len(scenes), len(segments),
    )
    return len(all_nodes)
