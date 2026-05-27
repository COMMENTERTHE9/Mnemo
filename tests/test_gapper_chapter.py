from mnemo.gappers.base import GapperLevel, GapperNode
from mnemo.gappers.chapter import build_chapter_nodes
from mnemo.gappers.meta import build_meta_node


def _scene(start, end, importance, tags=()):
    return GapperNode(
        level=GapperLevel.SCENE, node_id=f"v1_scene_{int(start):04d}",
        video_id="v1", start_time=start, end_time=end,
        importance=importance, summary="", narrative_tags=list(tags),
    )


def test_chapter_empty():
    assert build_chapter_nodes([], "v1") == []


def test_chapter_short_video_one_chapter():
    # 1 scene → 1 chapter
    scenes = [_scene(0.0, 19.0, 0.5)]
    chapters = build_chapter_nodes(scenes, "v1")
    assert len(chapters) == 1
    assert chapters[0].level == GapperLevel.CHAPTER
    assert chapters[0].start_time == 0.0


def test_chapter_long_video_multiple_chapters():
    # 25 scenes → 3 chapters (10, 10, 5)
    scenes = [_scene(i * 30.0, (i + 1) * 30.0, 0.5) for i in range(25)]
    chapters = build_chapter_nodes(scenes, "v1")
    assert len(chapters) == 3
    assert chapters[0].metadata["scene_count"] == 10
    assert chapters[2].metadata["scene_count"] == 5


def test_meta_none_when_no_chapters():
    assert build_meta_node([], "v1", 0.0) is None


def test_meta_summarises_chapters():
    ch1 = GapperNode(level=GapperLevel.CHAPTER, node_id="c1",
                     video_id="v1", start_time=0.0, end_time=300.0,
                     importance=0.4, summary="", narrative_tags=["a"])
    ch2 = GapperNode(level=GapperLevel.CHAPTER, node_id="c2",
                     video_id="v1", start_time=300.0, end_time=600.0,
                     importance=0.8, summary="", narrative_tags=["b"])
    meta = build_meta_node([ch1, ch2], "v1", 600.0)
    assert meta is not None
    assert meta.level == GapperLevel.META
    assert meta.node_id == "v1_meta"
    assert meta.start_time == 0.0 and meta.end_time == 600.0
    assert abs(meta.importance - 0.6) < 0.01
    assert "a" in meta.narrative_tags and "b" in meta.narrative_tags
    assert meta.metadata["chapter_count"] == 2
