
from mnemo.db import init_for_settings, enqueue_video, insert_gapper_report
from mnemo.gappers.frame import load_unified_frames
from mnemo.models import GapperReport


def _frame_report(vid, fn, ts_ms, imp, blur):
    return GapperReport(video_id=vid, gapper_type="frame", timestamp_ms=ts_ms,
                        gapper_id=f"f_{fn}", start_frame=fn, end_frame=fn,
                        summary="", importance=imp,
                        features={"blur_variance": blur})


def _motion_report(vid, fn, ts_ms, imp, total_movement, actions):
    return GapperReport(video_id=vid, gapper_type="motion", timestamp_ms=ts_ms,
                        gapper_id=f"m_{fn}", start_frame=fn, end_frame=fn,
                        summary="", importance=imp,
                        features={"motion_features": {
                            "total_movement": total_movement,
                            "action_hints": actions,
                        }})


def _audio_report(vid, fn, ts_ms, imp):
    return GapperReport(video_id=vid, gapper_type="audio", timestamp_ms=ts_ms,
                        gapper_id=f"a_{fn}", start_frame=0, end_frame=0,
                        summary="", importance=imp, features={})


def test_load_unified_frames_combines_three_sources(settings):
    conn = init_for_settings(settings)
    vid = enqueue_video(conn, "test://x")
    insert_gapper_report(conn, _frame_report(vid, 0, 0, 0.5, 100.0))
    insert_gapper_report(conn, _motion_report(vid, 0, 0, 0.8, 0.3, ["walking"]))
    insert_gapper_report(conn, _audio_report(vid, 0, 0, 0.6))

    frames = load_unified_frames(conn, vid)
    assert len(frames) == 1
    f = frames[0]
    assert f.frame_number == 0
    # 0.3*0.5 + 0.5*0.8 + 0.2*0.6 = 0.15 + 0.40 + 0.12 = 0.67
    assert abs(f.importance - 0.67) < 0.01
    assert f.has_audio is True
    assert "walking" in f.actions


def test_load_unified_frames_handles_missing_motion(settings):
    conn = init_for_settings(settings)
    vid = enqueue_video(conn, "test://x")
    insert_gapper_report(conn, _frame_report(vid, 0, 0, 0.5, 100.0))
    # No motion or audio
    frames = load_unified_frames(conn, vid)
    assert len(frames) == 1
    assert abs(frames[0].importance - 0.15) < 0.01  # only frame weight contributes
    assert frames[0].has_audio is False
    assert frames[0].actions == []


def test_load_unified_frames_sorted_by_frame_number(settings):
    conn = init_for_settings(settings)
    vid = enqueue_video(conn, "test://x")
    # Insert out of order
    insert_gapper_report(conn, _frame_report(vid, 60, 2000, 0.4, 80.0))
    insert_gapper_report(conn, _frame_report(vid, 0, 0, 0.5, 100.0))
    insert_gapper_report(conn, _frame_report(vid, 30, 1000, 0.6, 120.0))
    frames = load_unified_frames(conn, vid)
    assert [f.frame_number for f in frames] == [0, 30, 60]


def test_load_unified_frames_audio_window_match(settings):
    """Audio gappers are keyed by floor(seconds). Frame at t=1.5s should
    match audio segment at t=1.0s."""
    conn = init_for_settings(settings)
    vid = enqueue_video(conn, "test://x")
    insert_gapper_report(conn, _frame_report(vid, 45, 1500, 0.5, 100.0))
    insert_gapper_report(conn, _audio_report(vid, 0, 1000, 0.9))  # t=1s
    frames = load_unified_frames(conn, vid)
    assert frames[0].has_audio is True
