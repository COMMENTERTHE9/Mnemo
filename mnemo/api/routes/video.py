from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, Query, Request
from mnemo.db import enqueue_video, get_video_status
from mnemo.models import (
    ProcessVideoRequest, ProcessVideoResponse, VideoStatusResponse,
)

router = APIRouter()

@router.post("/process", response_model=ProcessVideoResponse)
async def process_video(req: ProcessVideoRequest, request: Request) -> ProcessVideoResponse:
    if not req.video_url:
        raise HTTPException(status_code=400, detail="video_url is required")
    conn = request.app.state.db
    video_id = enqueue_video(conn, req.video_url)
    return ProcessVideoResponse(
        video_id=video_id,
        status="queued",
        message="Video queued for processing",
        created_at=datetime.now(timezone.utc).isoformat(),
    )

@router.get("/status", response_model=VideoStatusResponse)
async def video_status(
    request: Request,
    video_id: str = Query(..., description="Video ID to look up"),
) -> VideoStatusResponse:
    conn = request.app.state.db
    status = get_video_status(conn, video_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Video not found")
    return VideoStatusResponse(video_id=video_id, **status)
