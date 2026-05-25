from fastapi import APIRouter, Query, Request
from mnemo.db import top_memory_nodes
from mnemo.models import QueryMemoryRequest, QueryMemoryResponse, QueryResult

router = APIRouter()


def _build_response(rows, video_id: str) -> QueryMemoryResponse:
    results = [
        QueryResult(
            node_id=r["node_id"],
            relevance_score=float(r["importance"] or 0.0),
            timestamp=float(r["start_time"] or 0.0),
            summary=r["summary"] or "",
            context="",
            reconstructable=True,
        )
        for r in rows
    ]
    return QueryMemoryResponse(video_id=video_id, results=results)


@router.get("/{video_id}/query", response_model=QueryMemoryResponse)
async def query_memory_get(
    video_id: str, request: Request, q: str = Query(default=""),
) -> QueryMemoryResponse:
    # NOTE: q is currently ignored — matches Go behavior. Real query in Sprint 3.
    rows = top_memory_nodes(request.app.state.db, video_id)
    return _build_response(rows, video_id)


@router.post("/{video_id}/query", response_model=QueryMemoryResponse)
async def query_memory_post(
    video_id: str, req: QueryMemoryRequest, request: Request,
) -> QueryMemoryResponse:
    rows = top_memory_nodes(request.app.state.db, video_id)
    return _build_response(rows, video_id)
