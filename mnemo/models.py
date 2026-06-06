"""Pydantic data model for Mnemo.

Derived from the original Cap'n Proto schemas (proto/{gapper,memory,communication}.capnp),
now deleted. These are the canonical shapes for API and storage.
"""
from __future__ import annotations
from enum import IntEnum, StrEnum
from typing import Any
from pydantic import BaseModel, Field


class GapperLevel(IntEnum):
    FRAME = 0
    SEGMENT = 1
    SCENE = 2
    CHAPTER = 3
    META = 4


class ProcessingLevel(StrEnum):
    QUICK = "quick"
    STANDARD = "standard"
    DEEP = "deep"


class VideoStatus(StrEnum):
    PENDING = "pending"
    DOWNLOADING = "downloading"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# ── API request/response shapes (preserve Go contract exactly) ─────────────

class ProcessVideoRequest(BaseModel):
    video_url: str
    processing_level: str = "standard"
    options: dict[str, Any] = Field(default_factory=dict)


class ProcessVideoResponse(BaseModel):
    video_id: str
    status: str
    message: str
    created_at: str  # RFC3339


class VideoStatusResponse(BaseModel):
    video_id: str
    status: str
    motion_status: str = "pending"
    gapper_status: str = "pending"


class QueryMemoryRequest(BaseModel):
    query: str = ""
    detail_level: int = 5
    start_time: float = 0.0
    end_time: float = 0.0


class QueryResult(BaseModel):
    node_id: str
    relevance_score: float
    timestamp: float
    summary: str
    context: str = ""
    reconstructable: bool = True


class QueryMemoryResponse(BaseModel):
    video_id: str
    results: list[QueryResult]


class UpdateCookiesRequest(BaseModel):
    cookies: str


class UpdateCookiesResponse(BaseModel):
    status: str
    message: str


# ── Internal domain models ────────────────────────────────────────────────

class GapperReport(BaseModel):
    video_id: str
    gapper_type: str
    timestamp_ms: int
    gapper_id: str
    start_frame: int
    end_frame: int
    summary: str = ""
    importance: float = 0.0
    features: dict[str, Any] = Field(default_factory=dict)


class MemoryNode(BaseModel):
    video_id: str
    node_level: int
    node_id: str
    parent_id: str | None = None
    start_time: float
    end_time: float
    summary: str = ""
    importance: float = 0.0
    narrative_tags: list[str] = Field(default_factory=list)
    deleted_by_ai: str | None = None
    compression_data: str | None = None
