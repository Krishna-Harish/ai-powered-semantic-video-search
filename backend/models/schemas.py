"""
schemas.py

Pydantic request/response models for the FastAPI backend.
Keeping these separate from the API routes keeps main.py focused
purely on routing/wiring.
"""

from pydantic import BaseModel
from typing import List, Optional


class TextSearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5


class VisualSearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5


class TextSearchResult(BaseModel):
    video_id: str
    start: float
    end: float
    text: str
    score: float


class VisualSearchResult(BaseModel):
    video_id: str
    timestamp: float
    frame_path: str
    score: float


class TextSearchResponse(BaseModel):
    query: str
    results: List[TextSearchResult]


class VisualSearchResponse(BaseModel):
    query: str
    results: List[VisualSearchResult]


class VideoInfo(BaseModel):
    video_id: str
    filename: str
    duration_seconds: Optional[float] = None


class VideoListResponse(BaseModel):
    videos: List[VideoInfo]


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
