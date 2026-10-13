"""
schemas.py

Pydantic request/response models for the FastAPI backend.
Keeping these separate from the API routes keeps main.py focused
purely on routing/wiring.
"""

from pydantic import BaseModel, Field
from typing import List, Literal, Optional


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
    title: Optional[str] = None        # set for videos ingested from YouTube
    source_url: Optional[str] = None


class VideoListResponse(BaseModel):
    videos: List[VideoInfo]


# ---------------------------------------------------------------------------
# YouTube ingestion
# ---------------------------------------------------------------------------

class YouTubeIngestRequest(BaseModel):
    url: str
    force: bool = False  # re-run transcription/embedding even if outputs exist


class IngestJob(BaseModel):
    job_id: str
    url: str
    video_id: str
    status: Literal["queued", "running", "completed", "failed"]
    stage: str
    message: str
    title: Optional[str] = None
    error: Optional[str] = None
    warnings: List[str] = []
    created_at: float
    updated_at: float


class IngestJobListResponse(BaseModel):
    jobs: List[IngestJob]


# ---------------------------------------------------------------------------
# LLM summaries and grounded Q&A
# ---------------------------------------------------------------------------

class SummaryRequest(BaseModel):
    video_id: str
    mode: Literal["summary", "takeaways", "section"] = "summary"
    start: Optional[float] = Field(None, ge=0)  # seconds; required for mode="section"
    end: Optional[float] = Field(None, ge=0)
    force: bool = False  # ignore the cached whole-video summary


class SummaryPoint(BaseModel):
    text: str
    timestamp: Optional[float] = None  # seconds, snapped to a transcript segment


class SummaryResponse(BaseModel):
    video_id: str
    mode: str
    start: Optional[float] = None
    end: Optional[float] = None
    summary: str
    points: List[SummaryPoint]
    model: str
    cached: bool = False


class AskRequest(BaseModel):
    question: str
    video_id: Optional[str] = None  # restrict evidence to one video
    top_k: Optional[int] = Field(None, ge=1, le=20)


class Citation(BaseModel):
    ref: int  # the [n] marker used in the answer text
    video_id: str
    start: float
    end: float
    text: str


class AskResponse(BaseModel):
    question: str
    video_id: Optional[str] = None
    answerable: bool
    answer: str
    citations: List[Citation]
    model: str


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
