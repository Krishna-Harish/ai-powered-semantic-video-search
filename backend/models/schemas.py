"""
schemas.py

Pydantic request/response models for the FastAPI backend.
Keeping these separate from the API routes keeps main.py focused
purely on routing/wiring.
"""

from pydantic import BaseModel, Field, StrictInt
from typing import Dict, List, Literal, Optional


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


class MultimodalSearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = Field(10, ge=1, le=50)
    # Relative weight of transcript vs visual evidence (default: config values)
    text_weight: Optional[float] = Field(None, ge=0, le=1)
    visual_weight: Optional[float] = Field(None, ge=0, le=1)
    method: Optional[Literal["auto", "calibrated", "rrf"]] = None


class MultimodalSearchResult(BaseModel):
    video_id: str
    timestamp: float           # where to start playing
    start: float
    end: float
    score: float               # fused relevance, 0..1
    text_score: Optional[float] = None    # raw MiniLM cosine (debugging)
    visual_score: Optional[float] = None  # raw CLIP cosine of the best frame (debugging)
    text_norm: Optional[float] = None     # calibrated 0..1
    visual_norm: Optional[float] = None
    matched_by: List[Literal["text", "visual"]]
    text: Optional[str] = None            # transcript excerpt, if the moment has one
    frame_filename: Optional[str] = None  # representative frame, if any
    frame_timestamp: Optional[float] = None


class MultimodalSearchResponse(BaseModel):
    query: str
    method: Literal["calibrated", "rrf"]
    text_weight: float
    visual_weight: float
    modalities: List[str]
    warnings: List[str]
    results: List[MultimodalSearchResult]


class VideoInfo(BaseModel):
    video_id: str
    filename: str
    duration_seconds: Optional[float] = None
    title: Optional[str] = None        # set for videos ingested from YouTube
    source_url: Optional[str] = None
    text_indexed: bool = False         # has a transcript + text embeddings
    visual_indexed: bool = False       # has CLIP frame embeddings
    index_problem: Optional[str] = None  # why the last indexing run failed, if it did


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


# ---------------------------------------------------------------------------
# Quiz generator
# ---------------------------------------------------------------------------

class QuizGenerateRequest(BaseModel):
    video_id: str
    num_questions: Literal[5, 10, 15] = 5
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    question_type: Literal["mcq", "true_false", "mixed"] = "mixed"


class QuizQuestionPublic(BaseModel):
    """A question as shown before submission: no answer, explanation or timestamp."""
    id: str
    type: Literal["mcq", "true_false"]
    question: str
    options: List[str]
    topic: str


class QuizPublic(BaseModel):
    quiz_id: str
    video_id: str
    difficulty: str
    question_type: str
    requested_count: int
    notice: Optional[str] = None  # e.g. why there are fewer questions than requested
    model: Optional[str] = None
    questions: List[QuizQuestionPublic]


class QuizSubmitRequest(BaseModel):
    # question id -> selected option index (None/missing = unanswered)
    answers: Dict[str, Optional[StrictInt]] = {}


class QuizQuestionResult(BaseModel):
    id: str
    type: str
    question: str
    options: List[str]
    topic: str
    selected_index: Optional[int] = None
    correct_index: int
    is_correct: bool
    explanation: str
    start: float  # where the concept is explained ("Rewatch this section")
    end: float


class QuizRevisionSection(BaseModel):
    start: float
    end: float


class QuizTopicResult(BaseModel):
    topic: str
    correct: int
    total: int
    percentage: int
    needs_revision: bool
    revise_sections: List[QuizRevisionSection]


class QuizResultResponse(BaseModel):
    quiz_id: str
    video_id: str
    score: int
    total: int
    percentage: int
    answered: int
    results: List[QuizQuestionResult]
    topics: List[QuizTopicResult]


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
