"""
main.py  (Milestone 8)

FastAPI backend exposing:
    GET  /videos          -- list processed videos
    POST /search/text     -- text/semantic search
    POST /search/visual    -- visual search
    POST /search/multimodal -- transcript + visual evidence fused into one ranking
    GET  /video/{video_id} -- stream a video file (for the <video> player)
    GET  /frame/{video_id}/{filename} -- serve a sampled frame image (for previews)
    POST /ingest/youtube   -- download + index a YouTube video (background job)
    GET  /ingest/jobs      -- list ingestion jobs
    GET  /ingest/jobs/{job_id} -- poll one ingestion job
    POST /ingest/reindex/{video_id} -- (re)index an already-downloaded video
    POST /summarize        -- LLM summary / key takeaways / section summary
    POST /ask              -- LLM answer grounded in transcript passages
                              (Gemini free tier or local Ollama, see services/llm.py)
    POST /quiz/generate    -- AI quiz from a video's transcript (answers stay on the server)
    GET  /quiz/{quiz_id}   -- a generated quiz, without answers
    POST /quiz/{quiz_id}/submit -- score answers; returns answers, explanations, timestamps
    GET  /app/             -- the frontend (same origin as the API)

Run with:  uvicorn backend.main:app --reload
"""

import os
import re
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.config import BASE_DIR, VIDEOS_DIR, FRAMES_DIR, SUPPORTED_VIDEO_EXTENSIONS
from backend.models.schemas import (
    TextSearchRequest,
    VisualSearchRequest,
    TextSearchResponse,
    VisualSearchResponse,
    MultimodalSearchRequest,
    MultimodalSearchResponse,
    VideoListResponse,
    VideoInfo,
    YouTubeIngestRequest,
    IngestJob,
    IngestJobListResponse,
    SummaryRequest,
    SummaryResponse,
    AskRequest,
    AskResponse,
    QuizGenerateRequest,
    QuizPublic,
    QuizSubmitRequest,
    QuizResultResponse,
)
from backend.services.search import text_search, visual_search
from backend.services.fusion import multimodal_search, FusionUnavailableError
from backend.services.video_processor import get_video_duration, VideoProcessingError
from backend.services.youtube_ingest import job_manager, load_video_info
from backend.services.indexing_pipeline import get_index_state
from backend.services.quiz import generate_quiz, load_quiz, public_view, score_quiz, QuizError, QuizNotFoundError
from backend.services.text_index import TextIndexError
from backend.services.visual_index import VisualIndexError
from backend.services.summarizer import (
    summarize,
    answer_question,
    SummaryError,
    TranscriptNotFoundError,
    LLMUnavailableError,
)

app = FastAPI(title="AI-Powered Semantic Search in Video")

# Allow the frontend (served separately or opened as a local file) to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/videos", response_model=VideoListResponse)
def list_videos():
    """Lists every video currently in data/videos/."""
    if not os.path.isdir(VIDEOS_DIR):
        return VideoListResponse(videos=[])

    videos = []
    for filename in sorted(os.listdir(VIDEOS_DIR)):
        ext = os.path.splitext(filename)[1].lower()
        if ext not in SUPPORTED_VIDEO_EXTENSIONS:
            continue
        video_id = os.path.splitext(filename)[0]
        video_path = os.path.join(VIDEOS_DIR, filename)
        try:
            duration = get_video_duration(video_path)
        except VideoProcessingError:
            duration = None
        info = load_video_info(video_id)
        videos.append(VideoInfo(
            video_id=video_id,
            filename=filename,
            duration_seconds=duration,
            title=info.get("title"),
            source_url=info.get("source_url"),
            **get_index_state(video_id),
        ))

    return VideoListResponse(videos=videos)


@app.post("/search/text", response_model=TextSearchResponse)
def search_text_endpoint(request: TextSearchRequest):
    try:
        result = text_search(request.query, top_k=request.top_k)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except TextIndexError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        # Covers: text index not built yet, model not installed, etc.
        raise HTTPException(status_code=500, detail=f"Text search failed: {e}")


@app.post("/search/visual", response_model=VisualSearchResponse)
def search_visual_endpoint(request: VisualSearchRequest):
    try:
        result = visual_search(request.query, top_k=request.top_k)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except VisualIndexError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Visual search failed: {e}")


_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def _check_safe_name(value: str, what: str):
    """Rejects ids/filenames that could escape their data directory."""
    if not _SAFE_ID_RE.match(value) or value in (".", ".."):
        raise HTTPException(status_code=400, detail=f"Invalid {what}.")


@app.post("/search/multimodal", response_model=MultimodalSearchResponse)
def search_multimodal_endpoint(request: MultimodalSearchRequest):
    """Ranks moments by what is said AND what is shown (see services/fusion.py)."""
    try:
        return multimodal_search(request.query, top_k=request.top_k, text_weight=request.text_weight,
                                 visual_weight=request.visual_weight, method=request.method)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FusionUnavailableError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Multimodal search failed: {e}")


@app.get("/video/{video_id}")
def get_video(video_id: str):
    """
    Streams the raw video file so the frontend <video> player can play it.
    FileResponse answers HTTP Range requests (206 Partial Content), which the
    browser needs to seek straight to a search result's timestamp without
    downloading the whole file first.
    """
    _check_safe_name(video_id, "video id")
    for ext in SUPPORTED_VIDEO_EXTENSIONS:
        candidate = os.path.join(VIDEOS_DIR, f"{video_id}{ext}")
        if os.path.isfile(candidate):
            return FileResponse(candidate)
    raise HTTPException(status_code=404, detail=f"Video '{video_id}' not found.")


@app.get("/frame/{video_id}/{filename}")
def get_frame(video_id: str, filename: str):
    """Serves a single sampled frame image, used for visual-search result previews."""
    _check_safe_name(video_id, "video id")
    _check_safe_name(filename, "frame filename")
    frame_path = os.path.join(FRAMES_DIR, video_id, filename)
    if not os.path.isfile(frame_path):
        raise HTTPException(status_code=404, detail="Frame not found.")
    return FileResponse(frame_path)


@app.post("/ingest/youtube", response_model=IngestJob, status_code=202)
def ingest_youtube_endpoint(request: YouTubeIngestRequest):
    """
    Queues a YouTube URL for download (yt-dlp) and indexing (FFmpeg, Whisper,
    sentence-transformer, CLIP, FAISS). Poll GET /ingest/jobs/{job_id} for progress.
    """
    try:
        return job_manager.submit(request.url, force=request.force)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/ingest/reindex/{video_id}", response_model=IngestJob, status_code=202)
def reindex_endpoint(video_id: str, force: bool = False):
    """Runs transcription/embedding/indexing again for a video already in data/videos/."""
    _check_safe_name(video_id, "video id")
    try:
        return job_manager.submit_reindex(video_id, force=force)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/ingest/jobs", response_model=IngestJobListResponse)
def list_ingest_jobs():
    return {"jobs": job_manager.list()}


@app.get("/ingest/jobs/{job_id}", response_model=IngestJob)
def get_ingest_job(job_id: str):
    job = job_manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return job


def _llm_http_error(e: Exception) -> HTTPException:
    if isinstance(e, ValueError):
        return HTTPException(status_code=400, detail=str(e))
    if isinstance(e, (TranscriptNotFoundError, QuizNotFoundError)):
        return HTTPException(status_code=404, detail=str(e))
    if isinstance(e, QuizError):
        return HTTPException(status_code=422, detail=str(e))
    if isinstance(e, (TextIndexError, VisualIndexError)):
        return HTTPException(status_code=409, detail=str(e))
    if isinstance(e, LLMUnavailableError):
        return HTTPException(status_code=503, detail=str(e))
    if isinstance(e, SummaryError):
        return HTTPException(status_code=502, detail=str(e))
    return HTTPException(status_code=500, detail=f"Request failed: {e}")


@app.post("/summarize", response_model=SummaryResponse)
def summarize_endpoint(request: SummaryRequest):
    _check_safe_name(request.video_id, "video id")
    try:
        return summarize(request.video_id, request.mode, request.start, request.end, force=request.force)
    except Exception as e:
        raise _llm_http_error(e)


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(request: AskRequest):
    if request.video_id:
        _check_safe_name(request.video_id, "video id")
    try:
        return answer_question(request.question, video_id=request.video_id, top_k=request.top_k)
    except Exception as e:
        raise _llm_http_error(e)


@app.post("/quiz/generate", response_model=QuizPublic)
def quiz_generate_endpoint(request: QuizGenerateRequest):
    """Generates a quiz from the video's transcript. Answers are NOT included."""
    _check_safe_name(request.video_id, "video id")
    try:
        return generate_quiz(request.video_id, request.num_questions, request.difficulty, request.question_type)
    except Exception as e:
        raise _llm_http_error(e)


@app.get("/quiz/{quiz_id}", response_model=QuizPublic)
def quiz_get_endpoint(quiz_id: str):
    try:
        return public_view(load_quiz(quiz_id))
    except Exception as e:
        raise _llm_http_error(e)


@app.post("/quiz/{quiz_id}/submit", response_model=QuizResultResponse)
def quiz_submit_endpoint(quiz_id: str, request: QuizSubmitRequest):
    """Scores the answers on the server and reveals answers + explanations."""
    try:
        return score_quiz(quiz_id, request.answers)
    except Exception as e:
        raise _llm_http_error(e)


# Serve the frontend from the API itself, so it works without a separate
# static server and without cross-origin issues: http://127.0.0.1:8000/app/
app.mount("/app", StaticFiles(directory=os.path.join(BASE_DIR, "frontend"), html=True), name="frontend")


@app.get("/")
def root():
    return {"message": "AI Video Search API is running. Open /app/ for the UI or /docs for API documentation."}
