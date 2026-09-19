"""
main.py  (Milestone 8)

FastAPI backend exposing:
    GET  /videos          -- list processed videos
    POST /search/text     -- text/semantic search
    POST /search/visual    -- visual search
    GET  /video/{video_id} -- stream a video file (for the <video> player)
    GET  /frame/{video_id}/{filename} -- serve a sampled frame image (for previews)

Run with:  uvicorn backend.main:app --reload
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.config import VIDEOS_DIR, FRAMES_DIR, SUPPORTED_VIDEO_EXTENSIONS
from backend.models.schemas import (
    TextSearchRequest,
    VisualSearchRequest,
    TextSearchResponse,
    VisualSearchResponse,
    VideoListResponse,
    VideoInfo,
)
from backend.services.search import text_search, visual_search
from backend.services.video_processor import get_video_duration, VideoProcessingError

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
        videos.append(VideoInfo(video_id=video_id, filename=filename, duration_seconds=duration))

    return VideoListResponse(videos=videos)


@app.post("/search/text", response_model=TextSearchResponse)
def search_text_endpoint(request: TextSearchRequest):
    try:
        result = text_search(request.query, top_k=request.top_k)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Visual search failed: {e}")


@app.get("/video/{video_id}")
def get_video(video_id: str):
    """Streams the raw video file so the frontend <video> player can play it."""
    for ext in SUPPORTED_VIDEO_EXTENSIONS:
        candidate = os.path.join(VIDEOS_DIR, f"{video_id}{ext}")
        if os.path.isfile(candidate):
            return FileResponse(candidate)
    raise HTTPException(status_code=404, detail=f"Video '{video_id}' not found.")


@app.get("/frame/{video_id}/{filename}")
def get_frame(video_id: str, filename: str):
    """Serves a single sampled frame image, used for visual-search result previews."""
    frame_path = os.path.join(FRAMES_DIR, video_id, filename)
    if not os.path.isfile(frame_path):
        raise HTTPException(status_code=404, detail="Frame not found.")
    return FileResponse(frame_path)


@app.get("/")
def root():
    return {"message": "AI Video Search API is running. See /docs for API documentation."}
