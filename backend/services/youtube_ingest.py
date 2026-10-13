"""
youtube_ingest.py

YouTube URL  -->  data/videos/yt_<youtube_id>.mp4  -->  fully indexed video

Lets a user paste a YouTube link instead of manually downloading and
preprocessing a video. The video is downloaded with yt-dlp, then handed to
indexing_pipeline.index_video() (FFmpeg -> Whisper -> embeddings -> FAISS).

Ingestion is slow (download + transcription can take minutes), so the API
runs it as a background job. Jobs are kept in memory and run one at a time:
the Whisper/CLIP models are heavy, and the FAISS indexes are rebuilt at the
end of each job.
"""

import os
import re
import json
import sys
import time
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse, parse_qs

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.config import (
    VIDEOS_DIR,
    METADATA_DIR,
    SUPPORTED_VIDEO_EXTENSIONS,
    YOUTUBE_VIDEO_ID_PREFIX,
    YOUTUBE_MAX_HEIGHT,
    YOUTUBE_MAX_DURATION_SECONDS,
)
from backend.services.indexing_pipeline import index_video, IndexingError


class YouTubeIngestError(Exception):
    pass


# ---------------------------------------------------------------------------
# URL handling
# ---------------------------------------------------------------------------

_YOUTUBE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
_YOUTUBE_HOSTS = {"youtube.com", "m.youtube.com", "music.youtube.com", "youtube-nocookie.com"}


def parse_youtube_id(url: str) -> str:
    """
    Extracts the 11-character video ID from any common YouTube URL form:
        https://www.youtube.com/watch?v=ID   (extra params like &t=, &list= ignored)
        https://youtu.be/ID
        https://www.youtube.com/shorts/ID | /embed/ID | /live/ID | /v/ID
    Raises ValueError for anything that isn't a single YouTube video URL.
    """
    if not url or not url.strip():
        raise ValueError("URL must not be empty.")

    url = url.strip()
    if "://" not in url:
        url = "https://" + url

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Unsupported URL scheme: '{parsed.scheme}'.")

    host = (parsed.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]

    candidate = None
    if host == "youtu.be":
        candidate = parsed.path.lstrip("/").split("/")[0]
    elif host in _YOUTUBE_HOSTS:
        path_parts = [p for p in parsed.path.split("/") if p]
        if parsed.path == "/watch" or parsed.path == "/watch/":
            candidate = parse_qs(parsed.query).get("v", [None])[0]
        elif len(path_parts) >= 2 and path_parts[0] in ("shorts", "embed", "live", "v"):
            candidate = path_parts[1]
    else:
        raise ValueError("Not a YouTube URL.")

    if not candidate or not _YOUTUBE_ID_RE.match(candidate):
        raise ValueError("Could not find a valid YouTube video ID in the URL.")
    return candidate


def video_id_for_youtube(youtube_id: str) -> str:
    """The local video_id used for a YouTube video (also its filename stem)."""
    return f"{YOUTUBE_VIDEO_ID_PREFIX}{youtube_id}"


def canonical_youtube_url(youtube_id: str) -> str:
    return f"https://www.youtube.com/watch?v={youtube_id}"


def find_local_video(video_id: str):
    """Returns the path of data/videos/<video_id>.<ext> if it exists, else None."""
    for ext in SUPPORTED_VIDEO_EXTENSIONS:
        candidate = os.path.join(VIDEOS_DIR, f"{video_id}{ext}")
        if os.path.isfile(candidate) and os.path.getsize(candidate) > 0:
            return candidate
    return None


def info_path(video_id: str) -> str:
    return os.path.join(METADATA_DIR, f"{video_id}_info.json")


def load_video_info(video_id: str) -> dict:
    """Returns saved source metadata (title, source_url, ...) or {} if none."""
    path = info_path(video_id)
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_youtube_video(url: str, progress=None) -> dict:
    """
    Downloads a YouTube video into data/videos/ as yt_<youtube_id>.mp4 and
    writes its metadata to data/metadata/yt_<youtube_id>_info.json.
    Idempotent: skips the download if the file already exists.

    Returns {"video_id", "video_path", "title", "source_url", ...}.
    """
    youtube_id = parse_youtube_id(url)
    video_id = video_id_for_youtube(youtube_id)
    source_url = canonical_youtube_url(youtube_id)

    existing = find_local_video(video_id)
    if existing:
        info = load_video_info(video_id) or {"video_id": video_id, "source_url": source_url}
        return {**info, "video_path": existing}

    try:
        import yt_dlp
    except ImportError:
        raise YouTubeIngestError("yt-dlp is not installed. Run: pip install yt-dlp")

    def hook(d):
        if progress is None or d.get("status") != "downloading":
            return
        total = d.get("total_bytes") or d.get("total_bytes_estimate")
        done = d.get("downloaded_bytes") or 0
        if total:
            progress("downloading", f"Downloading from YouTube... {100 * done / total:.0f}%")

    h = YOUTUBE_MAX_HEIGHT
    options = {
        # Prefer mp4/m4a streams (no re-encode needed), capped at max height.
        "format": (
            f"bv*[height<={h}][ext=mp4]+ba[ext=m4a]/b[height<={h}][ext=mp4]/"
            f"bv*[height<={h}]+ba/b[height<={h}]/b"
        ),
        "merge_output_format": "mp4",
        "outtmpl": os.path.join(VIDEOS_DIR, f"{video_id}.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "progress_hooks": [hook],
    }

    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            if progress is not None:
                progress("downloading", "Fetching video information...")
            meta = ydl.extract_info(source_url, download=False)

            if meta.get("is_live"):
                raise YouTubeIngestError("Live streams cannot be ingested. Try again after the stream ends.")
            duration = meta.get("duration")
            if YOUTUBE_MAX_DURATION_SECONDS and duration and duration > YOUTUBE_MAX_DURATION_SECONDS:
                raise YouTubeIngestError(
                    f"Video is {duration / 60:.0f} min long; the limit is "
                    f"{YOUTUBE_MAX_DURATION_SECONDS / 60:.0f} min (YOUTUBE_MAX_DURATION_SECONDS in config.py)."
                )

            ydl.process_ie_result(meta, download=True)
    except YouTubeIngestError:
        raise
    except Exception as e:  # yt_dlp.utils.DownloadError and friends
        raise YouTubeIngestError(f"Download failed: {e}")

    video_path = find_local_video(video_id)
    if video_path is None:
        raise YouTubeIngestError("Download finished but no video file was produced.")

    info = {
        "video_id": video_id,
        "youtube_id": youtube_id,
        "source_url": source_url,
        "title": meta.get("title"),
        "channel": meta.get("channel") or meta.get("uploader"),
        "duration_seconds": meta.get("duration"),
        "upload_date": meta.get("upload_date"),
    }
    with open(info_path(video_id), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    return {**info, "video_path": video_path}


def ingest_youtube_url(url: str, progress=None, force: bool = False) -> dict:
    """Download + index in one blocking call (used by the CLI script and job runner)."""
    downloaded = download_youtube_video(url, progress=progress)
    try:
        result = index_video(downloaded["video_path"], downloaded["video_id"], progress=progress, force=force)
    except IndexingError as e:
        raise YouTubeIngestError(str(e))
    return {**result, "title": downloaded.get("title"), "source_url": downloaded.get("source_url")}


# ---------------------------------------------------------------------------
# Background jobs
# ---------------------------------------------------------------------------

class IngestJobManager:
    """In-memory registry of ingestion jobs, executed one at a time."""

    ACTIVE_STATUSES = ("queued", "running")

    def __init__(self, runner=ingest_youtube_url):
        self._runner = runner
        self._jobs = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ingest")

    def submit(self, url: str, force: bool = False) -> dict:
        youtube_id = parse_youtube_id(url)  # validate before queueing
        video_id = video_id_for_youtube(youtube_id)

        with self._lock:
            # Don't queue the same video twice while it's still being processed.
            for job in self._jobs.values():
                if job["video_id"] == video_id and job["status"] in self.ACTIVE_STATUSES:
                    return dict(job)

            now = time.time()
            job = {
                "job_id": uuid.uuid4().hex[:12],
                "url": canonical_youtube_url(youtube_id),
                "video_id": video_id,
                "status": "queued",
                "stage": "queued",
                "message": "Waiting to start...",
                "title": None,
                "error": None,
                "warnings": [],
                "created_at": now,
                "updated_at": now,
            }
            self._jobs[job["job_id"]] = job

        self._executor.submit(self._run, job["job_id"], force)
        return dict(job)

    def get(self, job_id: str):
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job else None

    def list(self) -> list:
        with self._lock:
            jobs = [dict(j) for j in self._jobs.values()]
        return sorted(jobs, key=lambda j: j["created_at"], reverse=True)

    def _update(self, job_id: str, **fields):
        with self._lock:
            self._jobs[job_id].update(fields, updated_at=time.time())

    def _run(self, job_id: str, force: bool):
        job = self.get(job_id)
        self._update(job_id, status="running", stage="starting", message="Starting...")

        def progress(stage, message):
            self._update(job_id, stage=stage, message=message)

        try:
            result = self._runner(job["url"], progress=progress, force=force)
        except Exception as e:  # report every failure on the job, never crash the worker
            self._update(job_id, status="failed", stage="failed", message="Ingestion failed.", error=str(e))
            return

        self._update(
            job_id,
            status="completed",
            stage="completed",
            message="Video is indexed and searchable.",
            title=result.get("title"),
            warnings=result.get("warnings", []),
        )


job_manager = IngestJobManager()
