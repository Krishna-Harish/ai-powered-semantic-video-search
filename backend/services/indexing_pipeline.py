"""
indexing_pipeline.py

Runs the full per-video pipeline in one call, so a newly added video
(e.g. one just downloaded from YouTube) becomes searchable without
running the three scripts by hand:

    video file
        -> audio + frames            (video_processor, FFmpeg)
        -> transcript                (transcriber, Whisper)
        -> chunks -> text vectors    (transcript_chunker, text_embeddings)
        -> frame vectors             (visual_embeddings, CLIP)
        -> rebuild text + visual FAISS indexes

Text and visual indexing are independent: a video with no speech can still
be indexed visually, and vice versa. Only if BOTH fail is the video
considered not indexed.
"""

import os
import sys
import json
import time
import threading

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.config import EMBEDDINGS_DIR, METADATA_DIR
from backend.services.video_processor import process_video, VideoProcessingError
from backend.services.transcriber import transcribe_audio, TranscriptionError
from backend.services.transcript_chunker import chunk_transcript, ChunkingError
from backend.services.text_embeddings import embed_video_chunks, TextEmbeddingError
from backend.services.text_index import build_text_index, TextIndexError
from backend.services.visual_embeddings import embed_video_frames, VisualEmbeddingError
from backend.services.visual_index import build_visual_index, VisualIndexError

# The FAISS indexes are single files rebuilt from every video's embeddings,
# so two rebuilds must never run at the same time.
_index_lock = threading.Lock()


class IndexingError(Exception):
    pass


def _status_path(video_id: str) -> str:
    return os.path.join(METADATA_DIR, f"{video_id}_index_status.json")


def _save_status(video_id: str, error=None, warnings=()):
    """Remembers the last indexing outcome on disk, so the reason a video
    isn't searchable survives a server restart."""
    with open(_status_path(video_id), "w", encoding="utf-8") as f:
        json.dump({"error": error, "warnings": list(warnings), "time": time.time()}, f, indent=2)


def get_index_state(video_id: str) -> dict:
    """{"text_indexed", "visual_indexed", "index_problem"} for one video."""
    state = {
        "text_indexed": os.path.isfile(os.path.join(EMBEDDINGS_DIR, f"{video_id}_text.npy")),
        "visual_indexed": os.path.isfile(os.path.join(EMBEDDINGS_DIR, f"{video_id}_visual.npy")),
        "index_problem": None,
    }
    try:
        with open(_status_path(video_id), "r", encoding="utf-8") as f:
            status = json.load(f)
        problems = ([status["error"]] if status.get("error") else []) + status.get("warnings", [])
        state["index_problem"] = " | ".join(problems) or None
    except (OSError, ValueError, KeyError):
        pass
    return state


def _describe(e: Exception) -> str:
    """Our own errors already read well; for library errors include the type."""
    if isinstance(e, (TranscriptionError, ChunkingError, TextEmbeddingError, VisualEmbeddingError)):
        return str(e)
    return f"{type(e).__name__}: {e}"


def index_video(video_path: str, video_id: str, progress=None, force: bool = False) -> dict:
    """
    Processes one video end-to-end and rebuilds the search indexes.

    `progress` is an optional callback progress(stage, message) used to
    report which step is running (e.g. to a job tracker).

    Returns {"video_id", "duration_seconds", "num_frames", "text_indexed",
             "visual_indexed", "warnings": [...]}.
    Raises IndexingError if the video could not be processed at all, or if
    neither text nor visual indexing succeeded.
    The outcome is also saved to data/metadata/<video_id>_index_status.json.
    """
    try:
        result = _index_video(video_path, video_id, progress, force)
    except IndexingError as e:
        _save_status(video_id, error=str(e))
        raise
    _save_status(video_id, warnings=result["warnings"])
    return result


def _index_video(video_path, video_id, progress, force):
    def report(stage, message):
        if progress is not None:
            progress(stage, message)

    warnings = []

    report("extracting", "Extracting audio and frames with FFmpeg...")
    try:
        summary = process_video(video_path, video_id)
    except VideoProcessingError as e:
        raise IndexingError(f"Video processing failed: {e}")

    text_ok = False
    try:
        report("transcribing", "Transcribing audio with Whisper...")
        transcribe_audio(summary["audio_path"], video_id, force=force)
        report("chunking", "Chunking transcript...")
        chunk_transcript(video_id)
        report("embedding_text", "Generating text embeddings...")
        embed_video_chunks(video_id, force=force)
        text_ok = True
    except Exception as e:  # any failure here must not stop visual indexing
        warnings.append(f"Text indexing failed: {_describe(e)}")

    visual_ok = False
    try:
        report("embedding_frames", "Generating visual (CLIP) embeddings...")
        embed_video_frames(video_id, force=force)
        visual_ok = True
    except Exception as e:
        warnings.append(f"Visual indexing failed: {_describe(e)}")

    if not text_ok and not visual_ok:
        raise IndexingError("Neither text nor visual indexing succeeded. " + " | ".join(warnings))

    report("indexing", "Rebuilding search indexes...")
    with _index_lock:
        if text_ok:
            try:
                build_text_index()
            except TextIndexError as e:
                text_ok = False
                warnings.append(f"Text index rebuild failed: {e}")
        if visual_ok:
            try:
                build_visual_index()
            except VisualIndexError as e:
                visual_ok = False
                warnings.append(f"Visual index rebuild failed: {e}")

    if not text_ok and not visual_ok:
        raise IndexingError("Index rebuild failed. " + " | ".join(warnings))

    return {
        "video_id": video_id,
        "duration_seconds": summary["duration_seconds"],
        "num_frames": summary["num_frames"],
        "text_indexed": text_ok,
        "visual_indexed": visual_ok,
        "warnings": warnings,
    }
