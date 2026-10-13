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
import threading

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

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


def index_video(video_path: str, video_id: str, progress=None, force: bool = False) -> dict:
    """
    Processes one video end-to-end and rebuilds the search indexes.

    `progress` is an optional callback progress(stage, message) used to
    report which step is running (e.g. to a job tracker).

    Returns {"video_id", "duration_seconds", "num_frames", "text_indexed",
             "visual_indexed", "warnings": [...]}.
    Raises IndexingError if the video could not be processed at all, or if
    neither text nor visual indexing succeeded.
    """
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
    except (TranscriptionError, ChunkingError, TextEmbeddingError) as e:
        warnings.append(f"Text indexing skipped: {e}")

    visual_ok = False
    try:
        report("embedding_frames", "Generating visual (CLIP) embeddings...")
        embed_video_frames(video_id, force=force)
        visual_ok = True
    except VisualEmbeddingError as e:
        warnings.append(f"Visual indexing skipped: {e}")

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
