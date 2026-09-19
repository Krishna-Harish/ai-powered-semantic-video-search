"""
transcript_chunker.py  (Milestone 3)

Merges Whisper's raw segments (often short, sentence-fragment-length)
into slightly larger, more semantically meaningful chunks before embedding.

Why chunk at all instead of embedding every raw Whisper segment directly?
A single Whisper segment can be as short as "and then we" -- too little
context for the sentence-transformer to produce a meaningful embedding.
Merging consecutive segments up to a target duration gives the embedding
model enough context to capture real meaning, while still keeping chunks
short enough to map back to a precise, useful timestamp range.
"""

import os
import json
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.config import (
    TRANSCRIPTS_DIR,
    METADATA_DIR,
    CHUNK_TARGET_DURATION_SECONDS,
    CHUNK_MAX_DURATION_SECONDS,
)


class ChunkingError(Exception):
    pass


def chunk_transcript(video_id: str) -> str:
    """
    Reads data/transcripts/<video_id>.json and produces merged chunks,
    saved to data/metadata/<video_id>_chunks.json.

    Each chunk:
    {
        "video_id": "...",
        "chunk_id": 0,
        "start": 0.0,
        "end": 18.4,
        "text": "merged text of several segments"
    }

    Returns the path to the chunks JSON.
    """
    transcript_path = os.path.join(TRANSCRIPTS_DIR, f"{video_id}.json")
    if not os.path.isfile(transcript_path):
        raise ChunkingError(f"Transcript not found for '{video_id}': {transcript_path}")

    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript = json.load(f)

    segments = transcript.get("segments", [])
    if len(segments) == 0:
        raise ChunkingError(f"Transcript for '{video_id}' has no segments to chunk.")

    chunks = []
    current_texts = []
    current_start = None
    current_end = None

    def flush_chunk():
        if current_texts:
            chunks.append({
                "video_id": video_id,
                "chunk_id": len(chunks),
                "start": current_start,
                "end": current_end,
                "text": " ".join(current_texts).strip(),
            })

    for seg in segments:
        text = seg["text"].strip()
        if not text:
            continue  # skip empty segments (e.g. filtered silence)

        if current_start is None:
            current_start = seg["start"]

        would_be_duration = seg["end"] - current_start

        if current_texts and would_be_duration > CHUNK_MAX_DURATION_SECONDS:
            # current chunk is full; flush it and start a new one with this segment
            flush_chunk()
            current_texts = [text]
            current_start = seg["start"]
            current_end = seg["end"]
        else:
            current_texts.append(text)
            current_end = seg["end"]
            if would_be_duration >= CHUNK_TARGET_DURATION_SECONDS:
                flush_chunk()
                current_texts = []
                current_start = None
                current_end = None

    flush_chunk()  # flush whatever remains at the end

    if len(chunks) == 0:
        raise ChunkingError(f"Chunking produced zero chunks for '{video_id}'.")

    output_path = os.path.join(METADATA_DIR, f"{video_id}_chunks.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    return output_path
