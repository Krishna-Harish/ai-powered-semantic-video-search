"""
scripts/build_text_index.py  (Milestones 2-5)

For every video in data/videos/:
    1. Transcribe audio (Whisper)   -- skipped if already done
    2. Chunk the transcript          -- skipped if already done
    3. Embed the chunks (Sentence Transformer) -- skipped if already done
Then builds one combined FAISS text index over all videos.

Usage (from project root):
    python scripts/build_text_index.py
    python scripts/build_text_index.py --force   (re-run everything from scratch)
"""

import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.config import VIDEOS_DIR, AUDIO_DIR, SUPPORTED_VIDEO_EXTENSIONS
from backend.services.transcriber import transcribe_audio, TranscriptionError
from backend.services.transcript_chunker import chunk_transcript, ChunkingError
from backend.services.text_embeddings import embed_video_chunks, TextEmbeddingError
from backend.services.text_index import build_text_index, TextIndexError


def get_video_id(filename: str) -> str:
    return os.path.splitext(filename)[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Re-run all steps even if outputs exist.")
    args = parser.parse_args()

    video_files = [
        f for f in os.listdir(VIDEOS_DIR)
        if os.path.splitext(f)[1].lower() in SUPPORTED_VIDEO_EXTENSIONS
    ]
    if not video_files:
        print(f"No videos found in {VIDEOS_DIR}. Run scripts/process_videos.py first.")
        return

    for filename in sorted(video_files):
        video_id = get_video_id(filename)
        audio_path = os.path.join(AUDIO_DIR, f"{video_id}.wav")

        if not os.path.isfile(audio_path):
            print(f"[{video_id}] SKIPPED -- no audio found. Run scripts/process_videos.py first.")
            continue

        print(f"[{video_id}] Transcribing...")
        try:
            transcribe_audio(audio_path, video_id, force=args.force)
        except TranscriptionError as e:
            print(f"[{video_id}] ERROR during transcription: {e}")
            continue

        print(f"[{video_id}] Chunking transcript...")
        try:
            chunk_transcript(video_id)
        except ChunkingError as e:
            print(f"[{video_id}] ERROR during chunking: {e}")
            continue

        print(f"[{video_id}] Generating text embeddings...")
        try:
            embed_video_chunks(video_id, force=args.force)
        except TextEmbeddingError as e:
            print(f"[{video_id}] ERROR during embedding: {e}")
            continue

        print(f"[{video_id}] Done.\n")

    print("Building combined text FAISS index...")
    try:
        index_path = build_text_index()
        print(f"Text index built at: {index_path}")
    except TextIndexError as e:
        print(f"ERROR building text index: {e}")


if __name__ == "__main__":
    main()
