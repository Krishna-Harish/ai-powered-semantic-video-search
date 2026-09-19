"""
scripts/build_visual_index.py  (Milestones 6-7)

For every video in data/videos/, embeds its sampled frames using CLIP
(skipped if already done), then builds one combined FAISS visual index.

Usage (from project root):
    python scripts/build_visual_index.py
    python scripts/build_visual_index.py --force
"""

import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.config import VIDEOS_DIR, FRAMES_DIR, SUPPORTED_VIDEO_EXTENSIONS
from backend.services.visual_embeddings import embed_video_frames, VisualEmbeddingError
from backend.services.visual_index import build_visual_index, VisualIndexError


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
        frames_dir = os.path.join(FRAMES_DIR, video_id)

        if not os.path.isdir(frames_dir) or not os.listdir(frames_dir):
            print(f"[{video_id}] SKIPPED -- no frames found. Run scripts/process_videos.py first.")
            continue

        print(f"[{video_id}] Generating visual (CLIP) embeddings...")
        try:
            embed_video_frames(video_id, force=args.force)
            print(f"[{video_id}] Done.\n")
        except VisualEmbeddingError as e:
            print(f"[{video_id}] ERROR during visual embedding: {e}")
            continue

    print("Building combined visual FAISS index...")
    try:
        index_path = build_visual_index()
        print(f"Visual index built at: {index_path}")
    except VisualIndexError as e:
        print(f"ERROR building visual index: {e}")


if __name__ == "__main__":
    main()
