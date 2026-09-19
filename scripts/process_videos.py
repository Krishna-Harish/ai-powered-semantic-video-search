"""
scripts/process_videos.py  (Milestone 1)

Processes every video in data/videos/: extracts audio + sampled frames.
Idempotent -- re-running skips videos already processed.

Usage (from project root):
    python scripts/process_videos.py
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.config import VIDEOS_DIR, SUPPORTED_VIDEO_EXTENSIONS
from backend.services.video_processor import process_video, VideoProcessingError


def get_video_id(filename: str) -> str:
    return os.path.splitext(filename)[0]


def main():
    video_files = [
        f for f in os.listdir(VIDEOS_DIR)
        if os.path.splitext(f)[1].lower() in SUPPORTED_VIDEO_EXTENSIONS
    ]

    if not video_files:
        print(f"No video files found in {VIDEOS_DIR}. Add some videos and re-run.")
        return

    print(f"Found {len(video_files)} video(s) to process.\n")

    results = []
    for filename in sorted(video_files):
        video_path = os.path.join(VIDEOS_DIR, filename)
        video_id = get_video_id(filename)

        print(f"Processing '{filename}' (video_id='{video_id}')...")
        try:
            summary = process_video(video_path, video_id)
            print(f"  Duration: {summary['duration_seconds']:.1f}s")
            print(f"  Audio saved to: {summary['audio_path']}")
            print(f"  Frames saved to: {summary['frames_dir']} ({summary['num_frames']} frames)\n")
            results.append(summary)
        except VideoProcessingError as e:
            print(f"  ERROR processing '{filename}': {e}\n")

    print("---- Summary ----")
    for r in results:
        print(f"{r['video_id']}: {r['num_frames']} frames, {r['duration_seconds']:.1f}s")
    print(f"\nSuccessfully processed {len(results)} / {len(video_files)} videos.")


if __name__ == "__main__":
    main()
