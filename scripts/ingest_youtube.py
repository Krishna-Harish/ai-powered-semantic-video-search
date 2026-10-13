"""
scripts/ingest_youtube.py

Downloads one or more YouTube videos and runs the full pipeline on each
(audio/frames -> transcript -> text + visual embeddings -> FAISS indexes).
Same as pasting the URL into the web UI, but from the command line.

Usage (from project root):
    python scripts/ingest_youtube.py https://www.youtube.com/watch?v=VIDEO_ID [more URLs...]
    python scripts/ingest_youtube.py --force URL   (re-run transcription/embedding)
"""

import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.services.youtube_ingest import ingest_youtube_url, YouTubeIngestError


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("urls", nargs="+", help="YouTube video URL(s)")
    parser.add_argument("--force", action="store_true", help="Re-run all steps even if outputs exist.")
    args = parser.parse_args()

    succeeded = 0
    for url in args.urls:
        print(f"== {url}")

        def progress(stage, message):
            print(f"  [{stage}] {message}")

        try:
            result = ingest_youtube_url(url, progress=progress, force=args.force)
        except (ValueError, YouTubeIngestError) as e:
            print(f"  ERROR: {e}\n")
            continue

        succeeded += 1
        print(f"  Indexed '{result.get('title') or result['video_id']}' as video_id='{result['video_id']}' "
              f"(text: {result['text_indexed']}, visual: {result['visual_indexed']})")
        for warning in result["warnings"]:
            print(f"  WARNING: {warning}")
        print()

    print(f"Successfully ingested {succeeded} / {len(args.urls)} videos.")


if __name__ == "__main__":
    main()
