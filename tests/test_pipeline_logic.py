"""
tests/test_pipeline_logic.py

Automated tests for everything that does NOT require torch/faiss/whisper
to be installed: video processing (real ffmpeg), transcript chunking
(pure logic), and timestamp formatting.

Run with:  python -m pytest tests/ -v
       or: python tests/test_pipeline_logic.py
"""

import os
import sys
import json
import shutil
import subprocess
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.services.video_processor import process_video, VideoProcessingError
from backend.services.transcript_chunker import chunk_transcript
from backend.services.search import _format_timestamp
from backend.config import AUDIO_DIR, FRAMES_DIR, TRANSCRIPTS_DIR, METADATA_DIR, VIDEOS_DIR


class TestVideoProcessor(unittest.TestCase):
    TEST_VIDEO_ID = "unittest_video"

    @classmethod
    def setUpClass(cls):
        cls.video_path = os.path.join(VIDEOS_DIR, f"{cls.TEST_VIDEO_ID}.mp4")
        subprocess.run([
            "ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=duration=8:size=160x120:rate=5",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=8",
            "-c:v", "libx264", "-c:a", "aac", "-shortest", cls.video_path,
            "-loglevel", "error",
        ], check=True)

    @classmethod
    def tearDownClass(cls):
        for path in [
            cls.video_path,
            os.path.join(AUDIO_DIR, f"{cls.TEST_VIDEO_ID}.wav"),
            os.path.join(METADATA_DIR, f"{cls.TEST_VIDEO_ID}_frames.json"),
        ]:
            if os.path.isfile(path):
                os.remove(path)
        frames_dir = os.path.join(FRAMES_DIR, cls.TEST_VIDEO_ID)
        if os.path.isdir(frames_dir):
            shutil.rmtree(frames_dir)

    def test_process_video_produces_audio_and_frames(self):
        result = process_video(self.video_path, self.TEST_VIDEO_ID)
        self.assertTrue(os.path.isfile(result["audio_path"]))
        self.assertGreater(os.path.getsize(result["audio_path"]), 0)
        self.assertGreater(result["num_frames"], 0)
        self.assertAlmostEqual(result["duration_seconds"], 8.0, delta=0.5)

    def test_missing_video_raises(self):
        with self.assertRaises(VideoProcessingError):
            process_video(os.path.join(VIDEOS_DIR, "does_not_exist.mp4"), "nope")

    def test_unsupported_format_raises(self):
        bad_path = os.path.join(VIDEOS_DIR, "bad.txt")
        with open(bad_path, "w") as f:
            f.write("not a video")
        try:
            with self.assertRaises(VideoProcessingError):
                process_video(bad_path, "bad")
        finally:
            os.remove(bad_path)


class TestTranscriptChunker(unittest.TestCase):
    TEST_VIDEO_ID = "unittest_chunk_video"

    @classmethod
    def setUpClass(cls):
        segments = []
        t = 0.0
        for i in range(8):
            segments.append({"start": round(t, 1), "end": round(t + 4.0, 1), "text": f"sentence number {i}"})
            t += 4.0
        transcript = {"video_id": cls.TEST_VIDEO_ID, "task": "translate", "segments": segments}
        os.makedirs(TRANSCRIPTS_DIR, exist_ok=True)
        with open(os.path.join(TRANSCRIPTS_DIR, f"{cls.TEST_VIDEO_ID}.json"), "w") as f:
            json.dump(transcript, f)

    @classmethod
    def tearDownClass(cls):
        for path in [
            os.path.join(TRANSCRIPTS_DIR, f"{cls.TEST_VIDEO_ID}.json"),
            os.path.join(METADATA_DIR, f"{cls.TEST_VIDEO_ID}_chunks.json"),
        ]:
            if os.path.isfile(path):
                os.remove(path)

    def test_chunking_preserves_timestamps_and_merges(self):
        path = chunk_transcript(self.TEST_VIDEO_ID)
        chunks = json.load(open(path))
        self.assertGreater(len(chunks), 0)
        # every chunk's start must be <= end, and text must be non-empty
        for c in chunks:
            self.assertLessEqual(c["start"], c["end"])
            self.assertTrue(c["text"].strip())
        # chunks should be in chronological, non-overlapping order
        for a, b in zip(chunks, chunks[1:]):
            self.assertLessEqual(a["end"], b["start"] + 0.01)


class TestTimestampFormatting(unittest.TestCase):
    def test_format_timestamp(self):
        self.assertEqual(_format_timestamp(0), "00:00")
        self.assertEqual(_format_timestamp(4), "00:04")
        self.assertEqual(_format_timestamp(65), "01:05")
        self.assertEqual(_format_timestamp(143.2), "02:23")


if __name__ == "__main__":
    unittest.main(verbosity=2)
