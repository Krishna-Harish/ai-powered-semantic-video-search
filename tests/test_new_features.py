"""
tests/test_new_features.py

Tests for YouTube ingestion, timestamp playback and LLM summaries that run
without network access, an API key, or torch/faiss/whisper: yt-dlp, the
indexing pipeline and the Claude API call are replaced with fakes.

Run with:  python -m pytest tests/ -v
"""

import os
import sys
import json
import time
import shutil
import subprocess
import unittest
from unittest import mock

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.config import TRANSCRIPTS_DIR, METADATA_DIR, VIDEOS_DIR
from backend.services import summarizer
from backend.services.youtube_ingest import (
    parse_youtube_id,
    video_id_for_youtube,
    IngestJobManager,
)

try:
    from fastapi.testclient import TestClient
    from backend.main import app
    HAVE_FASTAPI = True
except ImportError:  # fastapi/httpx not installed
    HAVE_FASTAPI = False


class TestYouTubeUrlParsing(unittest.TestCase):
    VID = "dQw4w9WgXcQ"

    def test_accepts_common_url_forms(self):
        urls = [
            f"https://www.youtube.com/watch?v={self.VID}",
            f"https://youtube.com/watch?v={self.VID}&t=42s&list=PL123",
            f"https://m.youtube.com/watch?feature=share&v={self.VID}",
            f"https://youtu.be/{self.VID}?si=abc",
            f"https://www.youtube.com/shorts/{self.VID}",
            f"https://www.youtube.com/embed/{self.VID}",
            f"https://www.youtube.com/live/{self.VID}",
            f"youtube.com/watch?v={self.VID}",
            f"  https://music.youtube.com/watch?v={self.VID}  ",
        ]
        for url in urls:
            with self.subTest(url=url):
                self.assertEqual(parse_youtube_id(url), self.VID)

    def test_rejects_non_video_urls(self):
        for url in [
            "",
            "https://vimeo.com/123456",
            "https://www.youtube.com/",
            "https://www.youtube.com/watch?v=short",
            "https://www.youtube.com/playlist?list=PL123",
            "https://evil.example/watch?v=dQw4w9WgXcQ",
            "ftp://youtube.com/watch?v=dQw4w9WgXcQ",
        ]:
            with self.subTest(url=url):
                with self.assertRaises(ValueError):
                    parse_youtube_id(url)

    def test_video_id_is_filename_safe(self):
        self.assertEqual(video_id_for_youtube("a-b_c1234567"), "yt_a-b_c1234567")


class _FakeYoutubeDL:
    """Stands in for yt_dlp.YoutubeDL: 'downloads' by writing a small file."""
    meta = {}
    options = None

    def __init__(self, options):
        _FakeYoutubeDL.options = options

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def extract_info(self, url, download=False):
        return dict(self.meta, webpage_url=url)

    def process_ie_result(self, meta, download=True):
        path = self.options["outtmpl"].replace("%(ext)s", "mp4")
        with open(path, "wb") as f:
            f.write(b"fake video")


class TestYouTubeDownload(unittest.TestCase):
    VID = "zzTestVid01"

    def setUp(self):
        from backend.services import youtube_ingest
        self.module = youtube_ingest
        self.video_id = video_id_for_youtube(self.VID)
        self.fake_yt_dlp = mock.MagicMock(YoutubeDL=_FakeYoutubeDL)

    def tearDown(self):
        for path in [os.path.join(VIDEOS_DIR, f"{self.video_id}.mp4"), self.module.info_path(self.video_id)]:
            if os.path.isfile(path):
                os.remove(path)

    def download(self, meta):
        _FakeYoutubeDL.meta = meta
        with mock.patch.dict(sys.modules, {"yt_dlp": self.fake_yt_dlp}):
            return self.module.download_youtube_video(f"https://youtu.be/{self.VID}?t=5")

    def test_downloads_into_videos_dir_and_saves_metadata(self):
        result = self.download({"title": "Intro to CNNs", "channel": "Prof X", "duration": 600})
        self.assertEqual(result["video_path"], os.path.join(VIDEOS_DIR, f"{self.video_id}.mp4"))
        self.assertEqual(result["source_url"], f"https://www.youtube.com/watch?v={self.VID}")
        self.assertTrue(_FakeYoutubeDL.options["noplaylist"])
        self.assertEqual(_FakeYoutubeDL.options["merge_output_format"], "mp4")
        self.assertEqual(self.module.load_video_info(self.video_id)["title"], "Intro to CNNs")

        # second call is a no-op (no new download)
        _FakeYoutubeDL.options = None
        with mock.patch.dict(sys.modules, {"yt_dlp": self.fake_yt_dlp}):
            again = self.module.download_youtube_video(f"https://www.youtube.com/watch?v={self.VID}")
        self.assertIsNone(_FakeYoutubeDL.options)
        self.assertEqual(again["title"], "Intro to CNNs")

    def test_rejects_too_long_and_live_videos(self):
        with self.assertRaises(self.module.YouTubeIngestError):
            self.download({"title": "Marathon", "duration": 10 * 60 * 60})
        with self.assertRaises(self.module.YouTubeIngestError):
            self.download({"title": "Live", "is_live": True})
        self.assertIsNone(self.module.find_local_video(self.video_id))


class TestIngestJobManager(unittest.TestCase):
    URL = "https://youtu.be/dQw4w9WgXcQ"

    def wait_for(self, manager, job_id, timeout=5):
        deadline = time.time() + timeout
        while time.time() < deadline:
            job = manager.get(job_id)
            if job["status"] in ("completed", "failed"):
                return job
            time.sleep(0.02)
        self.fail("job did not finish")

    def test_successful_job_reports_progress_and_result(self):
        stages = []

        def runner(url, progress, force):
            progress("transcribing", "Transcribing...")
            stages.append(url)
            return {"title": "My video", "warnings": ["Visual indexing skipped: x"]}

        manager = IngestJobManager(runner=runner)
        job = manager.submit(self.URL)
        self.assertEqual(job["video_id"], "yt_dQw4w9WgXcQ")
        done = self.wait_for(manager, job["job_id"])
        self.assertEqual(done["status"], "completed")
        self.assertEqual(done["title"], "My video")
        self.assertEqual(done["warnings"], ["Visual indexing skipped: x"])
        self.assertEqual(stages, ["https://www.youtube.com/watch?v=dQw4w9WgXcQ"])

    def test_failed_job_keeps_error(self):
        def runner(url, progress, force):
            raise RuntimeError("Download failed: HTTP 403")

        manager = IngestJobManager(runner=runner)
        done = self.wait_for(manager, manager.submit(self.URL)["job_id"])
        self.assertEqual(done["status"], "failed")
        self.assertIn("403", done["error"])

    def test_same_video_is_not_queued_twice_while_active(self):
        import threading
        release = threading.Event()

        def runner(url, progress, force):
            release.wait(5)
            return {}

        manager = IngestJobManager(runner=runner)
        first = manager.submit(self.URL)
        second = manager.submit("https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=10")
        self.assertEqual(first["job_id"], second["job_id"])
        release.set()
        self.wait_for(manager, first["job_id"])
        self.assertEqual(len(manager.list()), 1)

    def test_invalid_url_raises_before_queueing(self):
        manager = IngestJobManager(runner=lambda *a, **k: {})
        with self.assertRaises(ValueError):
            manager.submit("https://example.com")
        self.assertEqual(manager.list(), [])


class TestTranscriptHelpers(unittest.TestCase):
    SEGMENTS = [
        {"start": 0.0, "end": 5.0, "text": "intro"},
        {"start": 5.0, "end": 12.0, "text": "gradient descent"},
        {"start": 12.0, "end": 20.0, "text": "backprop"},
        {"start": 65.0, "end": 70.0, "text": "summary"},
    ]

    def test_timestamps_round_trip(self):
        self.assertEqual(summarizer.format_timestamp(83), "01:23")
        self.assertEqual(summarizer.format_timestamp(3725), "1:02:05")
        self.assertEqual(summarizer.parse_timestamp("01:23"), 83)
        self.assertEqual(summarizer.parse_timestamp("[1:02:05]"), 3725)
        self.assertEqual(summarizer.parse_timestamp("12.5"), 12.5)
        self.assertEqual(summarizer.parse_timestamp(7), 7.0)
        for bad in ["", "abc", "1:2:3:4", "1:xx"]:
            with self.assertRaises(ValueError):
                summarizer.parse_timestamp(bad)

    def test_select_segments_uses_overlap(self):
        selected = summarizer.select_segments(self.SEGMENTS, 6, 13)
        self.assertEqual([s["text"] for s in selected], ["gradient descent", "backprop"])
        self.assertEqual(len(summarizer.select_segments(self.SEGMENTS)), 4)
        self.assertEqual(summarizer.select_segments(self.SEGMENTS, 30, 60), [])

    def test_snap_to_segment(self):
        self.assertEqual(summarizer.snap_to_segment(7, self.SEGMENTS), 5.0)    # inside a segment
        self.assertEqual(summarizer.snap_to_segment(40, self.SEGMENTS), 12.0)  # gap: nearest edge (ends at 20)
        self.assertEqual(summarizer.snap_to_segment(999, self.SEGMENTS), 65.0)

    def test_format_and_windows(self):
        text = summarizer.format_transcript(self.SEGMENTS[:2])
        self.assertEqual(text, "[00:00] intro\n[00:05] gradient descent")
        windows = summarizer.split_into_windows(self.SEGMENTS, max_chars=30)
        self.assertGreater(len(windows), 1)
        self.assertEqual(sum(len(w) for w in windows), len(self.SEGMENTS))
        self.assertEqual([s for w in windows for s in w], self.SEGMENTS)


class TestSummarize(unittest.TestCase):
    VIDEO_ID = "unittest_summary_video"

    @classmethod
    def setUpClass(cls):
        segments = [
            {"start": i * 10.0, "end": i * 10.0 + 10.0, "text": f"sentence {i}"} for i in range(12)
        ]
        with open(os.path.join(TRANSCRIPTS_DIR, f"{cls.VIDEO_ID}.json"), "w") as f:
            json.dump({"video_id": cls.VIDEO_ID, "segments": segments}, f)

    @classmethod
    def tearDownClass(cls):
        os.remove(os.path.join(TRANSCRIPTS_DIR, f"{cls.VIDEO_ID}.json"))

    def tearDown(self):
        for mode in summarizer.SUMMARY_MODES:
            path = summarizer._cache_path(self.VIDEO_ID, mode)
            if os.path.isfile(path):
                os.remove(path)

    def fake_llm(self, calls):
        def _call(system, prompt, schema):
            calls.append(prompt)
            return {
                "summary": "  A video about sentences.  ",
                "points": [
                    {"text": "First idea", "timestamp": "00:23"},   # snaps to 20.0
                    {"text": "Second idea", "timestamp": "not a time"},
                    {"text": "   ", "timestamp": "00:00"},          # dropped
                ],
            }
        return _call

    def test_whole_video_summary_snaps_timestamps_and_caches(self):
        calls = []
        with mock.patch.object(summarizer, "_call_llm", self.fake_llm(calls)):
            result = summarizer.summarize(self.VIDEO_ID, "summary")
            self.assertFalse(result["cached"])
            self.assertEqual(result["summary"], "A video about sentences.")
            self.assertEqual(result["points"], [
                {"text": "First idea", "timestamp": 20.0},
                {"text": "Second idea", "timestamp": None},
            ])
            self.assertIn("[01:50] sentence 11", calls[0])

            again = summarizer.summarize(self.VIDEO_ID, "summary")
            self.assertTrue(again["cached"])
            self.assertEqual(len(calls), 1)

            summarizer.summarize(self.VIDEO_ID, "summary", force=True)
            self.assertEqual(len(calls), 2)

    def test_section_summary_only_sends_that_section(self):
        calls = []
        with mock.patch.object(summarizer, "_call_llm", self.fake_llm(calls)):
            result = summarizer.summarize(self.VIDEO_ID, "section", start=30, end=60)
        self.assertEqual((result["start"], result["end"]), (30, 60))
        self.assertIn("sentence 3", calls[0])
        self.assertIn("sentence 5", calls[0])
        self.assertNotIn("sentence 2", calls[0])
        self.assertNotIn("sentence 6", calls[0])
        # timestamps are snapped within the section, never outside it
        self.assertEqual(result["points"][0]["timestamp"], 30.0)
        self.assertFalse(os.path.isfile(summarizer._cache_path(self.VIDEO_ID, "section")))

    def test_section_validation(self):
        with self.assertRaises(ValueError):
            summarizer.summarize(self.VIDEO_ID, "section", start=50, end=10)
        with self.assertRaises(ValueError):
            summarizer.summarize(self.VIDEO_ID, "section", start=500, end=600)
        with self.assertRaises(ValueError):
            summarizer.summarize(self.VIDEO_ID, "bogus")

    def test_long_transcript_is_map_reduced(self):
        calls = []
        with mock.patch.object(summarizer, "_call_llm", self.fake_llm(calls)), \
             mock.patch.object(summarizer, "LLM_TRANSCRIPT_WINDOW_CHARS", 60):
            summarizer.summarize(self.VIDEO_ID, "takeaways")
        self.assertGreater(len(calls), 2)            # several windows + one combine step
        self.assertIn("<part_summaries>", calls[-1])

    def test_missing_transcript(self):
        with self.assertRaises(summarizer.TranscriptNotFoundError):
            summarizer.summarize("no_such_video", "summary")


class TestAnswerQuestion(unittest.TestCase):
    PASSAGES = [
        {"video_id": "v1", "start": 10.0, "end": 30.0, "text": "Gradient descent updates weights."},
        {"video_id": "v2", "start": 50.0, "end": 70.0, "text": "CNNs use convolutions."},
    ]

    def test_citations_map_to_passages(self):
        prompts = []

        def fake(system, prompt, schema):
            prompts.append(prompt)
            return {"answerable": True, "answer": "It updates weights [1].", "citations": [1, 1, 7]}

        with mock.patch.object(summarizer, "_call_llm", fake):
            result = summarizer.answer_question("What is GD?", passages=self.PASSAGES)

        self.assertTrue(result["answerable"])
        self.assertEqual(len(result["citations"]), 1)  # duplicate and out-of-range refs dropped
        self.assertEqual(result["citations"][0]["video_id"], "v1")
        self.assertEqual(result["citations"][0]["start"], 10.0)
        self.assertIn('<passage id="2" video="v2" time="00:50-01:10">', prompts[0])

    def test_no_passages_skips_llm(self):
        with mock.patch.object(summarizer, "_call_llm", side_effect=AssertionError("called")):
            result = summarizer.answer_question("Anything?", passages=[])
        self.assertFalse(result["answerable"])
        self.assertEqual(result["citations"], [])


@unittest.skipUnless(HAVE_FASTAPI, "fastapi not installed")
class TestApi(unittest.TestCase):
    VIDEO_ID = "unittest_api_video"

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.video_path = os.path.join(VIDEOS_DIR, f"{cls.VIDEO_ID}.mp4")
        subprocess.run([
            "ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=duration=4:size=160x120:rate=5",
            "-c:v", "libx264", cls.video_path, "-loglevel", "error",
        ], check=True)

    @classmethod
    def tearDownClass(cls):
        os.remove(cls.video_path)

    def test_video_supports_range_requests_for_seeking(self):
        size = os.path.getsize(self.video_path)
        res = self.client.get(f"/video/{self.VIDEO_ID}", headers={"Range": "bytes=100-199"})
        self.assertEqual(res.status_code, 206)
        self.assertEqual(len(res.content), 100)
        self.assertEqual(res.headers["content-range"], f"bytes 100-199/{size}")

    def test_rejects_path_traversal(self):
        self.assertEqual(self.client.get("/frame/..%5C..%5Cetc/passwd").status_code, 400)
        self.assertEqual(self.client.get("/video/a%20b").status_code, 400)

    def test_ingest_rejects_bad_url(self):
        res = self.client.post("/ingest/youtube", json={"url": "https://example.com/x"})
        self.assertEqual(res.status_code, 400)

    def test_unknown_job(self):
        self.assertEqual(self.client.get("/ingest/jobs/nope").status_code, 404)

    def test_summarize_without_transcript_is_404(self):
        res = self.client.post("/summarize", json={"video_id": "no_such_video"})
        self.assertEqual(res.status_code, 404)

    def test_summarize_missing_api_key_is_503(self):
        with mock.patch.object(summarizer, "_call_llm", side_effect=summarizer.LLMUnavailableError("no key")):
            with mock.patch.object(summarizer, "load_segments", return_value=[{"start": 0, "end": 1, "text": "hi"}]):
                res = self.client.post("/summarize", json={"video_id": "x", "force": True})
        self.assertEqual(res.status_code, 503)

    def test_frontend_is_served(self):
        res = self.client.get("/app/")
        self.assertEqual(res.status_code, 200)
        self.assertIn("AI Video Search", res.text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
