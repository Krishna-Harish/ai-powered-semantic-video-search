"""
tests/test_fusion.py

Multimodal score fusion. Uses small hand-made vector spaces in which every
dimension is one "concept", so a query vector is one-hot and each raw
cosine score is simply a vector component -- the expected ranking is known
exactly. A real FAISS index is used where faiss is installed.

Run with:  python -m pytest tests/ -v
"""

import os
import sys
import json
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.services import fusion
from backend.services.fusion import (
    VectorStore, calibrate, fuse_scores, rrf_scores, calibration_reliable, normalize_weights,
    build_moments, search_stores,
)
from backend.config import (
    TEXT_SCORE_FLOOR, TEXT_SCORE_CEILING, VISUAL_SCORE_FLOOR, VISUAL_SCORE_CEILING,
)

try:
    import faiss
    HAVE_FAISS = True
except ImportError:
    HAVE_FAISS = False

# Text concepts:   0 revenue, 1 database normalization, 2 cars (never discussed)
# Visual concepts: 0 graph,   1 red car,                2 talking head
TEXT_CHUNKS = [
    # video "lec": four 20 s chunks
    ({"video_id": "lec", "chunk_id": 0, "start": 0.0, "end": 20.0, "text": "Our revenue grew 20% this quarter."},
     [0.60, 0.05, 0.02]),
    ({"video_id": "lec", "chunk_id": 1, "start": 20.0, "end": 40.0, "text": "Database normalization removes redundancy."},
     [0.05, 0.70, 0.02]),
    ({"video_id": "lec", "chunk_id": 2, "start": 40.0, "end": 60.0, "text": "Let's take a short break."},
     [0.05, 0.05, 0.05]),
    ({"video_id": "lec", "chunk_id": 3, "start": 60.0, "end": 80.0, "text": "Revenue is mentioned only briefly here."},
     [0.30, 0.05, 0.02]),
]
FRAMES = [
    # "lec": a frame every 5 s
    ("lec", 0.0, [0.15, 0.10, 0.27]), ("lec", 5.0, [0.20, 0.10, 0.25]),
    ("lec", 10.0, [0.28, 0.10, 0.20]),  # graph shown while revenue is discussed
    ("lec", 15.0, [0.22, 0.10, 0.20]),
    ("lec", 20.0, [0.12, 0.10, 0.28]), ("lec", 25.0, [0.12, 0.10, 0.28]),
    ("lec", 30.0, [0.12, 0.10, 0.28]), ("lec", 35.0, [0.12, 0.10, 0.28]),
    ("lec", 40.0, [0.12, 0.12, 0.20]),
    ("lec", 45.0, [0.12, 0.29, 0.15]),  # red car on screen during the break
    ("lec", 50.0, [0.12, 0.12, 0.20]), ("lec", 55.0, [0.12, 0.12, 0.20]),
    ("lec", 60.0, [0.12, 0.10, 0.27]), ("lec", 65.0, [0.12, 0.10, 0.27]),
    ("lec", 70.0, [0.12, 0.10, 0.27]), ("lec", 75.0, [0.12, 0.10, 0.27]),
    # "silent": no transcript at all
    ("silent", 0.0, [0.10, 0.12, 0.10]),
    ("silent", 5.0, [0.10, 0.31, 0.10]),   # red car
    ("silent", 10.0, [0.10, 0.30, 0.10]),  # still the car (adjacent -> merged moment)
    ("silent", 30.0, [0.10, 0.12, 0.10]),
]


def text_store():
    return VectorStore(np.array([v for _, v in TEXT_CHUNKS]), [m for m, _ in TEXT_CHUNKS])


def visual_store():
    meta = [{"video_id": vid, "timestamp": t, "filename": f"{vid}_{int(t):03d}.jpg",
             "path": f"/data/frames/{vid}/{vid}_{int(t):03d}.jpg"} for vid, t, _ in FRAMES]
    return VectorStore(np.array([v for _, _, v in FRAMES]), meta)


def q(dim, index):
    v = np.zeros(dim, dtype="float32")
    if index is not None:
        v[index] = 1.0
    return v


def run(text_concept, visual_concept, ts=True, vs=True, **kwargs):
    return search_stores(
        "q", text_store() if ts else None, visual_store() if vs else None,
        q(3, text_concept), q(3, visual_concept), top_k=kwargs.pop("top_k", 10), **kwargs)


class TestNormalizationAndFusion(unittest.TestCase):
    def test_calibration_maps_model_specific_ranges_to_0_1(self):
        self.assertEqual(calibrate(TEXT_SCORE_FLOOR, TEXT_SCORE_FLOOR, TEXT_SCORE_CEILING), 0.0)
        self.assertEqual(calibrate(TEXT_SCORE_CEILING, TEXT_SCORE_FLOOR, TEXT_SCORE_CEILING), 1.0)
        self.assertEqual(calibrate(-0.3, TEXT_SCORE_FLOOR, TEXT_SCORE_CEILING), 0.0)
        self.assertEqual(calibrate(0.99, TEXT_SCORE_FLOOR, TEXT_SCORE_CEILING), 1.0)
        self.assertIsNone(calibrate(None, 0, 1))
        # The same raw cosine means very different things for the two models:
        self.assertLess(calibrate(0.28, TEXT_SCORE_FLOOR, TEXT_SCORE_CEILING), 0.3)       # weak text match
        self.assertGreater(calibrate(0.28, VISUAL_SCORE_FLOOR, VISUAL_SCORE_CEILING), 0.8)  # strong CLIP match

    def test_soft_or_fusion_properties(self):
        w = (0.6, 0.4)
        self.assertEqual(fuse_scores(0, 0, *w), 0.0)
        self.assertAlmostEqual(fuse_scores(1, 0, *w), 1.0)
        self.assertAlmostEqual(fuse_scores(0, 1, *w), 1.0)
        # one modality alone is enough to score well...
        self.assertGreater(fuse_scores(0.9, 0, *w), 0.7)
        self.assertGreater(fuse_scores(0, 0.9, *w), 0.55)
        # ...agreement reinforces: both > either alone
        both = fuse_scores(0.7, 0.7, *w)
        self.assertGreater(both, fuse_scores(0.7, 0, *w))
        self.assertGreater(both, fuse_scores(0, 0.7, *w))
        # monotonic and weight-sensitive; None == no evidence
        self.assertGreater(fuse_scores(0.8, 0.5, *w), fuse_scores(0.8, 0.4, *w))
        self.assertGreater(fuse_scores(0.8, 0, 0.6, 0.4), fuse_scores(0.8, 0, 0.4, 0.6))
        self.assertEqual(fuse_scores(None, 0.5, *w), fuse_scores(0, 0.5, *w))

    def test_weights(self):
        self.assertEqual(normalize_weights(0.6, 0.4, True, True), (0.6, 0.4))
        self.assertEqual(normalize_weights(3, 1, True, True), (0.75, 0.25))
        self.assertEqual(normalize_weights(0.6, 0.4, False, True), (0.0, 1.0))   # text index missing
        self.assertEqual(normalize_weights(None, None, True, False), (1.0, 0.0))
        for args in ((0, 0, True, True), (-1, 1, True, True), (0, 1, True, False)):
            with self.assertRaises(ValueError):
                normalize_weights(*args)

    def test_rrf(self):
        s = rrf_scores({"a": 1, "b": 2}, {"b": 1, "c": 2}, 0.5, 0.5, k=60)
        self.assertGreater(s["b"], s["a"])  # good in both lists beats best in one
        self.assertGreater(s["a"], s["c"])
        self.assertLessEqual(max(s.values()), 1.0)
        self.assertAlmostEqual(rrf_scores({"x": 1}, {"x": 1}, 0.5, 0.5)["x"], 1.0)

    def test_calibration_reliability_check(self):
        self.assertTrue(calibration_reliable([0.1, 0.5, None, 0.3]))
        self.assertFalse(calibration_reliable([3.2, 7.5, 1.1]))   # not cosine similarities
        self.assertTrue(calibration_reliable([0.4, 0.4, 0.4]))    # flat = nothing relevant, still cosine
        self.assertTrue(calibration_reliable([]))


class TestMoments(unittest.TestCase):
    def test_frames_attach_to_chunks_and_loose_frames_merge(self):
        ts, vs = text_store(), visual_store()
        text_hits = [(0, 0.6), (1, 0.7)]
        visual_hits = [(2, 0.30), (9, 0.29), (17, 0.31), (18, 0.30)]  # lec@10, lec@45, silent@5, silent@10
        moments = build_moments(text_hits, visual_hits, ts, vs)
        by_key = {(m["video_id"], m["start"]): m for m in moments}

        self.assertEqual(len(moments), 4)  # chunk0 (text+frame), chunk1, chunk2 (via frame), silent 5-15
        self.assertEqual(by_key[("lec", 0.0)]["frame_rows"], [0, 1, 2, 3])  # all frames sampled in 0-20 s
        self.assertEqual(by_key[("lec", 40.0)]["chunk_row"], 2)  # frame at 45 s found its chunk
        silent = by_key[("silent", 5.0)]
        self.assertIsNone(silent["chunk_row"])
        self.assertEqual((silent["frame_rows"], silent["end"]), ([17, 18], 15.0))  # merged

    def test_long_silent_stretch_is_split_into_chunk_sized_moments(self):
        meta = [{"video_id": "s", "timestamp": float(t), "filename": "f", "path": "f"} for t in range(0, 300, 5)]
        vs = VectorStore(np.full((len(meta), 3), 0.2), meta)
        moments = build_moments([], [(i, 0.2) for i in range(len(meta))], None, vs)
        self.assertGreater(len(moments), 5)
        self.assertTrue(all(m["end"] - m["start"] <= 40 for m in moments))

    def test_without_text_store_all_frames_are_visual_moments(self):
        moments = build_moments([], [(2, 0.3), (9, 0.29)], None, visual_store())
        self.assertTrue(all(m["chunk_row"] is None for m in moments))
        self.assertEqual(len(moments), 2)  # 10 s and 45 s are not adjacent


class TestMultimodalRanking(unittest.TestCase):
    def test_combined_query_ranks_moment_with_both_kinds_of_evidence_first(self):
        # "the presenter showing a graph while discussing revenue"
        out = run(text_concept=0, visual_concept=0)
        top = out["results"][0]
        self.assertEqual((top["video_id"], top["start"]), ("lec", 0.0))
        self.assertEqual(top["matched_by"], ["text", "visual"])
        self.assertEqual(top["frame_timestamp"], 10.0)  # the graph frame is the evidence shown
        self.assertEqual(top["text_score"], 0.6)
        self.assertEqual(top["visual_score"], 0.28)
        self.assertEqual(out["method"], "calibrated")
        # agreement beats either modality alone
        alone_t = fuse_scores(top["text_norm"], 0, 0.6, 0.4)
        alone_v = fuse_scores(0, top["visual_norm"], 0.6, 0.4)
        self.assertGreater(top["score"], max(alone_t, alone_v))
        # the weaker revenue mention (60-80 s, no graph) ranks below it
        second = [r for r in out["results"] if r["start"] == 60.0][0]
        self.assertLess(second["score"], top["score"])

    def test_visual_only_query(self):
        # "a red car driving through a street": nothing in the transcripts
        out = run(text_concept=2, visual_concept=1)
        results = out["results"]
        self.assertTrue(results)
        self.assertTrue(all(r["matched_by"] == ["visual"] for r in results))
        self.assertEqual([r["video_id"] for r in results[:2]], ["silent", "lec"])  # 0.31 > 0.29
        silent = results[0]
        self.assertIsNone(silent["text"])           # no transcript for this video
        self.assertIsNone(silent["text_score"])
        self.assertEqual(silent["timestamp"], 5.0)  # plays from the car frame, not the window start
        lec = [r for r in results if r["video_id"] == "lec"][0]
        self.assertEqual(lec["text"], "Let's take a short break.")  # excerpt still shown as context
        self.assertEqual(lec["frame_timestamp"], 45.0)

    def test_text_only_query(self):
        # "explain database normalization": frames add nothing
        out = run(text_concept=1, visual_concept=2)  # visual query matches only generic talking heads
        top = out["results"][0]
        self.assertEqual((top["video_id"], top["start"]), ("lec", 20.0))
        self.assertIn("text", top["matched_by"])
        self.assertGreater(top["score"], 0.75)

    def test_strong_single_modality_beats_weak_agreement(self):
        out = run(text_concept=0, visual_concept=0)
        scores = {r["start"]: r for r in out["results"] if r["video_id"] == "lec"}
        # 60-80 s: weak text (0.30) and weak visual (0.12) -> below a strong single-modality match
        strong_single = fuse_scores(calibrate(0.6, TEXT_SCORE_FLOOR, TEXT_SCORE_CEILING), 0, 0.6, 0.4)
        self.assertLess(scores[60.0]["score"], strong_single)

    def test_results_sorted_and_deduplicated(self):
        out = run(text_concept=0, visual_concept=0)
        scores = [r["score"] for r in out["results"]]
        self.assertEqual(scores, sorted(scores, reverse=True))
        keys = [(r["video_id"], r["start"]) for r in out["results"]]
        self.assertEqual(len(keys), len(set(keys)))

    def test_no_match_returns_empty(self):
        out = run(text_concept=None, visual_concept=None)  # zero query vectors: nothing relates
        self.assertEqual(out["results"], [])

    def test_missing_modalities(self):
        visual_only = run(text_concept=0, visual_concept=1, ts=False)
        self.assertEqual(visual_only["modalities"], ["visual"])
        self.assertEqual(visual_only["visual_weight"], 1.0)
        self.assertTrue(all(r["text"] is None for r in visual_only["results"]))

        text_only = run(text_concept=1, visual_concept=None, vs=False)
        self.assertEqual(text_only["modalities"], ["text"])
        self.assertEqual(text_only["results"][0]["start"], 20.0)
        self.assertTrue(all(r["frame_filename"] is None for r in text_only["results"]))

    def test_weights_change_the_ranking(self):
        # visual-heavy weights favour the car moment for a mixed query
        mixed = dict(text_concept=0, visual_concept=1)
        text_heavy = run(**mixed, text_weight=0.9, visual_weight=0.1)["results"][0]
        visual_heavy = run(**mixed, text_weight=0.1, visual_weight=0.9)["results"][0]
        self.assertEqual(text_heavy["start"], 0.0)       # revenue chunk
        self.assertEqual(visual_heavy["video_id"], "silent")  # red car

    def test_rank_fusion_fallback_when_scores_are_not_cosines(self):
        vs = visual_store()
        vs.vectors = vs.vectors * 10  # e.g. an index built without normalization
        out = search_stores("q", text_store(), vs, q(3, 0), q(3, 0), top_k=5)
        self.assertEqual(out["method"], "rrf")
        self.assertEqual((out["results"][0]["video_id"], out["results"][0]["start"]), ("lec", 0.0))
        forced = run(text_concept=0, visual_concept=0, method="rrf")
        self.assertEqual(forced["method"], "rrf")
        self.assertEqual(forced["results"][0]["start"], 0.0)

    def test_candidate_lists_are_bounded(self):
        # many frames: only FUSION_VISUAL_CANDIDATES are considered, re-scoring is one matmul
        n = 5000
        meta = [{"video_id": "long", "timestamp": float(i * 5), "filename": f"{i}.jpg", "path": f"{i}.jpg"}
                for i in range(n)]
        vecs = np.zeros((n, 3), dtype="float32")
        vecs[:, 1] = 0.1
        vecs[4321, 1] = 0.31
        vs = VectorStore(vecs, meta)
        with mock.patch.object(vs, "search", wraps=vs.search) as search:
            out = search_stores("q", None, vs, None, q(3, 1), top_k=3)
        self.assertEqual(search.call_args[0][1], fusion.FUSION_VISUAL_CANDIDATES)
        self.assertEqual(out["results"][0]["frame_timestamp"], 4321 * 5.0)


@unittest.skipUnless(HAVE_FAISS, "faiss not installed")
class TestWithRealIndexes(unittest.TestCase):
    """The real loading path: FAISS index files + metadata JSON, as built by the pipeline."""

    def setUp(self):
        from backend.services import text_index, visual_index
        self.dir = tempfile.mkdtemp()
        self.paths = {}
        for name, mod, store in (("text", text_index, text_store()), ("visual", visual_index, visual_store())):
            idx = faiss.IndexFlatIP(store.vectors.shape[1])
            idx.add(store.vectors)
            ip, mp = os.path.join(self.dir, f"{name}.index"), os.path.join(self.dir, f"{name}.json")
            faiss.write_index(idx, ip)
            with open(mp, "w") as f:
                json.dump(store.metadata, f)
            self.paths[name] = (mod, ip, mp)
        fusion._STORE_CACHE.clear()

    def tearDown(self):
        shutil.rmtree(self.dir)
        fusion._STORE_CACHE.clear()

    def patched(self, missing=()):
        patches = []
        for name, (mod, ip, mp) in self.paths.items():
            if name in missing:
                ip = mp = os.path.join(self.dir, "does-not-exist")
            patches += [mock.patch.object(mod, "INDEX_PATH", ip), mock.patch.object(mod, "METADATA_PATH", mp)]
        patches += [
            mock.patch("backend.services.text_embeddings.embed_text_query", return_value=q(3, 0)),
            mock.patch("backend.services.visual_embeddings.embed_text_query_clip", return_value=q(3, 0)),
        ]
        return patches

    def search(self, missing=(), **kwargs):
        patches = self.patched(missing)
        for p in patches:
            p.start()
        try:
            return fusion.multimodal_search("graph while discussing revenue", **kwargs)
        finally:
            for p in patches:
                p.stop()

    def test_end_to_end_and_store_cache(self):
        out = self.search(top_k=3)
        self.assertEqual((out["results"][0]["video_id"], out["results"][0]["start"]), ("lec", 0.0))
        self.assertEqual(out["warnings"], [])

        _, ip, mp = self.paths["visual"]
        first = fusion._load_store(ip, mp)
        self.assertIs(fusion._load_store(ip, mp), first)  # cached, not re-read from disk
        np.testing.assert_allclose(first.vectors, visual_store().vectors, rtol=1e-6)
        os.utime(ip, (os.path.getmtime(ip) + 5, os.path.getmtime(ip) + 5))
        self.assertIsNot(fusion._load_store(ip, mp), first)  # index rebuilt -> reloaded

    def test_missing_index_falls_back_with_warning(self):
        out = self.search(missing=("text",))
        self.assertEqual(out["modalities"], ["visual"])
        self.assertIn("Transcript search unavailable", out["warnings"][0])
        with self.assertRaises(fusion.FusionUnavailableError):
            self.search(missing=("text", "visual"))

    def test_invalid_queries(self):
        for bad in ("", "   "):
            with self.assertRaises(ValueError):
                fusion.multimodal_search(bad)
        with self.assertRaises(ValueError):
            self.search(method="magic")


class TestApi(unittest.TestCase):
    def setUp(self):
        try:
            from fastapi.testclient import TestClient
            from backend.main import app
        except ImportError:
            self.skipTest("fastapi not installed")
        self.client = TestClient(app)

    def test_endpoint(self):
        fake = search_stores("revenue graph", text_store(), visual_store(), q(3, 0), q(3, 0), top_k=3)
        with mock.patch("backend.main.multimodal_search", return_value=fake) as call:
            res = self.client.post("/search/multimodal", json={"query": "revenue graph", "top_k": 3,
                                                              "text_weight": 0.7, "visual_weight": 0.3})
        self.assertEqual(res.status_code, 200, res.text)
        self.assertEqual(call.call_args.kwargs["text_weight"], 0.7)
        body = res.json()
        self.assertEqual(body["results"][0]["matched_by"], ["text", "visual"])
        self.assertIn("text_score", body["results"][0])

    def test_errors(self):
        self.assertEqual(self.client.post("/search/multimodal", json={"query": "  "}).status_code, 400)
        self.assertEqual(self.client.post("/search/multimodal", json={"query": "x", "text_weight": 2}).status_code, 422)
        with mock.patch("backend.main.multimodal_search",
                        side_effect=fusion.FusionUnavailableError("no index")):
            self.assertEqual(self.client.post("/search/multimodal", json={"query": "x"}).status_code, 409)


if __name__ == "__main__":
    unittest.main(verbosity=2)
