"""
fusion.py

Multimodal search: combines transcript search (sentence-transformer + FAISS)
and visual search (CLIP + FAISS) into one ranking of video MOMENTS, so a
query can match what is said, what is shown, or both.

    query
      |-- MiniLM query vector --> text FAISS index   --> candidate chunks
      |-- CLIP   query vector --> visual FAISS index --> candidate frames
      v
    moments: a transcript chunk plus the frames sampled during it, or
             (where no transcript covers a frame) a visual-only frame window
      v
    re-score EVERY moment on BOTH modalities from the stored vectors
      v
    calibrate each modality's raw cosine score to 0..1, fuse, rank

Why re-score both modalities: a moment found by the transcript list also
gets its visual score (and vice versa), so text and visual evidence can
reinforce each other without requiring a moment to be in both top-k lists.

Why calibrate: MiniLM and CLIP cosine similarities have very different
distributions (a strong CLIP text-image match is ~0.3, a strong MiniLM
match ~0.6+), so raw scores are not comparable and are never added
directly. Each is mapped linearly between a per-model floor (unrelated)
and ceiling (strong match) to 0..1 -- see TEXT/VISUAL_SCORE_* in config.

Fusion (weighted soft-OR): fused = 1 - (1 - t)^w_t * (1 - v)^w_v
  * one strong modality is enough to rank high (no need for both to match);
  * agreement between modalities raises the score above either alone;
  * a modality with no evidence (0) neither helps nor hurts;
  * w_t / w_v (summing to 1) set how much each modality counts.

Fallback: reciprocal rank fusion (RRF), which uses only rank order, when a
modality's raw scores don't look like calibrated cosine similarities.

Nothing is re-embedded: the existing FAISS indexes and their metadata are
used as-is (loaded once and cached until the index files change).
"""

import os
import json
import bisect
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.config import (
    DEFAULT_TOP_K,
    FRAME_SAMPLE_INTERVAL_SECONDS,
    CHUNK_MAX_DURATION_SECONDS,
    FUSION_TEXT_WEIGHT,
    FUSION_VISUAL_WEIGHT,
    FUSION_METHOD,
    TEXT_SCORE_FLOOR,
    TEXT_SCORE_CEILING,
    VISUAL_SCORE_FLOOR,
    VISUAL_SCORE_CEILING,
    FUSION_TEXT_CANDIDATES,
    FUSION_VISUAL_CANDIDATES,
    FUSION_RRF_K,
    FUSION_MIN_SCORE,
    FUSION_EVIDENCE_THRESHOLD,
)

METHODS = ("auto", "calibrated", "rrf")


class FusionUnavailableError(Exception):
    """Neither the text nor the visual index can be searched."""
    pass


# ---------------------------------------------------------------------------
# Score normalization and fusion (pure functions)
# ---------------------------------------------------------------------------

def calibrate(raw, floor: float, ceiling: float):
    """Maps a raw cosine similarity to 0..1: <= floor -> 0, >= ceiling -> 1, linear between."""
    if raw is None:
        return None
    return float(np.clip((raw - floor) / (ceiling - floor), 0.0, 1.0))


def fuse_scores(text_norm, visual_norm, text_weight: float, visual_weight: float) -> float:
    """
    Weighted soft-OR of two normalized scores (None = no evidence of that kind).
    Weights must be >= 0 and sum to 1.
    """
    t = text_norm or 0.0
    v = visual_norm or 0.0
    return float(1.0 - (1.0 - t) ** text_weight * (1.0 - v) ** visual_weight)


def rrf_scores(text_ranks: dict, visual_ranks: dict, text_weight: float, visual_weight: float,
               k: int = FUSION_RRF_K) -> dict:
    """
    Reciprocal rank fusion. text_ranks / visual_ranks map moment key -> rank
    (1 = best). Returns key -> score scaled to 0..1 (1 = ranked first in
    every modality).
    """
    best = (text_weight + visual_weight) / (k + 1)
    keys = set(text_ranks) | set(visual_ranks)
    out = {}
    for key in keys:
        s = 0.0
        if key in text_ranks:
            s += text_weight / (k + text_ranks[key])
        if key in visual_ranks:
            s += visual_weight / (k + visual_ranks[key])
        out[key] = s / best if best > 0 else 0.0
    return out


def calibration_reliable(raw_scores) -> bool:
    """
    Calibration assumes cosine similarities of normalized vectors. If a
    modality's scores fall outside [-1, 1] (e.g. an index built without
    normalization, or a different model), the per-model floor/ceiling don't
    apply and rank fusion should be used instead. (Identical scores are NOT
    a reason: that just means nothing in that modality relates to the query.)
    """
    scores = np.asarray([s for s in raw_scores if s is not None], dtype=float)
    if len(scores) == 0:
        return True
    return bool(scores.max() <= 1.0 + 1e-3 and scores.min() >= -1.0 - 1e-3)


def normalize_weights(text_weight, visual_weight, text_available: bool, visual_available: bool):
    """Validates weights, drops unavailable modalities and rescales to sum to 1."""
    tw = FUSION_TEXT_WEIGHT if text_weight is None else float(text_weight)
    vw = FUSION_VISUAL_WEIGHT if visual_weight is None else float(visual_weight)
    if tw < 0 or vw < 0:
        raise ValueError("Weights must not be negative.")
    if tw + vw == 0:
        raise ValueError("At least one of the text/visual weights must be above 0.")
    if not text_available:
        tw = 0.0
    if not visual_available:
        vw = 0.0
    total = tw + vw
    if total == 0:
        raise ValueError("The only searchable modality has weight 0; raise its weight.")
    return tw / total, vw / total


# ---------------------------------------------------------------------------
# Vector stores (the existing FAISS indexes + metadata, cached)
# ---------------------------------------------------------------------------

class VectorStore:
    """
    One modality's index: FAISS for candidate search, plus the stored
    vectors (read once from the flat index) for exact re-scoring of
    specific chunks/frames, and per-video sorted lookups by time.
    """

    def __init__(self, vectors: np.ndarray, metadata: list, index=None):
        if len(vectors) != len(metadata):
            raise ValueError("Vector/metadata count mismatch.")
        self.vectors = np.asarray(vectors, dtype="float32")
        self.metadata = metadata
        self.index = index
        self._by_video = None

    def search(self, query_vector: np.ndarray, k: int) -> list:
        """[(row, raw_score)] best first, using FAISS when available."""
        k = min(k, len(self.metadata))
        if k <= 0:
            return []
        q = np.asarray(query_vector, dtype="float32").reshape(1, -1)
        if self.index is not None:
            scores, ids = self.index.search(q, k)
            return [(int(i), float(s)) for s, i in zip(scores[0], ids[0]) if i != -1]
        sims = self.vectors @ q[0]
        top = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i])) for i in top]

    def scores(self, rows: list, query_vector: np.ndarray) -> np.ndarray:
        """Exact similarity of the query to specific rows (one matrix product)."""
        if not rows:
            return np.zeros(0, dtype="float32")
        return self.vectors[rows] @ np.asarray(query_vector, dtype="float32")

    def by_video(self, time_key: str) -> dict:
        """video_id -> ([times sorted], [rows in the same order]); built once."""
        if self._by_video is None:
            groups = {}
            for row, m in enumerate(self.metadata):
                groups.setdefault(m["video_id"], []).append((float(m[time_key]), row))
            self._by_video = {
                vid: ([t for t, _ in sorted(items)], [r for _, r in sorted(items)])
                for vid, items in groups.items()
            }
        return self._by_video


_STORE_CACHE = {}


def _load_store(index_path: str, meta_path: str) -> VectorStore:
    """Loads a FAISS flat index + metadata once; reloads only if the files change."""
    import faiss

    stamp = (os.path.getmtime(index_path), os.path.getmtime(meta_path))
    cached = _STORE_CACHE.get(index_path)
    if cached and cached[0] == stamp:
        return cached[1]

    index = faiss.read_index(index_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    vectors = index.reconstruct_n(0, index.ntotal)
    store = VectorStore(vectors, metadata, index)
    _STORE_CACHE[index_path] = (stamp, store)
    return store


# ---------------------------------------------------------------------------
# Moments: merge chunk and frame candidates into time ranges
# ---------------------------------------------------------------------------

def _chunk_covering(text_store: VectorStore, video_id: str, t: float):
    """Row of the transcript chunk of `video_id` whose [start, end) contains t, or None."""
    starts, rows = text_store.by_video("start").get(video_id, ([], []))
    i = bisect.bisect_right(starts, t) - 1
    if i >= 0:
        chunk = text_store.metadata[rows[i]]
        if chunk["start"] <= t < chunk["end"]:
            return rows[i]
    return None


def _frames_in(visual_store: VectorStore, video_id: str, start: float, end: float,
               interval: float) -> list:
    """Frame rows sampled in [start, end); if none, the nearest frame within one interval."""
    times, rows = visual_store.by_video("timestamp").get(video_id, ([], []))
    if not times:
        return []
    lo = bisect.bisect_left(times, start)
    hi = bisect.bisect_left(times, end)
    if hi > lo:
        return rows[lo:hi]
    nearest = min(range(len(times)), key=lambda i: abs(times[i] - (start + end) / 2))
    if abs(times[nearest] - (start + end) / 2) <= interval:
        return [rows[nearest]]
    return []


def build_moments(text_hits: list, visual_hits: list, text_store, visual_store,
                  interval: float = FRAME_SAMPLE_INTERVAL_SECONDS) -> list:
    """
    Turns candidate chunk rows and frame rows into de-duplicated moments:
      * a transcript chunk (from either list) -> one moment with the frames
        sampled during it attached;
      * a frame with no transcript chunk around it -> a visual-only moment;
        adjacent visual-only frames of a video are merged into one, capped
        at the same length as a transcript chunk so a long silent stretch
        doesn't collapse into a single huge moment.
    """
    chunk_rows, loose_frames = set(), set()
    for row, _ in text_hits:
        chunk_rows.add(row)
    for row, _ in visual_hits:
        frame = visual_store.metadata[row]
        covering = _chunk_covering(text_store, frame["video_id"], float(frame["timestamp"])) if text_store else None
        if covering is not None:
            chunk_rows.add(covering)
        else:
            loose_frames.add(row)

    moments = []
    for row in sorted(chunk_rows):
        chunk = text_store.metadata[row]
        frames = (_frames_in(visual_store, chunk["video_id"], float(chunk["start"]), float(chunk["end"]), interval)
                  if visual_store else [])
        moments.append({
            "video_id": chunk["video_id"], "start": float(chunk["start"]), "end": float(chunk["end"]),
            "chunk_row": row, "frame_rows": list(frames),
        })

    by_video = {}
    for row in loose_frames:
        f = visual_store.metadata[row]
        by_video.setdefault(f["video_id"], []).append((float(f["timestamp"]), row))
    for video_id, frames in by_video.items():
        frames.sort()
        group = [frames[0]]
        for item in frames[1:] + [None]:
            if (item is not None and item[0] - group[-1][0] <= interval
                    and item[0] - group[0][0] < CHUNK_MAX_DURATION_SECONDS):
                group.append(item)
                continue
            moments.append({
                "video_id": video_id, "start": group[0][0], "end": group[-1][0] + interval,
                "chunk_row": None, "frame_rows": [r for _, r in group],
            })
            if item is not None:
                group = [item]
    return moments


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------

def rank_moments(moments: list, text_store, visual_store, text_query, visual_query,
                 text_weight: float, visual_weight: float, method: str = "auto",
                 top_k: int = DEFAULT_TOP_K) -> dict:
    """
    Scores every moment on both modalities, fuses and ranks them.
    Weights must already be normalized (see normalize_weights).
    Returns {"method": used method, "results": [...]}.
    """
    # Exact re-scoring with two matrix products (one per modality).
    chunk_rows = [m["chunk_row"] for m in moments if m["chunk_row"] is not None]
    text_raw = dict(zip(chunk_rows, text_store.scores(chunk_rows, text_query))) if text_store else {}
    frame_rows = sorted({r for m in moments for r in m["frame_rows"]})
    frame_raw = dict(zip(frame_rows, visual_store.scores(frame_rows, visual_query))) if visual_store else {}

    for m in moments:
        m["text_raw"] = float(text_raw[m["chunk_row"]]) if m["chunk_row"] in text_raw else None
        best = max(m["frame_rows"], key=lambda r: frame_raw[r], default=None) if visual_store else None
        m["best_frame_row"] = best
        m["visual_raw"] = float(frame_raw[best]) if best is not None else None
        m["text_norm"] = calibrate(m["text_raw"], TEXT_SCORE_FLOOR, TEXT_SCORE_CEILING)
        m["visual_norm"] = calibrate(m["visual_raw"], VISUAL_SCORE_FLOOR, VISUAL_SCORE_CEILING)

    if method == "auto":
        reliable = (calibration_reliable([m["text_raw"] for m in moments])
                    and calibration_reliable([m["visual_raw"] for m in moments]))
        method = "calibrated" if reliable else "rrf"

    if method == "calibrated":
        for m in moments:
            m["fused"] = fuse_scores(m["text_norm"], m["visual_norm"], text_weight, visual_weight)
        ranked = [m for m in moments if m["fused"] >= FUSION_MIN_SCORE]
    elif method == "rrf":
        def ranks(key):
            scored = sorted((i for i, m in enumerate(moments) if m[key] is not None),
                            key=lambda i: -moments[i][key])
            return {i: r + 1 for r, i in enumerate(scored)}
        fused = rrf_scores(ranks("text_raw"), ranks("visual_raw"), text_weight, visual_weight)
        for i, m in enumerate(moments):
            m["fused"] = fused.get(i, 0.0)
        ranked = list(moments)
    else:
        raise ValueError(f"method must be one of {METHODS}")

    ranked.sort(key=lambda m: (-m["fused"], -max(m["text_norm"] or 0, m["visual_norm"] or 0), m["start"]))
    return {"method": method, "results": [_to_result(m, text_store, visual_store) for m in ranked[:top_k]]}


def _to_result(m: dict, text_store, visual_store) -> dict:
    chunk = text_store.metadata[m["chunk_row"]] if m["chunk_row"] is not None else None
    frame = visual_store.metadata[m["best_frame_row"]] if m["best_frame_row"] is not None else None
    matched = [name for name, norm in (("text", m["text_norm"]), ("visual", m["visual_norm"]))
               if norm is not None and norm >= FUSION_EVIDENCE_THRESHOLD]
    return {
        "video_id": m["video_id"],
        # Play from the start of the spoken chunk; for visual-only moments,
        # from the frame that matched best.
        "timestamp": m["start"] if chunk or not frame else float(frame["timestamp"]),
        "start": m["start"],
        "end": m["end"],
        "score": round(m["fused"], 4),
        "text_score": None if m["text_raw"] is None else round(m["text_raw"], 4),
        "visual_score": None if m["visual_raw"] is None else round(m["visual_raw"], 4),
        "text_norm": None if m["text_norm"] is None else round(m["text_norm"], 4),
        "visual_norm": None if m["visual_norm"] is None else round(m["visual_norm"], 4),
        "matched_by": matched,
        "text": chunk["text"] if chunk else None,
        "frame_filename": (frame.get("filename") or os.path.basename(frame["path"])) if frame else None,
        "frame_timestamp": float(frame["timestamp"]) if frame else None,
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def multimodal_search(query: str, top_k: int = None, text_weight: float = None,
                      visual_weight: float = None, method: str = None) -> dict:
    """
    Searches transcripts and frames together. Either modality may be missing
    (no index yet, model can't load); the search then uses the other one and
    says so in "warnings". Raises FusionUnavailableError if neither works.
    """
    if not query or not query.strip():
        raise ValueError("Query must not be empty.")
    method = method or FUSION_METHOD
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    top_k = top_k or DEFAULT_TOP_K

    from backend.services import text_index, visual_index

    warnings = []
    text_store = visual_store = text_query = visual_query = None

    try:
        text_index.require_text_index()
        text_store = _load_store(text_index.INDEX_PATH, text_index.METADATA_PATH)
        from backend.services.text_embeddings import embed_text_query
        text_query = embed_text_query(query)
    except Exception as e:  # missing index, faiss/model unavailable...
        text_store = None
        warnings.append(f"Transcript search unavailable, using visual evidence only: {e}")

    try:
        visual_index.require_visual_index()
        visual_store = _load_store(visual_index.INDEX_PATH, visual_index.METADATA_PATH)
        from backend.services.visual_embeddings import embed_text_query_clip
        visual_query = embed_text_query_clip(query)
    except Exception as e:
        visual_store = None
        warnings.append(f"Visual search unavailable, using transcript evidence only: {e}")

    if text_store is None and visual_store is None:
        raise FusionUnavailableError(" | ".join(warnings))

    return search_stores(query, text_store, visual_store, text_query, visual_query,
                         top_k=top_k, text_weight=text_weight, visual_weight=visual_weight,
                         method=method, warnings=warnings)


def search_stores(query: str, text_store, visual_store, text_query, visual_query, top_k: int = None,
                  text_weight: float = None, visual_weight: float = None, method: str = "auto",
                  warnings: list = None) -> dict:
    """The fusion pipeline on already-loaded stores and query vectors (testable without models)."""
    tw, vw = normalize_weights(text_weight, visual_weight, text_store is not None, visual_store is not None)
    text_hits = text_store.search(text_query, FUSION_TEXT_CANDIDATES) if text_store is not None and tw > 0 else []
    visual_hits = (visual_store.search(visual_query, FUSION_VISUAL_CANDIDATES)
                   if visual_store is not None and vw > 0 else [])

    moments = build_moments(text_hits, visual_hits, text_store if tw > 0 else None,
                            visual_store if vw > 0 else None)
    ranked = rank_moments(moments, text_store if tw > 0 else None, visual_store if vw > 0 else None,
                          text_query, visual_query, tw, vw, method=method, top_k=top_k or DEFAULT_TOP_K)
    return {
        "query": query,
        "method": ranked["method"],
        "text_weight": round(tw, 4),
        "visual_weight": round(vw, 4),
        "modalities": [name for name, ok in (("text", tw > 0), ("visual", vw > 0)) if ok],
        "warnings": warnings or [],
        "results": ranked["results"],
    }
