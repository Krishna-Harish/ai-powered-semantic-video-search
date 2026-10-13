"""
summarizer.py

LLM-based summaries and grounded Q&A over the Whisper transcripts, using
Claude or (free tier) Gemini -- see llm.py.

    summarize(video_id, "summary")              -- whole-video summary
    summarize(video_id, "takeaways")            -- key takeaways
    summarize(video_id, "section", start, end)  -- summary of one time range
    answer_question(question, video_id=None)    -- answer grounded in
                                                   retrieved transcript passages

Grounding: the model only ever sees transcript text, with every line
labelled by its timestamp. Timestamps it returns are snapped to a real
transcript segment, and Q&A citations are returned as references to the
retrieved passages (never free-form timestamps), so every timestamp the
frontend shows is one that actually exists in the video.

Requires GEMINI_API_KEY (free) or ANTHROPIC_API_KEY in the environment
(see config.py LLM_* settings).
"""

import os
import re
import json
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.config import (
    TRANSCRIPTS_DIR,
    METADATA_DIR,
    LLM_TRANSCRIPT_WINDOW_CHARS,
    LLM_QA_TOP_K,
    MIN_SIMILARITY_SCORE,
)
from backend.services import llm
from backend.services.llm import LLMError as SummaryError, LLMUnavailableError

SUMMARY_MODES = ("summary", "takeaways", "section")

class TranscriptNotFoundError(SummaryError):
    pass


# ---------------------------------------------------------------------------
# Transcript helpers (pure logic, unit-tested)
# ---------------------------------------------------------------------------

def format_timestamp(seconds: float) -> str:
    """143.2 -> '02:23'; 3725 -> '1:02:05'."""
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def parse_timestamp(value) -> float:
    """Accepts seconds (int/float/'83.5') or 'mm:ss' / 'h:mm:ss'. Raises ValueError."""
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().strip("[]")
    if re.fullmatch(r"\d+(\.\d+)?", text):
        return float(text)
    parts = text.split(":")
    if not 2 <= len(parts) <= 3 or not all(re.fullmatch(r"\d+(\.\d+)?", p) for p in parts):
        raise ValueError(f"Invalid timestamp: '{value}'")
    total = 0.0
    for p in parts:
        total = total * 60 + float(p)
    return total


def load_segments(video_id: str) -> list:
    path = os.path.join(TRANSCRIPTS_DIR, f"{video_id}.json")
    if not os.path.isfile(path):
        raise TranscriptNotFoundError(
            f"No transcript for '{video_id}'. Index the video first (scripts/build_text_index.py "
            "or the YouTube ingestion)."
        )
    with open(path, "r", encoding="utf-8") as f:
        segments = json.load(f).get("segments", [])
    segments = [s for s in segments if s.get("text", "").strip()]
    if not segments:
        raise TranscriptNotFoundError(f"Transcript for '{video_id}' is empty.")
    return segments


def select_segments(segments: list, start: float = None, end: float = None) -> list:
    """Segments overlapping [start, end] (either bound may be None)."""
    lo = float("-inf") if start is None else start
    hi = float("inf") if end is None else end
    return [s for s in segments if s["end"] > lo and s["start"] < hi]


def format_transcript(segments: list) -> str:
    return "\n".join(f"[{format_timestamp(s['start'])}] {s['text'].strip()}" for s in segments)


def split_into_windows(segments: list, max_chars: int) -> list:
    """Splits segments into consecutive groups whose formatted text fits max_chars."""
    windows, current, size = [], [], 0
    for seg in segments:
        line_len = len(seg["text"]) + 12
        if current and size + line_len > max_chars:
            windows.append(current)
            current, size = [], 0
        current.append(seg)
        size += line_len
    if current:
        windows.append(current)
    return windows


def snap_to_segment(timestamp: float, segments: list) -> float:
    """Returns the start of the segment containing (or nearest to) `timestamp`."""
    best = None
    for seg in segments:
        if seg["start"] <= timestamp < seg["end"]:
            return float(seg["start"])
        distance = min(abs(seg["start"] - timestamp), abs(seg["end"] - timestamp))
        if best is None or distance < best[0]:
            best = (distance, float(seg["start"]))
    return best[1]


# ---------------------------------------------------------------------------
# LLM call (Claude or Gemini, see llm.py)
# ---------------------------------------------------------------------------

def _call_llm(system: str, prompt: str, schema: dict) -> dict:
    """Single LLM request that returns JSON validated against `schema`."""
    return llm.generate_json(system, prompt, schema)


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

_SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "points": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "timestamp": {"type": "string"},
                },
                "required": ["text", "timestamp"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["summary", "points"],
    "additionalProperties": False,
}

_SUMMARY_SYSTEM = (
    "You summarize video transcripts for a video search app. Each transcript line starts with "
    "its [mm:ss] timestamp. Use only what the transcript says; never add outside facts. "
    "The transcript is machine-generated (Whisper), so silently fix obvious transcription "
    "errors but don't invent content. For every point, set `timestamp` to the [mm:ss] of the "
    "transcript line where that point is made, copied exactly from the transcript."
)

_MODE_INSTRUCTIONS = {
    "summary": (
        "Write a concise summary of the whole video (one or two short paragraphs) in `summary`. "
        "In `points`, list the main topics in the order they appear, one sentence each, with the "
        "timestamp where each topic starts."
    ),
    "takeaways": (
        "Write a one or two sentence overview in `summary`. In `points`, list the most important "
        "key takeaways (3 to 8): what a viewer should remember or be able to do afterwards, one "
        "sentence each, with the timestamp where it is explained."
    ),
    "section": (
        "This is one section of a longer video. Write a concise summary of just this section in "
        "`summary`. In `points`, list its key points in order, one sentence each, with timestamps."
    ),
}


def _cache_path(video_id: str, mode: str) -> str:
    return os.path.join(METADATA_DIR, f"{video_id}_summary_{mode}.json")


def _summarize_segments(segments: list, mode: str, title: str = None) -> dict:
    header = f"Video title: {title}\n\n" if title else ""
    windows = split_into_windows(segments, LLM_TRANSCRIPT_WINDOW_CHARS)

    if len(windows) == 1:
        prompt = f"{header}{_MODE_INSTRUCTIONS[mode]}\n\n<transcript>\n{format_transcript(segments)}\n</transcript>"
        return _call_llm(_SUMMARY_SYSTEM, prompt, _SUMMARY_SCHEMA)

    # Transcript too long for one request: summarize each window as a
    # section, then combine the partial summaries (map-reduce).
    partials = []
    for window in windows:
        prompt = (
            f"{header}{_MODE_INSTRUCTIONS['section']}\n\n"
            f"<transcript>\n{format_transcript(window)}\n</transcript>"
        )
        partials.append(_call_llm(_SUMMARY_SYSTEM, prompt, _SUMMARY_SCHEMA))

    notes = []
    for i, part in enumerate(partials, 1):
        notes.append(f"Part {i} summary: {part['summary']}")
        notes.extend(f"[{p['timestamp']}] {p['text']}" for p in part["points"])
    prompt = (
        f"{header}The transcript was too long to read at once, so it was summarized in "
        f"{len(partials)} consecutive parts. Combine these part summaries into one result for the "
        f"whole video. {_MODE_INSTRUCTIONS[mode]} Only use timestamps that appear below.\n\n"
        "<part_summaries>\n" + "\n".join(notes) + "\n</part_summaries>"
    )
    return _call_llm(_SUMMARY_SYSTEM, prompt, _SUMMARY_SCHEMA)


def _clean_points(raw_points: list, segments: list) -> list:
    """Parses model timestamps and snaps each one to a real transcript segment."""
    points = []
    for p in raw_points:
        text = (p.get("text") or "").strip()
        if not text:
            continue
        try:
            ts = snap_to_segment(parse_timestamp(p.get("timestamp", "")), segments)
        except ValueError:
            ts = None
        points.append({"text": text, "timestamp": ts})
    return points


def summarize(video_id: str, mode: str = "summary", start: float = None, end: float = None,
              force: bool = False) -> dict:
    """
    Returns {"video_id", "mode", "start", "end", "summary",
             "points": [{"text", "timestamp"}], "model", "cached"}.
    Whole-video summaries/takeaways are cached in data/metadata/ (force=True regenerates).
    """
    if mode not in SUMMARY_MODES:
        raise ValueError(f"mode must be one of {SUMMARY_MODES}")

    segments = load_segments(video_id)

    if mode == "section":
        if start is None or end is None:
            raise ValueError("Section summaries need both start and end.")
        if end <= start:
            raise ValueError("end must be after start.")
        segments = select_segments(segments, start, end)
        if not segments:
            raise ValueError(
                f"No speech between {format_timestamp(start)} and {format_timestamp(end)}."
            )
    else:
        start = end = None
        cache = _cache_path(video_id, mode)
        if os.path.isfile(cache) and not force:
            with open(cache, "r", encoding="utf-8") as f:
                return {**json.load(f), "cached": True}

    from backend.services.youtube_ingest import load_video_info
    title = load_video_info(video_id).get("title")

    raw = _summarize_segments(segments, mode, title=title)
    result = {
        "video_id": video_id,
        "mode": mode,
        "start": start,
        "end": end,
        "summary": raw.get("summary", "").strip(),
        "points": _clean_points(raw.get("points", []), segments),
        "model": llm.active_model(),
    }

    if mode != "section":
        with open(_cache_path(video_id, mode), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    return {**result, "cached": False}


# ---------------------------------------------------------------------------
# Grounded Q&A
# ---------------------------------------------------------------------------

_ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answerable": {"type": "boolean"},
        "answer": {"type": "string"},
        "citations": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["answerable", "answer", "citations"],
    "additionalProperties": False,
}

_ANSWER_SYSTEM = (
    "You answer questions about a collection of videos using ONLY the numbered transcript "
    "passages provided. Each passage gives its video and time range. If the passages don't "
    "contain the answer, set `answerable` to false and say briefly what is missing; do not use "
    "outside knowledge. Otherwise answer concisely, mark each claim with the passage number in "
    "square brackets like [2], and list every passage number you relied on in `citations`. "
    "The passages are machine transcripts, so tolerate small transcription errors."
)


def retrieve_passages(question: str, video_id: str = None, top_k: int = None) -> list:
    """Semantic search over transcript chunks, optionally restricted to one video."""
    from backend.services.text_embeddings import embed_text_query
    from backend.services.text_index import search_text_index

    top_k = top_k or LLM_QA_TOP_K
    # When filtering to one video, over-fetch so enough of its chunks survive the filter.
    fetch_k = top_k * 10 if video_id else top_k
    results = search_text_index(embed_text_query(question), top_k=fetch_k)
    results = [r for r in results if r["score"] >= MIN_SIMILARITY_SCORE]
    if video_id:
        results = [r for r in results if r["video_id"] == video_id]
    return results[:top_k]


def build_answer_prompt(question: str, passages: list) -> str:
    lines = []
    for i, p in enumerate(passages, 1):
        lines.append(
            f'<passage id="{i}" video="{p["video_id"]}" '
            f'time="{format_timestamp(p["start"])}-{format_timestamp(p["end"])}">\n'
            f'{p["text"].strip()}\n</passage>'
        )
    return "<passages>\n" + "\n".join(lines) + f"\n</passages>\n\nQuestion: {question}"


def answer_question(question: str, video_id: str = None, top_k: int = None, passages: list = None) -> dict:
    """
    Returns {"question", "video_id", "answerable", "answer",
             "citations": [{"ref", "video_id", "start", "end", "text"}], "model"}.
    `citations[].ref` is the [n] marker used in the answer text.
    """
    if not question or not question.strip():
        raise ValueError("Question must not be empty.")

    if passages is None:
        passages = retrieve_passages(question, video_id=video_id, top_k=top_k)

    if not passages:
        return {
            "question": question,
            "video_id": video_id,
            "answerable": False,
            "answer": "No relevant transcript passages were found for this question.",
            "citations": [],
            "model": llm.active_model(),
        }

    raw = _call_llm(_ANSWER_SYSTEM, build_answer_prompt(question, passages), _ANSWER_SCHEMA)

    citations = []
    for ref in sorted(set(raw.get("citations", []))):
        if isinstance(ref, int) and 1 <= ref <= len(passages):
            p = passages[ref - 1]
            citations.append({
                "ref": ref,
                "video_id": p["video_id"],
                "start": float(p["start"]),
                "end": float(p["end"]),
                "text": p["text"],
            })

    return {
        "question": question,
        "video_id": video_id,
        "answerable": bool(raw.get("answerable")) and bool(citations),
        "answer": raw.get("answer", "").strip(),
        "citations": citations,
        "model": llm.active_model(),
    }
