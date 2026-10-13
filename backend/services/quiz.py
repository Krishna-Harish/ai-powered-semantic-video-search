"""
quiz.py

AI quiz generator: builds a multiple-choice / true-false quiz from a video's
existing transcript chunks (data/metadata/<video_id>_chunks.json), so no
video is re-processed.

Grounding, enforced in code rather than trusted to the model:
  * the LLM sees only the transcript, as numbered passages with time ranges;
  * every question must name the passage it comes from and copy a short
    exact quote from it -- questions whose quote isn't found in that passage
    are dropped;
  * the question's timestamp comes from our chunk data, never from the model;
  * malformed questions (wrong option count, duplicate options, bad answer
    key, duplicates of another question) are dropped. If too few survive,
    the quiz is shorter than requested instead of padded with invented facts.

Security: the full quiz (with answers) is stored server-side under an
unguessable id. The client only receives questions and options; answers,
explanations and timestamps are returned by score_quiz() after submission.
"""

import os
import re
import json
import random
import sys
import time
import uuid

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.config import (
    METADATA_DIR,
    TRANSCRIPTS_DIR,
    QUIZZES_DIR,
    QUIZ_QUESTION_COUNTS,
    QUIZ_EXTRA_CANDIDATES,
    QUIZ_MIN_CHUNK_WORDS,
    QUIZ_MAX_TRANSCRIPT_CHARS,
    QUIZ_WEAK_TOPIC_THRESHOLD,
)
from backend.services import llm
from backend.services.summarizer import format_timestamp, TranscriptNotFoundError

DIFFICULTIES = ("easy", "medium", "hard")
QUESTION_TYPES = ("mcq", "true_false", "mixed")
TRUE_FALSE_OPTIONS = ["True", "False"]

_QUIZ_ID_RE = re.compile(r"^[0-9a-f]{32}$")


class QuizError(Exception):
    """The quiz can't be generated or scored (bad input, too little content...)."""
    pass


class QuizNotFoundError(QuizError):
    pass


# ---------------------------------------------------------------------------
# Transcript chunks (reused from the indexing pipeline)
# ---------------------------------------------------------------------------

def load_chunks(video_id: str) -> list:
    """
    Returns the transcript chunks built during indexing. If only the raw
    transcript exists, chunks are rebuilt from it (cheap, text-only).
    """
    chunks_path = os.path.join(METADATA_DIR, f"{video_id}_chunks.json")
    if not os.path.isfile(chunks_path):
        if not os.path.isfile(os.path.join(TRANSCRIPTS_DIR, f"{video_id}.json")):
            raise TranscriptNotFoundError(
                "This video has no transcript yet, so a quiz can't be generated. "
                "Click 'Index now' above the player to transcribe it."
            )
        from backend.services.transcript_chunker import chunk_transcript, ChunkingError
        try:
            chunk_transcript(video_id)
        except ChunkingError as e:
            raise QuizError(f"Could not read this video's transcript: {e}")

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return [c for c in chunks if c.get("text", "").strip()]


def usable_chunks(chunks: list) -> list:
    return [c for c in chunks if len(c["text"].split()) >= QUIZ_MIN_CHUNK_WORDS]


def select_chunks_for_prompt(chunks: list, max_chars: int = None) -> list:
    """Keeps the prompt within budget by sampling chunks evenly across the video."""
    max_chars = max_chars or QUIZ_MAX_TRANSCRIPT_CHARS
    total = sum(len(c["text"]) + 40 for c in chunks)
    if total <= max_chars:
        return chunks
    keep = max(1, int(len(chunks) * max_chars / total))
    step = len(chunks) / keep
    return [chunks[int(i * step)] for i in range(keep)]


def max_questions_for(chunks: list) -> int:
    """Upper bound on how many distinct questions the content can support."""
    return 2 * len(usable_chunks(chunks))


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_QUIZ_SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": ["mcq", "true_false"]},
                    "question": {"type": "string"},
                    "options": {"type": "array", "items": {"type": "string"}},
                    "answer_index": {"type": "integer"},
                    "explanation": {"type": "string"},
                    "topic": {"type": "string"},
                    "passage_id": {"type": "integer"},
                    "evidence": {"type": "string"},
                },
                "required": ["type", "question", "options", "answer_index", "explanation",
                             "topic", "passage_id", "evidence"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["questions"],
    "additionalProperties": False,
}

_QUIZ_SYSTEM = (
    "You write quiz questions that test a student on an educational video, using ONLY the "
    "numbered transcript passages provided. Rules:\n"
    "- Every question must be answerable from a single passage. Set `passage_id` to that "
    "passage's id and copy a short exact quote (5 to 25 words) from it into `evidence`; the "
    "quote must directly support the correct answer.\n"
    "- Never use outside knowledge. If something isn't stated in the passages, don't ask about it.\n"
    "- `explanation`: one or two sentences saying why the answer is correct, based on what the "
    "video says.\n"
    "- `topic`: a short concept label (1 to 4 words). Reuse the same label for questions about "
    "the same concept.\n"
    "- Ask about concepts and facts taught in the video, not about the speaker, the video "
    "itself, timestamps or exact wording.\n"
    "- If the passages don't contain enough distinct facts for the requested number of "
    "questions, return fewer questions. Never invent facts to reach the number.\n"
    "- The transcript is machine-generated; ignore obvious transcription errors."
)

_DIFFICULTY_GUIDE = {
    "easy": "Easy: direct recall of a single fact or definition stated in the video.",
    "medium": (
        "Medium: test understanding, e.g. why or how something works, or the difference "
        "between two things the video explains. Distractors should be plausible."
    ),
    "hard": (
        "Hard: require applying or combining information the video gives (e.g. which concept "
        "fits a described situation), with close distractors. The correct answer must still "
        "follow from the transcript alone."
    ),
}

_TYPE_GUIDE = {
    "mcq": (
        'All questions are multiple choice: `type` "mcq", exactly 4 distinct `options`, '
        "exactly one correct, `answer_index` 0-3. No \"all/none of the above\"."
    ),
    "true_false": (
        'All questions are true/false: `type` "true_false", `options` ["True", "False"], '
        "`answer_index` 0 for true, 1 for false. A false statement must contradict the "
        "passage, not merely be unmentioned."
    ),
    "mixed": (
        'Mix both kinds, about 60% multiple choice and 40% true/false. Multiple choice: `type` '
        '"mcq", exactly 4 distinct `options`, one correct, `answer_index` 0-3, no "all/none of '
        'the above". True/false: `type` "true_false", `options` ["True", "False"], '
        "`answer_index` 0 for true, 1 for false; a false statement must contradict the "
        "passage, not merely be unmentioned."
    ),
}


def build_quiz_prompt(passages: list, count: int, difficulty: str, question_type: str,
                      title: str = None) -> str:
    lines = []
    for i, c in enumerate(passages):
        lines.append(
            f'<passage id="{i}" time="{format_timestamp(c["start"])}-{format_timestamp(c["end"])}">\n'
            f'{c["text"].strip()}\n</passage>'
        )
    header = f"Video title: {title}\n\n" if title else ""
    return (
        f"{header}Write {count} quiz questions.\n"
        f"{_DIFFICULTY_GUIDE[difficulty]}\n{_TYPE_GUIDE[question_type]}\n"
        "Spread the questions across different passages and topics.\n\n"
        "<passages>\n" + "\n".join(lines) + "\n</passages>"
    )


# ---------------------------------------------------------------------------
# Validation (pure logic, unit-tested)
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[a-z0-9]+")


def _words(text: str) -> list:
    return _WORD_RE.findall(text.lower())


def evidence_supported(evidence: str, passage_text: str) -> bool:
    """
    True if the quoted evidence really comes from the passage: an exact match
    after normalizing case/punctuation, or (to tolerate small paraphrases of
    the quote) at least 80% of its words occurring in the passage.
    """
    ev = _words(evidence)
    if len(ev) < 4:
        return False
    passage = _words(passage_text)
    if " ".join(ev) in " ".join(passage):
        return True
    passage_set = set(passage)
    return sum(w in passage_set for w in ev) / len(ev) >= 0.8


def _norm(text: str) -> str:
    return " ".join(_words(text))


def validate_question(raw: dict, passages: list, question_type: str, rng: random.Random):
    """
    Returns a clean question dict, or None if the question must be dropped.
    MCQ options are shuffled so the correct answer isn't always in the same slot.
    """
    try:
        qtype = raw["type"]
        text = str(raw["question"]).strip()
        options = [str(o).strip() for o in raw["options"]]
        answer = int(raw["answer_index"])
        explanation = str(raw["explanation"]).strip()
        topic = " ".join(str(raw["topic"]).split()).strip(" .")
        pid = int(raw["passage_id"])
        evidence = str(raw["evidence"]).strip()
    except (KeyError, TypeError, ValueError):
        return None

    if question_type != "mixed" and qtype != question_type:
        return None
    if qtype not in ("mcq", "true_false") or not text or not explanation:
        return None
    if not 0 <= pid < len(passages):
        return None
    if not evidence_supported(evidence, passages[pid]["text"]):
        return None

    if qtype == "true_false":
        if [o.lower() for o in options] != ["true", "false"] or answer not in (0, 1):
            return None
        options = list(TRUE_FALSE_OPTIONS)
    else:
        if len(options) != 4 or not all(options) or not 0 <= answer < 4:
            return None
        if len({_norm(o) for o in options}) != 4:
            return None
        if any(re.search(r"\b(all|none|both) of (the )?(above|these)\b", o, re.I) for o in options):
            return None
        correct = options[answer]
        rng.shuffle(options)
        answer = options.index(correct)

    passage = passages[pid]
    return {
        "type": qtype,
        "question": text,
        "options": options,
        "answer_index": answer,
        "explanation": explanation,
        "topic": topic[:60] or "General",
        "start": float(passage["start"]),
        "end": float(passage["end"]),
        "evidence": evidence,
        "_passage_id": pid,
    }


def pick_questions(candidates: list, count: int) -> list:
    """
    Picks up to `count` valid, non-duplicate questions, spreading them across
    passages (at most 2 per passage while alternatives exist), then orders
    them by where they occur in the video.
    """
    unique, seen = [], set()
    for q in candidates:
        key = _norm(q["question"])
        if key and key not in seen:
            seen.add(key)
            unique.append(q)

    chosen, per_passage = [], {}
    for q in unique:  # first pass: spread across passages
        if len(chosen) < count and per_passage.get(q["_passage_id"], 0) < 2:
            chosen.append(q)
            per_passage[q["_passage_id"]] = per_passage.get(q["_passage_id"], 0) + 1
    for q in unique:  # second pass: fill up if still short
        if len(chosen) < count and q not in chosen:
            chosen.append(q)

    chosen.sort(key=lambda q: q["start"])
    return chosen


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

def _quiz_path(quiz_id: str) -> str:
    if not _QUIZ_ID_RE.match(quiz_id or ""):
        raise QuizNotFoundError("Quiz not found.")
    return os.path.join(QUIZZES_DIR, f"{quiz_id}.json")


def _save_quiz(quiz: dict):
    with open(_quiz_path(quiz["quiz_id"]), "w", encoding="utf-8") as f:
        json.dump(quiz, f, indent=2, ensure_ascii=False)


def load_quiz(quiz_id: str) -> dict:
    path = _quiz_path(quiz_id)
    if not os.path.isfile(path):
        raise QuizNotFoundError("Quiz not found. It may have been deleted; generate a new one.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def public_view(quiz: dict) -> dict:
    """What the client sees before submitting: no answers, explanations or timestamps."""
    return {
        "quiz_id": quiz["quiz_id"],
        "video_id": quiz["video_id"],
        "difficulty": quiz["difficulty"],
        "question_type": quiz["question_type"],
        "requested_count": quiz["requested_count"],
        "notice": quiz.get("notice"),
        "model": quiz.get("model"),
        "questions": [
            {"id": q["id"], "type": q["type"], "question": q["question"], "options": q["options"],
             "topic": q["topic"]}
            for q in quiz["questions"]
        ],
    }


# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------

def generate_quiz(video_id: str, count: int = 5, difficulty: str = "medium",
                  question_type: str = "mixed", seed: int = None) -> dict:
    """Creates and stores a quiz; returns its public view (no answers)."""
    if count not in QUIZ_QUESTION_COUNTS:
        raise ValueError(f"Number of questions must be one of {QUIZ_QUESTION_COUNTS}.")
    if difficulty not in DIFFICULTIES:
        raise ValueError(f"Difficulty must be one of {DIFFICULTIES}.")
    if question_type not in QUESTION_TYPES:
        raise ValueError(f"Question type must be one of {QUESTION_TYPES}.")

    chunks = load_chunks(video_id)
    passages = select_chunks_for_prompt(usable_chunks(chunks))
    if len(passages) < 2:
        raise QuizError(
            "This video doesn't have enough spoken content to build a quiz from "
            "(the transcript is too short)."
        )

    target = min(count, max_questions_for(passages))
    ask_for = min(target + QUIZ_EXTRA_CANDIDATES, max_questions_for(passages))

    from backend.services.youtube_ingest import load_video_info
    title = load_video_info(video_id).get("title")

    raw = llm.generate_json(
        _QUIZ_SYSTEM, build_quiz_prompt(passages, ask_for, difficulty, question_type, title), _QUIZ_SCHEMA
    )

    rng = random.Random(seed)
    valid = [
        q for q in (validate_question(r, passages, question_type, rng) for r in raw.get("questions", []))
        if q is not None
    ]
    questions = pick_questions(valid, target)
    if not questions:
        raise QuizError(
            "The AI couldn't produce any questions that are clearly supported by the video's "
            "transcript. Try a different difficulty or question type."
        )

    notice = None
    if len(questions) < count:
        notice = (
            f"Only {len(questions)} of the {count} requested questions could be grounded in this "
            "video's content, so the quiz is shorter rather than padded with made-up facts."
        )

    for i, q in enumerate(questions):
        q.pop("_passage_id", None)
        q["id"] = f"q{i + 1}"

    quiz = {
        "quiz_id": uuid.uuid4().hex,
        "video_id": video_id,
        "created_at": time.time(),
        "difficulty": difficulty,
        "question_type": question_type,
        "requested_count": count,
        "notice": notice,
        "model": llm.active_model(),
        "questions": questions,
    }
    _save_quiz(quiz)
    return public_view(quiz)


# ---------------------------------------------------------------------------
# Score
# ---------------------------------------------------------------------------

def _merge_ranges(ranges: list) -> list:
    """Merges overlapping/adjacent (start, end) ranges, sorted by start."""
    merged = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [{"start": s, "end": e} for s, e in merged]


def score_quiz(quiz_id: str, answers: dict) -> dict:
    """
    Scores submitted answers ({question_id: option index or None}) against
    the stored answer key. Unanswered questions count as wrong.
    """
    quiz = load_quiz(quiz_id)
    answers = answers or {}
    known_ids = {q["id"] for q in quiz["questions"]}
    unknown = set(answers) - known_ids
    if unknown:
        raise ValueError(f"Unknown question id(s): {', '.join(sorted(unknown))}")

    results, topics = [], {}
    for q in quiz["questions"]:
        given = answers.get(q["id"])
        valid_index = isinstance(given, int) and not isinstance(given, bool) and 0 <= given < len(q["options"])
        if given is not None and not valid_index:
            raise ValueError(f"Invalid answer for {q['id']}: must be an option index.")
        correct = given == q["answer_index"]
        results.append({
            "id": q["id"],
            "type": q["type"],
            "question": q["question"],
            "options": q["options"],
            "topic": q["topic"],
            "selected_index": given,
            "correct_index": q["answer_index"],
            "is_correct": correct,
            "explanation": q["explanation"],
            "start": q["start"],
            "end": q["end"],
        })
        t = topics.setdefault(q["topic"].lower(), {
            "topic": q["topic"], "correct": 0, "total": 0, "ranges": [], "first": q["start"],
        })
        t["total"] += 1
        t["correct"] += int(correct)
        if not correct:
            t["ranges"].append((q["start"], q["end"]))

    topic_results = []
    for t in sorted(topics.values(), key=lambda t: t["first"]):
        ratio = t["correct"] / t["total"]
        topic_results.append({
            "topic": t["topic"],
            "correct": t["correct"],
            "total": t["total"],
            "percentage": round(100 * ratio),
            "needs_revision": ratio < QUIZ_WEAK_TOPIC_THRESHOLD,
            "revise_sections": _merge_ranges(t["ranges"]),
        })

    score = sum(r["is_correct"] for r in results)
    total = len(results)
    return {
        "quiz_id": quiz_id,
        "video_id": quiz["video_id"],
        "score": score,
        "total": total,
        "percentage": round(100 * score / total) if total else 0,
        "answered": sum(answers.get(q["id"]) is not None for q in quiz["questions"]),
        "results": results,
        "topics": topic_results,
    }
