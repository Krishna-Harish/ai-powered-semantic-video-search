"""
tests/test_quiz.py

AI quiz generator: grounding/validation of LLM output, secure scoring and
the API. The LLM is replaced by a fake, so no API key or Ollama is needed.

Run with:  python -m pytest tests/ -v
"""

import os
import sys
import json
import random
import unittest
from unittest import mock

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.config import METADATA_DIR, TRANSCRIPTS_DIR, QUIZZES_DIR
from backend.services import quiz as quiz_mod

try:
    from fastapi.testclient import TestClient
    from backend.main import app
    HAVE_FASTAPI = True
except ImportError:
    HAVE_FASTAPI = False

VIDEO_ID = "unittest_quiz_video"

CHUNKS = [
    {"video_id": VIDEO_ID, "chunk_id": 0, "start": 0.0, "end": 20.0,
     "text": "A USB port is used to connect devices like keyboards, mice and flash drives to a computer."},
    {"video_id": VIDEO_ID, "chunk_id": 1, "start": 20.0, "end": 41.0,
     "text": "HDMI carries both high definition video and audio over a single cable to a monitor or TV."},
    {"video_id": VIDEO_ID, "chunk_id": 2, "start": 41.0, "end": 60.0,
     "text": "The Ethernet port, also called RJ45, provides a wired network connection to the internet."},
    {"video_id": VIDEO_ID, "chunk_id": 3, "start": 60.0, "end": 64.0, "text": "Okay, next."},  # too short
    {"video_id": VIDEO_ID, "chunk_id": 4, "start": 64.0, "end": 85.0,
     "text": "A VGA port is an older analog video connector, and it does not carry any audio signal at all."},
]
# Passage ids the LLM sees: usable chunks only -> 0:USB 1:HDMI 2:Ethernet 3:VGA


def mcq(question, options, answer, pid, evidence, topic="Ports"):
    return {"type": "mcq", "question": question, "options": options, "answer_index": answer,
            "explanation": "Stated in the video.", "topic": topic, "passage_id": pid, "evidence": evidence}


def tf(statement, answer, pid, evidence, topic="Ports"):
    return {"type": "true_false", "question": statement, "options": ["True", "False"], "answer_index": answer,
            "explanation": "Stated in the video.", "topic": topic, "passage_id": pid, "evidence": evidence}


GOOD = [
    mcq("What is a USB port used for?", ["Connecting devices", "Cooling", "Power only", "Audio out"], 0, 0,
        "used to connect devices like keyboards, mice and flash drives", topic="USB"),
    mcq("What does HDMI carry?", ["Only audio", "Video and audio", "Only network data", "Only power"], 1, 1,
        "HDMI carries both high definition video and audio", topic="Video ports"),
    tf("The Ethernet port is also called RJ45.", 0, 2, "The Ethernet port, also called RJ45", topic="Networking"),
    tf("VGA carries audio signals.", 1, 3, "it does not carry any audio signal at all", topic="Video ports"),
]
BAD = [
    # evidence not in the passage (hallucinated fact)
    mcq("What year was USB invented?", ["1996", "2001", "1985", "2010"], 0, 0, "USB was invented in 1996 by Intel engineers"),
    # passage id out of range
    tf("Thunderbolt is fast.", 0, 9, "Thunderbolt is very fast indeed"),
    # wrong number of options
    mcq("Pick one", ["A", "B", "C"], 0, 1, "HDMI carries both high definition video and audio"),
    # duplicate options
    mcq("What does HDMI do?", ["Video", "video", "Audio", "Power"], 0, 1, "HDMI carries both high definition video and audio"),
    # "all of the above"
    mcq("HDMI carries?", ["Video", "Audio", "Both", "All of the above"], 3, 1, "HDMI carries both high definition video and audio"),
    # answer index out of range
    mcq("USB?", ["a", "b", "c", "d"], 7, 0, "used to connect devices like keyboards, mice and flash drives"),
    # duplicate of a GOOD question
    mcq("what is a usb port used for", ["Connecting devices", "Heat", "Light", "Sound"], 0, 0,
        "used to connect devices like keyboards, mice and flash drives"),
]


def write_chunks():
    with open(os.path.join(METADATA_DIR, f"{VIDEO_ID}_chunks.json"), "w") as f:
        json.dump(CHUNKS, f)


def cleanup():
    for path in [os.path.join(METADATA_DIR, f"{VIDEO_ID}_chunks.json"),
                 os.path.join(TRANSCRIPTS_DIR, f"{VIDEO_ID}.json")]:
        if os.path.isfile(path):
            os.remove(path)


class TestValidation(unittest.TestCase):
    PASSAGES = quiz_mod.usable_chunks(CHUNKS)

    def validate(self, raw, qtype="mixed"):
        return quiz_mod.validate_question(raw, self.PASSAGES, qtype, random.Random(1))

    def test_evidence_check(self):
        text = CHUNKS[1]["text"]
        self.assertTrue(quiz_mod.evidence_supported("HDMI carries both high-definition video", text))
        self.assertTrue(quiz_mod.evidence_supported("HDMI carries video and audio over a single cable", text))  # 1 word off
        self.assertFalse(quiz_mod.evidence_supported("HDMI was invented by Sony in 2002", text))
        self.assertFalse(quiz_mod.evidence_supported("HDMI", text))  # too short to prove anything

    def test_good_questions_pass_and_get_chunk_timestamps(self):
        out = [self.validate(q) for q in GOOD]
        self.assertTrue(all(out))
        self.assertEqual((out[1]["start"], out[1]["end"]), (20.0, 41.0))   # HDMI chunk
        self.assertEqual((out[3]["start"], out[3]["end"]), (64.0, 85.0))   # VGA chunk, short chunk skipped
        self.assertEqual(out[2]["options"], ["True", "False"])

    def test_bad_questions_are_dropped(self):
        for raw in BAD[:-1]:
            with self.subTest(q=raw["question"]):
                self.assertIsNone(self.validate(raw))

    def test_type_filter(self):
        self.assertIsNone(self.validate(GOOD[2], qtype="mcq"))
        self.assertIsNone(self.validate(GOOD[0], qtype="true_false"))
        self.assertIsNotNone(self.validate(GOOD[0], qtype="mcq"))

    def test_mcq_shuffle_keeps_correct_answer(self):
        for seed in range(20):
            q = quiz_mod.validate_question(GOOD[1], self.PASSAGES, "mixed", random.Random(seed))
            self.assertEqual(q["options"][q["answer_index"]], "Video and audio")
        positions = {quiz_mod.validate_question(GOOD[1], self.PASSAGES, "mixed", random.Random(s))["answer_index"]
                     for s in range(40)}
        self.assertGreater(len(positions), 1)  # answer isn't always in the same slot

    def test_pick_dedupes_spreads_and_orders_by_time(self):
        rng = random.Random(0)
        cands = [quiz_mod.validate_question(q, self.PASSAGES, "mixed", rng) for q in GOOD + [BAD[-1]]]
        picked = quiz_mod.pick_questions([c for c in cands if c], 10)
        self.assertEqual(len(picked), 4)  # duplicate removed
        self.assertEqual([p["start"] for p in picked], sorted(p["start"] for p in picked))

    def test_long_transcripts_are_sampled_across_the_video(self):
        many = [{"start": i * 20.0, "end": i * 20.0 + 20, "text": "word " * 50} for i in range(200)]
        picked = quiz_mod.select_chunks_for_prompt(many, max_chars=10_000)
        self.assertLess(len(picked), 200)
        self.assertEqual(picked[0]["start"], 0.0)
        self.assertGreater(picked[-1]["start"], 3000)  # reaches the end of the video


class TestGenerateAndScore(unittest.TestCase):
    def setUp(self):
        write_chunks()
        self.created = []

    def tearDown(self):
        cleanup()
        for qid in self.created:
            path = os.path.join(QUIZZES_DIR, f"{qid}.json")
            if os.path.isfile(path):
                os.remove(path)

    def generate(self, llm_questions, **kwargs):
        prompts = []

        def fake(system, prompt, schema):
            prompts.append(prompt)
            return {"questions": llm_questions}

        with mock.patch.object(quiz_mod.llm, "generate_json", fake):
            public = quiz_mod.generate_quiz(VIDEO_ID, seed=3, **kwargs)
        self.created.append(public["quiz_id"])
        return public, prompts

    def test_public_quiz_has_no_answers(self):
        public, prompts = self.generate(GOOD + BAD, count=5)
        dumped = json.dumps(public)
        for secret in ("answer_index", "explanation", "evidence", "start", "end"):
            self.assertNotIn(f'"{secret}"', dumped)
        self.assertEqual(len(public["questions"]), 4)        # only grounded questions survive
        self.assertIn("Only 4 of the 5", public["notice"])   # fewer, not padded
        self.assertIn('<passage id="3" time="01:04-01:25">', prompts[0])
        self.assertNotIn("Okay, next.", prompts[0])          # contentless chunk not sent

    def test_scoring_topics_and_revision(self):
        public, _ = self.generate(GOOD, count=5, question_type="mixed")
        stored = quiz_mod.load_quiz(public["quiz_id"])
        key = {q["id"]: q["answer_index"] for q in stored["questions"]}
        ids = [q["id"] for q in stored["questions"]]  # time order: USB, HDMI, Ethernet, VGA

        answers = {ids[0]: key[ids[0]], ids[1]: 1 - key[ids[1]] if key[ids[1]] < 2 else 0,
                   ids[2]: key[ids[2]]}  # ids[3] unanswered
        r = quiz_mod.score_quiz(public["quiz_id"], answers)
        self.assertEqual((r["score"], r["total"], r["percentage"], r["answered"]), (2, 4, 50, 3))
        self.assertFalse(r["results"][1]["is_correct"])
        self.assertIsNone(r["results"][3]["selected_index"])
        self.assertEqual(r["results"][1]["start"], 20.0)

        topics = {t["topic"]: t for t in r["topics"]}
        self.assertEqual((topics["Video ports"]["correct"], topics["Video ports"]["total"]), (0, 2))
        self.assertTrue(topics["Video ports"]["needs_revision"])
        self.assertEqual(topics["Video ports"]["revise_sections"],
                         [{"start": 20.0, "end": 41.0}, {"start": 64.0, "end": 85.0}])
        self.assertFalse(topics["USB"]["needs_revision"])

    def test_invalid_submissions(self):
        public, _ = self.generate(GOOD, count=5)
        qid = public["quiz_id"]
        with self.assertRaises(ValueError):
            quiz_mod.score_quiz(qid, {"q99": 0})
        with self.assertRaises(ValueError):
            quiz_mod.score_quiz(qid, {"q1": 4})
        with self.assertRaises(ValueError):
            quiz_mod.score_quiz(qid, {"q1": True})
        with self.assertRaises(quiz_mod.QuizNotFoundError):
            quiz_mod.score_quiz("../../etc/passwd", {})

    def test_nothing_grounded_is_an_error(self):
        with self.assertRaises(quiz_mod.QuizError):
            self.generate(BAD[:-1], count=5)  # last BAD item is only a duplicate, valid alone

    def test_too_little_content(self):
        with open(os.path.join(METADATA_DIR, f"{VIDEO_ID}_chunks.json"), "w") as f:
            json.dump(CHUNKS[:1] + CHUNKS[3:4], f)
        with self.assertRaises(quiz_mod.QuizError):
            self.generate(GOOD, count=5)

    def test_bad_settings(self):
        for kwargs in ({"count": 7}, {"difficulty": "extreme"}, {"question_type": "essay"}):
            with self.assertRaises(ValueError):
                quiz_mod.generate_quiz(VIDEO_ID, **{"count": 5, **kwargs})

    def test_chunks_rebuilt_from_transcript_without_reprocessing(self):
        os.remove(os.path.join(METADATA_DIR, f"{VIDEO_ID}_chunks.json"))
        with open(os.path.join(TRANSCRIPTS_DIR, f"{VIDEO_ID}.json"), "w") as f:
            json.dump({"video_id": VIDEO_ID, "segments": [
                {"start": c["start"], "end": c["end"], "text": c["text"]} for c in CHUNKS]}, f)
        self.assertGreater(len(quiz_mod.load_chunks(VIDEO_ID)), 0)

    def test_no_transcript(self):
        cleanup()
        with self.assertRaises(quiz_mod.TranscriptNotFoundError):
            quiz_mod.load_chunks(VIDEO_ID)


@unittest.skipUnless(HAVE_FASTAPI, "fastapi not installed")
class TestQuizApi(unittest.TestCase):
    def setUp(self):
        write_chunks()
        self.client = TestClient(app)

    def tearDown(self):
        cleanup()

    def test_generate_submit_flow(self):
        with mock.patch.object(quiz_mod.llm, "generate_json", return_value={"questions": GOOD}):
            res = self.client.post("/quiz/generate", json={
                "video_id": VIDEO_ID, "num_questions": 5, "difficulty": "easy", "question_type": "mixed"})
        self.assertEqual(res.status_code, 200, res.text)
        public = res.json()
        self.assertNotIn("answer_index", res.text)
        qid = public["quiz_id"]
        try:
            again = self.client.get(f"/quiz/{qid}")
            self.assertEqual(again.json()["questions"], public["questions"])
            self.assertNotIn("explanation", again.text)

            first = public["questions"][0]["id"]
            res = self.client.post(f"/quiz/{qid}/submit", json={"answers": {first: 0}})
            self.assertEqual(res.status_code, 200, res.text)
            body = res.json()
            self.assertEqual(body["total"], 4)
            self.assertIn("explanation", body["results"][0])
            self.assertIn("start", body["results"][0])

            self.assertEqual(self.client.post(f"/quiz/{qid}/submit", json={"answers": {first: True}}).status_code, 422)
            self.assertEqual(self.client.post(f"/quiz/{qid}/submit", json={"answers": {first: 9}}).status_code, 400)
        finally:
            os.remove(os.path.join(QUIZZES_DIR, f"{qid}.json"))

    def test_errors(self):
        self.assertEqual(self.client.post("/quiz/generate", json={"video_id": VIDEO_ID, "num_questions": 7}).status_code, 422)
        self.assertEqual(self.client.get("/quiz/" + "0" * 32).status_code, 404)
        self.assertEqual(self.client.post("/quiz/generate", json={"video_id": "no_such_video"}).status_code, 404)
        with mock.patch.object(quiz_mod.llm, "generate_json",
                               side_effect=quiz_mod.llm.LLMUnavailableError("Ollama is not running")):
            res = self.client.post("/quiz/generate", json={"video_id": VIDEO_ID})
        self.assertEqual(res.status_code, 503)
        self.assertIn("Ollama", res.json()["detail"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
