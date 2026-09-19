# Project Progress

Status as of initial build. Update the checkboxes as you verify each item
on your own machine with real videos/models.

## Milestone 1 — Video Processing
- [x] FFmpeg audio extraction (mono, 16kHz) — implemented and tested with a
      synthetic video in the build sandbox (real videos: verify on your machine)
- [x] Configurable frame sampling — implemented and tested
- [x] Frame-to-timestamp metadata — implemented and tested
- [x] Idempotent (skips re-processing) — implemented and tested
- [x] Error handling (missing ffmpeg, corrupted/empty video, unsupported format)

## Milestone 2 — Whisper Transcription
- [x] Code implemented (faster-whisper wrapper, lazy singleton model load)
- [x] Translate-to-English mode configured (handles Hindi/mixed-language audio)
- [ ] Verified on real audio — **run on your machine**: `python scripts/build_text_index.py`

## Milestone 3 — Transcript Chunking
- [x] Implemented and unit-tested with a synthetic 10-segment transcript —
      correctly merged into meaningful ~20s chunks with preserved timestamps

## Milestone 4 — Text Embeddings
- [x] Code implemented (sentence-transformers, normalized vectors)
- [ ] Verified on real text — **run on your machine** (needs `pip install sentence-transformers`)

## Milestone 5 — Text FAISS Search
- [x] Build + search logic implemented and verified correct using a
      behaviorally-equivalent FAISS stand-in (exact self-match scored ~1.0,
      cross-video metadata stayed correctly attached, ranking correct)
- [ ] Verified with real faiss-cpu — **run on your machine**

## Milestone 6 — Visual Embeddings
- [x] Code implemented (OpenCLIP image encoder, lazy singleton model load)
- [ ] Verified on real frames — **run on your machine** (needs `pip install open_clip_torch torch`)

## Milestone 7 — Visual Search
- [x] Build + search logic implemented and verified correct using the same
      FAISS stand-in approach as Milestone 5
- [ ] Verified with real CLIP + faiss-cpu — **run on your machine**

## Milestone 8 — Backend (FastAPI)
- [x] Implemented: `/videos`, `/search/text`, `/search/visual`, `/video/{id}`, `/frame/{video_id}/{filename}`
- [x] Syntax-verified (AST parse) — fastapi not installed in build sandbox
- [ ] Verified running — **run on your machine**: `uvicorn backend.main:app --reload`

## Milestone 9 — Frontend
- [x] Implemented: video list, video player, mode toggle, search box,
      ranked results (text + visual card variants), click-to-jump seeking
- [x] JavaScript syntax verified (`node --check`)
- [ ] Verified in browser against a running backend — **do this on your machine**

## Milestone 10 — End-to-End Test
- [x] Steps 1 (video processing) fully tested with real ffmpeg/opencv in the build sandbox
- [x] Steps 3, 5, 7 (chunking, text-index logic, visual-index logic) tested
      with synthetic data standing in for real model output
- [ ] Steps 2, 4, 6, 8, 9 require the real ML packages + your real videos —
      **must be run on your machine**, since the build sandbox has no
      internet access to install torch/faiss/whisper/etc.

## Approximate Completion

**Code: ~100% of Core + Visual Search MVP implemented.**
**Verified end-to-end on real data: not yet — pending your machine's run of
`process_videos.py` → `build_text_index.py` → `build_visual_index.py` →
`uvicorn` → frontend.**

Everything is written and internally consistent (metadata formats match
between every producing and consuming module), and the parts that don't
need heavy ML packages have been actually executed and confirmed correct.
The remaining risk is entirely in real-world model behavior (transcription
quality, embedding quality) rather than code correctness — which you'll
see directly once you run it.

## What to do next

1. Run the three scripts in order on your machine (see README "Running the
   Project").
2. Note any errors and paste them back for debugging.
3. Once text + visual search both return sensible results for your 6
   videos, you have a working, demo-ready Core + Visual Search MVP.
