# AI-Powered Semantic Search in Video

A system that lets you search *inside* a collection of videos using natural
language, and jump directly to the relevant moment — instead of scrubbing
through long videos manually.

## Problem Statement

Huge amounts of information live inside videos (lectures, tutorials,
meetings, seminars), but finding a specific moment inside a long video is
slow and manual. Traditional video search finds *which video* is about a
topic. This project finds *where inside* a video that topic is discussed —
by what is said, and by what is visually shown.

## Motivation & Relevance

- Educational platforms: jump straight to the part of a lecture that
  explains a specific concept.
- Training/onboarding videos: find the exact demo or explanation needed.
- Meeting/seminar archives: locate when a topic was discussed without
  re-watching the whole recording.

Traditional search: *"find videos about machine learning."*
This system: *"find WHERE machine learning is discussed inside these videos"*
— and even *"find WHERE a diagram is shown"*, without needing spoken words
to match.

## Objectives

1. Transcribe spoken content with timestamps (text search).
2. Retrieve visually relevant frames even when the speaker never says the
   matching words (visual search).
3. Let a user query in natural language and jump directly to the matching
   timestamp in a video player.

## System Architecture

```
                    VIDEO COLLECTION
                          |
              +-----------+-----------+
              |                       |
            AUDIO                   FRAMES
              |                       |
           Whisper                  CLIP (image encoder)
              |                       |
    Timestamped transcript     Visual embeddings (FAISS)
              |
    Transcript chunking
              |
    Sentence Transformer
              |
    Text embeddings (FAISS)

                    USER QUERY
                          |
          +---------------+---------------+
          |                               |
      TEXT MODE                      VISUAL MODE
          |                               |
  Sentence Transformer              CLIP (text encoder)
          |                               |
     Text FAISS search             Visual FAISS search
          |                               |
          +---------------+---------------+
                          |
                Ranked results (video + timestamp)
                          |
                    Video player seeks to timestamp
```

Text and visual retrieval are kept as **separate** indexes/pipelines in this
version. Combining their scores (multimodal fusion) is a future extension,
not implemented here.

## Technologies

| Component | Tool |
|---|---|
| Video/audio processing | FFmpeg |
| Speech recognition | faster-whisper |
| Text embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Visual embeddings | OpenCLIP (ViT-B-32) |
| Vector similarity search | FAISS |
| Backend API | FastAPI |
| YouTube download | yt-dlp |
| Summaries / grounded Q&A | Gemini API, free tier (`google-genai`), or Claude API (`anthropic`) |
| Frontend | HTML / CSS / vanilla JS |

## AI Models Used, and Why

**All models below are PRETRAINED. None are trained or fine-tuned in this
project.** The 6 videos are our *searchable corpus*, not a training dataset —
they are never used to update any model's weights.

- **Whisper (faster-whisper)** — converts speech to timestamped text.
  Chosen because it's free, runs locally, works well on noisy/accented
  audio, and supports translation to English from other spoken languages
  (useful since our corpus includes Hindi/Hindi-English lecture content).
- **Sentence Transformer (all-MiniLM-L6-v2)** — converts a sentence into a
  384-dimension vector capturing its *meaning*. Chosen for being small,
  fast on CPU, and well-validated for semantic search tasks.
- **CLIP (OpenCLIP, ViT-B-32)** — converts images AND text into the *same*
  512-dimension vector space, which is what makes "search a diagram by
  describing it in words" possible. It's the only practical free way to do
  text-to-image semantic matching.
- **FAISS** — NOT a machine learning model. It's a vector similarity search
  library: given a query vector, it finds the closest stored vectors,
  extremely fast, even across thousands of vectors.

## Dataset / Video Corpus

5–6 short (5–15 min) educational/instructional videos (ML, DL, CV, AI,
Data Science, Networks topics), placed in `data/videos/`. Chosen to contain
both spoken explanation and visible slides/diagrams, so both search modes
have real content to retrieve.

## Installation

### Prerequisites
- Python 3.10 – 3.14
- FFmpeg installed and on PATH (`ffmpeg -version` should work)

### Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows (PowerShell: venv\Scripts\Activate.ps1)
# source venv/bin/activate   # Mac/Linux

python -m pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` lists minimum versions rather than exact pins, so pip
picks builds that exist for your Python version (exact old pins such as
`torch==2.4.1` have no builds for Python 3.13+).

First run of transcription/embedding steps will download pretrained model
weights automatically (requires internet the first time only; cached
afterwards).

## Running the Project

**1. Process videos** (audio + frame extraction):
```bash
python scripts/process_videos.py
```

**2. Build the text search index** (transcribe -> chunk -> embed -> index):
```bash
python scripts/build_text_index.py
```

**3. Build the visual search index** (embed frames -> index):
```bash
python scripts/build_visual_index.py
```

**4. Start the backend:**
```bash
uvicorn backend.main:app --reload
```
API docs available at `http://127.0.0.1:8000/docs`.

**5. Open the frontend:**
Go to `http://127.0.0.1:8000/app/` (the backend serves the frontend). You can
also open `frontend/index.html` directly in a browser; it then talks to the
API at `http://127.0.0.1:8000`.

## Adding a Video from YouTube

Paste a YouTube URL into the box at the top of the page and click
**Add video**. The backend then, as a background job:

1. downloads the video with **yt-dlp** (max 720p, saved as
   `data/videos/yt_<youtube_id>.mp4`, title saved to
   `data/metadata/yt_<youtube_id>_info.json`),
2. extracts audio and frames with FFmpeg,
3. transcribes with Whisper and builds text embeddings,
4. builds CLIP frame embeddings,
5. rebuilds both FAISS indexes.

The page shows each job's current stage and the new video appears in the
video list when it is done. Jobs run one at a time and are kept in memory,
so a backend restart forgets the job list (downloaded/indexed videos stay).
If a video has no speech, it is still indexed visually (and vice versa);
the job shows a warning.

From the command line instead:
```bash
python scripts/ingest_youtube.py https://www.youtube.com/watch?v=VIDEO_ID
```

Limits are set in `backend/config.py` (`YOUTUBE_MAX_HEIGHT`,
`YOUTUBE_MAX_DURATION_SECONDS`, default 2 hours). Live streams and playlists
are not accepted; only download videos you have the right to use.

## Jumping to the Exact Moment

Every search result, summary point and answer citation is a link to its
moment. Clicking it opens the right video in the player and starts playback
at that timestamp. Links have the form `/app/#v=<video_id>&t=<seconds>`,
so they can be copied, bookmarked or opened in a new tab, and
**Copy link to this moment** under the player shares the current position.

The `/video/{video_id}` endpoint answers HTTP Range requests, which lets the
browser seek straight to the timestamp without downloading the whole file
first (this is why `fastapi` is pinned to 0.115.6 or later).

## AI Summaries and Questions (Gemini free tier or Claude)

Needs one API key. The free option is Google Gemini:

1. Create a free key at https://aistudio.google.com/apikey (no billing needed).
2. Set it before starting the backend:
   ```bash
   export GEMINI_API_KEY=your-key      # Windows (cmd): set GEMINI_API_KEY=your-key
   ```

If you also have a paid Anthropic key (`ANTHROPIC_API_KEY`), Claude is used
instead; without it, Gemini is used automatically. To force one, set
`LLM_PROVIDER=gemini` or `LLM_PROVIDER=anthropic`. The Gemini model defaults
to `gemini-flash-latest` (free tier); change it with `GEMINI_MODEL`.

The free tier limits requests per minute and per day. If you hit the limit
the app says so; wait a minute and retry. Whole-video summaries are cached,
so asking again for the same video doesn't use quota.

Under the player, for the selected video:
- **Summarize video**: short summary plus the main topics with timestamps.
- **Key takeaways**: the 3–8 most important points with timestamps.
- **Summarize section**: summary of a time range (type `mm:ss` times or
  press **now** to take the player's current time).

In the search panel, **Ask AI** answers a question using only transcript
passages found by text search (optionally only from the selected video).
Each claim in the answer links to the passage it came from.

How the answers stay grounded:
- The model only sees transcript text, with a timestamp on every line.
- Summary timestamps are snapped to a real transcript segment.
- Answer citations refer to the numbered passages that were retrieved, and
  the timestamps come from those passages, not from the model.
- If the passages don't cover the question, the answer says so.

Whole-video summaries/takeaways are cached in `data/metadata/` (pass
`"force": true` to `/summarize` to regenerate). Very long transcripts are
summarized in parts and then combined. Provider and models are set in
`backend/config.py` (`LLM_PROVIDER`, `GEMINI_MODEL`, `LLM_MODEL`).

API endpoints:

| Endpoint | Purpose |
|---|---|
| `POST /ingest/youtube` `{"url"}` | Queue a YouTube video for download + indexing |
| `GET /ingest/jobs`, `GET /ingest/jobs/{job_id}` | Ingestion progress |
| `POST /summarize` `{"video_id", "mode": "summary"\|"takeaways"\|"section", "start", "end"}` | LLM summary |
| `POST /ask` `{"question", "video_id"?}` | Answer grounded in transcript passages |

## How Text Search Works

1. Your query is converted into a vector using the same Sentence
   Transformer used on the transcripts.
2. FAISS compares that vector against every stored transcript-chunk vector.
3. The closest matches (by cosine similarity) are returned with their
   video ID, timestamp range, and original transcript text.

## How Visual Search Works

1. Your query is converted into a vector using CLIP's **text** encoder.
2. Every sampled frame was earlier converted into a vector using CLIP's
   **image** encoder — living in the same vector space.
3. FAISS finds the frames whose vectors are closest to your query vector,
   returned with video ID, timestamp, and a frame preview.

## Example Queries

- Text search: `"Where is gradient descent explained?"`
- Text search: `"What does the speaker say about backpropagation?"`
- Visual search: `"neural network architecture diagram"`
- Visual search: `"a graph showing loss decreasing over time"`

## Limitations

- Frame sampling (default every 5s) can miss visual content shown only very
  briefly, or in the last few seconds of a video (an FFmpeg `fps` filter
  edge case near end-of-stream).
- Text and visual search scores are not combined (no multimodal fusion yet).
- Whisper's `translate` mode standardizes all transcripts to English, which
  improves cross-video consistency but loses exact original-language
  wording.
- Small CPU-friendly models (MiniLM, ViT-B-32, Whisper `small`) trade some
  accuracy for speed — larger variants would improve quality but slow down
  processing on a laptop.
- No authentication, no cloud deployment — this is a local, single-user
  demo system by design.

## Future Improvements

- Multimodal score fusion (combine text + visual relevance).
- Reranking, query expansion, temporal context windows.
- Formal evaluation: Precision@K, Recall@K, timestamp accuracy, latency.
- Larger/better embedding models if hardware allows.
