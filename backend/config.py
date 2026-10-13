"""
Central configuration file.
All paths and tunable settings live here so nothing is hardcoded
inside individual modules. Change values here to tune the pipeline.
"""

import os

# ---------------------------------------------------------------------------
# Base project paths
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load settings such as GEMINI_API_KEY from the project's .env file (copy
# .env.example to .env), so they don't have to be typed in every terminal.
# Variables already set in the terminal take precedence over the file.
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(BASE_DIR, ".env"))
except ImportError:  # python-dotenv not installed: use terminal variables only
    pass

DATA_DIR = os.path.join(BASE_DIR, "data")
VIDEOS_DIR = os.path.join(DATA_DIR, "videos")
AUDIO_DIR = os.path.join(DATA_DIR, "audio")
FRAMES_DIR = os.path.join(DATA_DIR, "frames")
TRANSCRIPTS_DIR = os.path.join(DATA_DIR, "transcripts")
EMBEDDINGS_DIR = os.path.join(DATA_DIR, "embeddings")
INDEXES_DIR = os.path.join(DATA_DIR, "indexes")
METADATA_DIR = os.path.join(DATA_DIR, "metadata")
QUIZZES_DIR = os.path.join(DATA_DIR, "quizzes")

for folder in [VIDEOS_DIR, AUDIO_DIR, FRAMES_DIR, TRANSCRIPTS_DIR,
               EMBEDDINGS_DIR, INDEXES_DIR, METADATA_DIR, QUIZZES_DIR]:
    os.makedirs(folder, exist_ok=True)

# ---------------------------------------------------------------------------
# Video processing settings (Milestone 1)
# ---------------------------------------------------------------------------

# Extract one frame every N seconds. Smaller = more coverage, more storage.
FRAME_SAMPLE_INTERVAL_SECONDS = 5

AUDIO_SAMPLE_RATE = 16000   # Hz, required format for Whisper
AUDIO_CHANNELS = 1          # mono

SUPPORTED_VIDEO_EXTENSIONS = [".mp4", ".mkv", ".mov", ".avi", ".webm"]

# ---------------------------------------------------------------------------
# Whisper / transcription settings (Milestone 2)
# ---------------------------------------------------------------------------

# Small model chosen deliberately: good accuracy/speed trade-off on a CPU
# laptop. Options (smallest->largest): tiny, base, small, medium, large-v3
# For faster (less accurate) transcription set the WHISPER_MODEL environment
# variable, e.g. WHISPER_MODEL=base (~2-3x faster than small).
WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL", "small")

# 1 = greedy decoding: about 2x faster than the default beam search (5) on
# CPU, with only a small accuracy cost.
WHISPER_BEAM_SIZE = 1

# CPU threads for Whisper. 0 lets the library pick (often only 4 threads);
# we use every core the machine has.
WHISPER_CPU_THREADS = os.cpu_count() or 0

# "transcribe" keeps original spoken language.
# "translate" converts any spoken language directly to English text.
# We default to "translate" because our text-embedding model
# (all-MiniLM-L6-v2) is English-optimized, and our videos may include
# Hindi/Hindi-English content. Standardizing to English keeps semantic
# search quality consistent across all videos regardless of source language.
WHISPER_TASK = "translate"

# "cpu" is the safe default for a student laptop with no dedicated GPU.
# Set to "cuda" manually if you have a working NVIDIA GPU + CUDA installed.
WHISPER_DEVICE = "cpu"

# int8 quantization keeps faster-whisper fast and light on CPU.
WHISPER_COMPUTE_TYPE = "int8"

# ---------------------------------------------------------------------------
# Transcript chunking settings (Milestone 3)
# ---------------------------------------------------------------------------

# Chunks are built by merging consecutive Whisper segments until this
# many seconds of speech have accumulated, preserving start/end times.
CHUNK_TARGET_DURATION_SECONDS = 20
CHUNK_MAX_DURATION_SECONDS = 35

# ---------------------------------------------------------------------------
# Text embedding settings (Milestone 4)
# ---------------------------------------------------------------------------

TEXT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TEXT_EMBEDDING_DIM = 384  # output dimension of all-MiniLM-L6-v2

# ---------------------------------------------------------------------------
# Visual embedding settings (Milestone 6)
# ---------------------------------------------------------------------------

# OpenCLIP model + pretrained weight tag. ViT-B-32 is the standard
# lightweight CLIP variant, fine for CPU inference on sampled frames.
CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "openai"
CLIP_EMBEDDING_DIM = 512  # output dimension of ViT-B-32

# ---------------------------------------------------------------------------
# Search settings
# ---------------------------------------------------------------------------

DEFAULT_TOP_K = 5
MIN_SIMILARITY_SCORE = 0.15  # results below this score are filtered out

# ---------------------------------------------------------------------------
# Multimodal search (score fusion of transcript + visual evidence)
# ---------------------------------------------------------------------------

# Relative importance of what is SAID vs what is SHOWN. Normalized to sum to 1;
# can be overridden per request. Speech carries most of the content in lectures.
FUSION_TEXT_WEIGHT = 0.6
FUSION_VISUAL_WEIGHT = 0.4

# "auto": calibrated score fusion, falling back to rank fusion (RRF) when a
# modality's raw scores don't look like cosine similarities.
# "calibrated" / "rrf": force one method.
FUSION_METHOD = "auto"

# Raw cosine similarities of the two models live on different scales, so each
# is mapped to 0..1 between a "floor" (typical score of an unrelated pair -> 0)
# and a "ceiling" (a clearly strong match -> 1). Approximate values for the
# default models; tune them on your own videos if needed.
TEXT_SCORE_FLOOR = 0.15     # all-MiniLM-L6-v2, query vs transcript chunk
TEXT_SCORE_CEILING = 0.65
VISUAL_SCORE_FLOOR = 0.18   # CLIP ViT-B-32, query text vs frame
VISUAL_SCORE_CEILING = 0.30

# How many transcript chunks / frames each index contributes as candidates
# before every candidate moment is re-scored on both modalities.
FUSION_TEXT_CANDIDATES = 50
FUSION_VISUAL_CANDIDATES = 100

FUSION_RRF_K = 60              # standard reciprocal-rank-fusion constant
FUSION_MIN_SCORE = 0.05        # calibrated results below this fused score are dropped
FUSION_EVIDENCE_THRESHOLD = 0.2  # normalized score at which a modality counts as "matched"

# ---------------------------------------------------------------------------
# YouTube ingestion settings
# ---------------------------------------------------------------------------

# Videos ingested from a YouTube URL are stored as data/videos/yt_<youtube_id>.mp4
YOUTUBE_VIDEO_ID_PREFIX = "yt_"

# Cap the download resolution: frames are only used for CLIP (224px input),
# so higher resolutions just cost disk space and download time.
YOUTUBE_MAX_HEIGHT = 720

# Refuse videos longer than this (seconds). Whisper on CPU is roughly
# real-time with the "small" model, so very long videos take a long time.
# Set to None to disable the limit.
YOUTUBE_MAX_DURATION_SECONDS = 2 * 60 * 60

# ---------------------------------------------------------------------------
# LLM summary / Q&A settings
# ---------------------------------------------------------------------------

# Which LLM generates summaries, answers and quizzes:
#   "auto"   -- Gemini if GEMINI_API_KEY (or GOOGLE_API_KEY) is set,
#               otherwise a local Ollama model
#   "gemini" -- always Gemini (free key: https://aistudio.google.com/apikey)
#   "ollama" -- always the local Ollama model (free, offline, no key)
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "auto")

# Ollama (https://ollama.com): run `ollama pull llama3.2` once, then keep the
# Ollama app running. Bigger models (e.g. llama3.1:8b, qwen2.5:7b) give better
# quizzes/summaries but need more RAM.
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")
OLLAMA_CONTEXT_TOKENS = 16384   # Ollama's default context is too small for a transcript
OLLAMA_TIMEOUT_SECONDS = 600    # local models on CPU can be slow

# Gemini model. "gemini-flash-latest" always points at Google's current Flash
# model, which is available on the free tier.
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")

LLM_MAX_OUTPUT_TOKENS = 16000

# Longest transcript (characters of "[mm:ss] text" lines, roughly an hour of
# speech) summarized in one request. Longer videos: summarize a section.
LLM_MAX_TRANSCRIPT_CHARS = 60_000

# Number of transcript chunks retrieved as evidence for a grounded answer.
LLM_QA_TOP_K = 8

# ---------------------------------------------------------------------------
# Quiz generator settings
# ---------------------------------------------------------------------------

QUIZ_QUESTION_COUNTS = (5, 10, 15)

# Ask the LLM for a few extra questions, because some fail validation
# (no supporting quote, duplicate, malformed options) and are dropped.
QUIZ_EXTRA_CANDIDATES = 4

# Transcript chunks with fewer words than this carry too little content to
# base a question on.
QUIZ_MIN_CHUNK_WORDS = 12

# At most this much transcript text (characters) is sent per quiz request.
# Longer videos are sampled evenly across their whole length.
QUIZ_MAX_TRANSCRIPT_CHARS = 60_000

# A topic where the student scored below this fraction is flagged for revision.
QUIZ_WEAK_TOPIC_THRESHOLD = 0.6

# ---------------------------------------------------------------------------
# API settings (Milestone 8)
# ---------------------------------------------------------------------------

API_HOST = "127.0.0.1"
API_PORT = 8000
