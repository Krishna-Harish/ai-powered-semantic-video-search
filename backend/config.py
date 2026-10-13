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

DATA_DIR = os.path.join(BASE_DIR, "data")
VIDEOS_DIR = os.path.join(DATA_DIR, "videos")
AUDIO_DIR = os.path.join(DATA_DIR, "audio")
FRAMES_DIR = os.path.join(DATA_DIR, "frames")
TRANSCRIPTS_DIR = os.path.join(DATA_DIR, "transcripts")
EMBEDDINGS_DIR = os.path.join(DATA_DIR, "embeddings")
INDEXES_DIR = os.path.join(DATA_DIR, "indexes")
METADATA_DIR = os.path.join(DATA_DIR, "metadata")

for folder in [VIDEOS_DIR, AUDIO_DIR, FRAMES_DIR, TRANSCRIPTS_DIR,
               EMBEDDINGS_DIR, INDEXES_DIR, METADATA_DIR]:
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
WHISPER_MODEL_SIZE = "small"

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

# Which LLM generates summaries and answers:
#   "auto"      -- Claude if ANTHROPIC_API_KEY is set, otherwise Gemini if
#                  GEMINI_API_KEY (or GOOGLE_API_KEY) is set
#   "gemini"    -- always Gemini (free tier key: https://aistudio.google.com/apikey)
#   "anthropic" -- always Claude (paid API key)
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "auto")

# Gemini model. "gemini-flash-latest" always points at Google's current Flash
# model, which is available on the free tier.
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")

# Claude model and thinking depth (low | medium | high | xhigh | max).
LLM_MODEL = "claude-opus-5-5"

LLM_EFFORT = "medium"

LLM_MAX_OUTPUT_TOKENS = 16000

# Transcripts longer than this (characters of "[mm:ss] text" lines) are
# summarized window-by-window and then combined, instead of in one request.
LLM_TRANSCRIPT_WINDOW_CHARS = 200_000

# Number of transcript chunks retrieved as evidence for a grounded answer.
LLM_QA_TOP_K = 8

# ---------------------------------------------------------------------------
# API settings (Milestone 8)
# ---------------------------------------------------------------------------

API_HOST = "127.0.0.1"
API_PORT = 8000
