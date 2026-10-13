"""
transcriber.py  (Milestone 2)

audio.wav  -->  timestamped transcript (JSON)

Uses faster-whisper, a pretrained speech-recognition model. We are NOT
training this model -- it is used purely for inference/feature extraction
on our video corpus.

NOTE: faster-whisper + its model weights must be installed/downloaded on
the machine actually running this (requires internet on first run to fetch
weights from Hugging Face). See requirements.txt.
"""

import os
import json
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.config import (
    TRANSCRIPTS_DIR,
    WHISPER_MODEL_SIZE,
    WHISPER_TASK,
    WHISPER_DEVICE,
    WHISPER_COMPUTE_TYPE,
)

_model = None  # lazy-loaded singleton, so the (large) model loads only once per process


class TranscriptionError(Exception):
    pass


def _get_model():
    """Lazily loads the faster-whisper model once and reuses it for every video."""
    global _model
    if _model is None:
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise TranscriptionError(
                "faster-whisper is not installed. Run: pip install faster-whisper"
            )
        try:
            _model = WhisperModel(
                WHISPER_MODEL_SIZE,
                device=WHISPER_DEVICE,
                compute_type=WHISPER_COMPUTE_TYPE,
            )
        except Exception as e:
            raise TranscriptionError(
                f"Could not load the Whisper '{WHISPER_MODEL_SIZE}' model. The first run downloads it "
                f"from Hugging Face, so it needs internet access. ({type(e).__name__}: {e})"
            )
    return _model


def transcribe_audio(audio_path: str, video_id: str, force: bool = False) -> str:
    """
    Transcribes an audio file into a timestamped transcript JSON.
    Idempotent: skips re-transcription if the output already exists
    (unless force=True).

    Output format (data/transcripts/<video_id>.json):
    {
        "video_id": "video1",
        "task": "translate",
        "segments": [
            {"start": 0.0, "end": 5.7, "text": "..."},
            ...
        ]
    }

    Returns the path to the transcript JSON.
    """
    if not os.path.isfile(audio_path):
        raise TranscriptionError(f"Audio file not found: {audio_path}")
    if os.path.getsize(audio_path) == 0:
        raise TranscriptionError(f"Audio file is empty: {audio_path}")

    output_path = os.path.join(TRANSCRIPTS_DIR, f"{video_id}.json")

    if os.path.isfile(output_path) and not force:
        return output_path

    model = _get_model()

    try:
        segments_iter, info = model.transcribe(
            audio_path,
            task=WHISPER_TASK,
            vad_filter=True,  # skip silent stretches, avoids hallucinated text
        )
        segments = [
            {"start": float(s.start), "end": float(s.end), "text": s.text.strip()}
            for s in segments_iter
        ]
    except Exception as e:
        raise TranscriptionError(f"Whisper failed to transcribe '{audio_path}': {e}")

    if len(segments) == 0:
        raise TranscriptionError(
            f"Transcription produced zero segments for '{audio_path}'. "
            "The audio might be silent or unintelligible."
        )

    transcript = {
        "video_id": video_id,
        "task": WHISPER_TASK,
        "detected_language": getattr(info, "language", None),
        "segments": segments,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(transcript, f, indent=2, ensure_ascii=False)

    return output_path
