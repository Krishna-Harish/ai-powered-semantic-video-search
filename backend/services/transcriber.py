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
import wave

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.config import (
    TRANSCRIPTS_DIR,
    WHISPER_MODEL_SIZE,
    WHISPER_TASK,
    WHISPER_DEVICE,
    WHISPER_COMPUTE_TYPE,
    WHISPER_BEAM_SIZE,
    WHISPER_CPU_THREADS,
)

_model = None  # lazy-loaded singleton, so the (large) model loads only once per process


class TranscriptionError(Exception):
    pass


def load_wav_16k_mono(audio_path: str) -> np.ndarray:
    """
    Reads the 16 kHz mono 16-bit WAV written by video_processor.extract_audio
    into the float32 array Whisper expects (values in [-1, 1]).

    We pass this array to faster-whisper instead of the file path, so it
    doesn't have to decode audio itself through PyAV -- newer PyAV releases
    removed an argument faster-whisper still passes, which crashes with
    "open() got an unexpected keyword argument 'metadata_errors'".
    """
    try:
        with wave.open(audio_path, "rb") as wav:
            channels, width, rate = wav.getnchannels(), wav.getsampwidth(), wav.getframerate()
            frames = wav.readframes(wav.getnframes())
    except (wave.Error, EOFError) as e:
        raise TranscriptionError(f"Could not read audio file '{audio_path}': {e}")

    if width != 2 or rate != 16000:
        raise TranscriptionError(
            f"Expected 16 kHz 16-bit audio in '{audio_path}', got {rate} Hz {8 * width}-bit. "
            "Delete the file and re-run processing to regenerate it."
        )
    audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio


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
                cpu_threads=WHISPER_CPU_THREADS,
            )
        except Exception as e:
            raise TranscriptionError(
                f"Could not load the Whisper '{WHISPER_MODEL_SIZE}' model. The first run downloads it "
                f"from Hugging Face, so it needs internet access. ({type(e).__name__}: {e})"
            )
    return _model


def _clock(seconds: float) -> str:
    return f"{int(seconds // 60)}:{int(seconds % 60):02d}"


def transcribe_audio(audio_path: str, video_id: str, force: bool = False, progress=None) -> str:
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

    `progress(message)` is called as segments are decoded, e.g.
    "Transcribing with Whisper... 45% (4:31 of 10:02)".

    Returns the path to the transcript JSON.
    """
    if not os.path.isfile(audio_path):
        raise TranscriptionError(f"Audio file not found: {audio_path}")
    if os.path.getsize(audio_path) == 0:
        raise TranscriptionError(f"Audio file is empty: {audio_path}")

    output_path = os.path.join(TRANSCRIPTS_DIR, f"{video_id}.json")

    if os.path.isfile(output_path) and not force:
        return output_path

    audio = load_wav_16k_mono(audio_path)
    duration = len(audio) / 16000.0

    if progress is not None:
        progress(f"Loading Whisper '{WHISPER_MODEL_SIZE}' model (downloads on first use)...")
    model = _get_model()
    if progress is not None:
        progress("Transcribing with Whisper... 0%")

    try:
        segments_iter, info = model.transcribe(
            audio,
            task=WHISPER_TASK,
            beam_size=WHISPER_BEAM_SIZE,
            vad_filter=True,  # skip silent stretches, avoids hallucinated text
        )
        # segments_iter is lazy: decoding happens as we iterate, so this is
        # where the time goes and where progress can be reported.
        segments = []
        for s in segments_iter:
            segments.append({"start": float(s.start), "end": float(s.end), "text": s.text.strip()})
            if progress is not None and duration > 0:
                done = min(float(s.end), duration)
                progress(
                    f"Transcribing with Whisper... {100 * done / duration:.0f}% "
                    f"({_clock(done)} of {_clock(duration)})"
                )
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
