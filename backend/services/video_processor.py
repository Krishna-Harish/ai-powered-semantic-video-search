"""
video_processor.py  (Milestone 1)

raw video file  -->  audio.wav  +  sampled frame images + frame metadata

Uses FFmpeg for extraction. No AI models involved in this module.
"""

import os
import json
import subprocess
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.config import (
    AUDIO_DIR,
    FRAMES_DIR,
    METADATA_DIR,
    AUDIO_SAMPLE_RATE,
    AUDIO_CHANNELS,
    FRAME_SAMPLE_INTERVAL_SECONDS,
    SUPPORTED_VIDEO_EXTENSIONS,
)


class VideoProcessingError(Exception):
    """Raised when something goes wrong extracting audio/frames from a video."""
    pass


def _check_ffmpeg_installed():
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise VideoProcessingError(
            "FFmpeg was not found on your system PATH. "
            "Install it and confirm 'ffmpeg -version' works before running this pipeline."
        )


def _validate_video_file(video_path: str):
    if not os.path.isfile(video_path):
        raise VideoProcessingError(f"Video file does not exist: {video_path}")

    ext = os.path.splitext(video_path)[1].lower()
    if ext not in SUPPORTED_VIDEO_EXTENSIONS:
        raise VideoProcessingError(
            f"Unsupported video format '{ext}'. Supported: {SUPPORTED_VIDEO_EXTENSIONS}"
        )

    if os.path.getsize(video_path) == 0:
        raise VideoProcessingError(f"Video file is empty/corrupted: {video_path}")


def get_video_duration(video_path: str) -> float:
    """Returns video duration in seconds using ffprobe."""
    command = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0 or not result.stdout.strip():
        raise VideoProcessingError(
            f"Could not read duration for '{video_path}'. It may be corrupted.\n{result.stderr}"
        )
    return float(result.stdout.strip())


def extract_audio(video_path: str, video_id: str) -> str:
    """Extracts mono, 16kHz audio from a video using FFmpeg. Idempotent."""
    _check_ffmpeg_installed()
    _validate_video_file(video_path)

    output_path = os.path.join(AUDIO_DIR, f"{video_id}.wav")

    if os.path.isfile(output_path) and os.path.getsize(output_path) > 0:
        return output_path  # already extracted, skip re-work

    command = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",
        "-ac", str(AUDIO_CHANNELS),
        "-ar", str(AUDIO_SAMPLE_RATE),
        output_path,
    ]
    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0:
        raise VideoProcessingError(
            f"FFmpeg failed to extract audio from '{video_path}'.\n{result.stderr}"
        )
    if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
        raise VideoProcessingError(
            f"Audio extraction produced an empty file for '{video_path}'. "
            "The video might have no audio track."
        )
    return output_path


def extract_frames(video_path: str, video_id: str, interval_seconds: int = None) -> dict:
    """
    Extracts one frame every `interval_seconds` seconds using FFmpeg.
    Saves frame_000001.jpg, frame_000002.jpg, ... and a metadata JSON
    mapping each frame filename to its timestamp in the source video.
    Idempotent: skips extraction if frames + metadata already exist.

    Returns dict: {"frames_dir": ..., "metadata_path": ..., "num_frames": ...}
    """
    _check_ffmpeg_installed()
    _validate_video_file(video_path)

    if interval_seconds is None:
        interval_seconds = FRAME_SAMPLE_INTERVAL_SECONDS

    output_dir = os.path.join(FRAMES_DIR, video_id)
    os.makedirs(output_dir, exist_ok=True)
    metadata_path = os.path.join(METADATA_DIR, f"{video_id}_frames.json")

    existing_frames = [f for f in os.listdir(output_dir) if f.endswith(".jpg")]
    if existing_frames and os.path.isfile(metadata_path):
        with open(metadata_path, "r") as f:
            existing_meta = json.load(f)
        return {
            "frames_dir": output_dir,
            "metadata_path": metadata_path,
            "num_frames": len(existing_meta),
        }

    fps_filter = f"fps=1/{interval_seconds}"
    output_pattern = os.path.join(output_dir, "frame_%06d.jpg")

    command = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vf", fps_filter,
        "-qscale:v", "2",
        output_pattern,
    ]
    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0:
        raise VideoProcessingError(
            f"FFmpeg failed to extract frames from '{video_path}'.\n{result.stderr}"
        )

    frame_files = sorted(f for f in os.listdir(output_dir) if f.endswith(".jpg"))
    if len(frame_files) == 0:
        raise VideoProcessingError(
            f"No frames extracted from '{video_path}'. The file might be corrupted."
        )

    # Build frame -> timestamp metadata.
    # Frame N (1-indexed) corresponds to timestamp (N-1) * interval_seconds,
    # matching ffmpeg's fps=1/interval sampling behavior (first frame at t=0).
    metadata = []
    for idx, filename in enumerate(frame_files):
        timestamp = idx * interval_seconds
        metadata.append({
            "frame_id": idx + 1,
            "filename": filename,
            "timestamp": float(timestamp),
            "path": os.path.join(output_dir, filename),
        })

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return {
        "frames_dir": output_dir,
        "metadata_path": metadata_path,
        "num_frames": len(metadata),
    }


def process_video(video_path: str, video_id: str) -> dict:
    """Runs audio + frame extraction for one video. Returns a summary dict."""
    duration = get_video_duration(video_path)
    audio_path = extract_audio(video_path, video_id)
    frame_result = extract_frames(video_path, video_id)

    return {
        "video_id": video_id,
        "video_path": video_path,
        "duration_seconds": duration,
        "audio_path": audio_path,
        "frames_dir": frame_result["frames_dir"],
        "frame_metadata_path": frame_result["metadata_path"],
        "num_frames": frame_result["num_frames"],
    }
