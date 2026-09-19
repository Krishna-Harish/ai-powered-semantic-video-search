"""
visual_embeddings.py  (Milestone 6)

sampled frames  -->  visual embeddings (vectors)
text query      -->  query embedding in the SAME vector space

Uses OpenCLIP (CLIP), a PRETRAINED multimodal model. CLIP has two
encoders that were jointly trained to map images and text into a
shared vector space -- so a photo of a cat and the text "a cat" end
up close together, even though one is an image and one is text.
This is what makes text-to-image search possible.

We do not train or fine-tune CLIP. It is used purely for feature
extraction on our sampled video frames and on the user's text query.
"""

import os
import json
import sys
import numpy as np
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.config import METADATA_DIR, EMBEDDINGS_DIR, CLIP_MODEL_NAME, CLIP_PRETRAINED

_model = None
_preprocess = None
_tokenizer = None


class VisualEmbeddingError(Exception):
    pass


def _get_clip():
    global _model, _preprocess, _tokenizer
    if _model is None:
        try:
            import open_clip
            import torch
        except ImportError:
            raise VisualEmbeddingError(
                "open_clip_torch / torch is not installed. "
                "Run: pip install open_clip_torch torch"
            )
        _model, _, _preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED
        )
        _tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
        _model.eval()
    return _model, _preprocess, _tokenizer


def embed_text_query_clip(query: str) -> np.ndarray:
    """Embeds a text query using CLIP's text encoder (for visual search)."""
    import torch
    model, _, tokenizer = _get_clip()
    tokens = tokenizer([query])
    with torch.no_grad():
        features = model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
    return features[0].numpy().astype("float32")


def embed_video_frames(video_id: str, force: bool = False) -> str:
    """
    Reads data/metadata/<video_id>_frames.json (produced by video_processor),
    embeds every sampled frame image using CLIP's image encoder, and saves
    the resulting vectors + aligned metadata.

    Saves:
        data/embeddings/<video_id>_visual.npy       -- (num_frames, 512) float32
        data/embeddings/<video_id>_visual_meta.json -- frame metadata, same row order

    Idempotent unless force=True.
    Returns the path to the saved .npy file.
    """
    import torch

    frames_meta_path = os.path.join(METADATA_DIR, f"{video_id}_frames.json")
    if not os.path.isfile(frames_meta_path):
        raise VisualEmbeddingError(f"Frame metadata not found for '{video_id}': {frames_meta_path}")

    vectors_path = os.path.join(EMBEDDINGS_DIR, f"{video_id}_visual.npy")
    meta_path = os.path.join(EMBEDDINGS_DIR, f"{video_id}_visual_meta.json")

    if os.path.isfile(vectors_path) and os.path.isfile(meta_path) and not force:
        return vectors_path

    with open(frames_meta_path, "r") as f:
        frames = json.load(f)

    if len(frames) == 0:
        raise VisualEmbeddingError(f"No frames to embed for '{video_id}'.")

    model, preprocess, _ = _get_clip()

    vectors = []
    valid_frames = []
    for frame in frames:
        frame_path = frame["path"]
        if not os.path.isfile(frame_path):
            continue  # skip missing frame, don't crash the whole batch
        try:
            image = Image.open(frame_path).convert("RGB")
            image_input = preprocess(image).unsqueeze(0)
            with torch.no_grad():
                features = model.encode_image(image_input)
                features = features / features.norm(dim=-1, keepdim=True)
            vectors.append(features[0].numpy())
            valid_frames.append(frame)
        except Exception:
            continue  # skip unreadable/corrupted frame image

    if len(vectors) == 0:
        raise VisualEmbeddingError(f"No frames could be embedded for '{video_id}'.")

    vectors = np.asarray(vectors, dtype="float32")

    np.save(vectors_path, vectors)
    with open(meta_path, "w") as f:
        json.dump(valid_frames, f, indent=2)

    return vectors_path
