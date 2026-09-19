"""
text_embeddings.py  (Milestone 4)

transcript chunks  -->  text embeddings (vectors)

Uses sentence-transformers (all-MiniLM-L6-v2), a PRETRAINED model.
We do not train or fine-tune this model. It converts text into a
384-dimensional vector that captures semantic meaning, so that
sentences with similar meaning end up close together in vector space
-- even if they don't share exact words.
"""

import os
import json
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.config import METADATA_DIR, EMBEDDINGS_DIR, TEXT_EMBEDDING_MODEL

_model = None  # lazy-loaded singleton


class TextEmbeddingError(Exception):
    pass


def _get_model():
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise TextEmbeddingError(
                "sentence-transformers is not installed. "
                "Run: pip install sentence-transformers"
            )
        _model = SentenceTransformer(TEXT_EMBEDDING_MODEL)
    return _model


def embed_text_query(query: str) -> np.ndarray:
    """Embeds a single user query string. Returns a normalized 1D vector."""
    model = _get_model()
    vector = model.encode([query], normalize_embeddings=True)[0]
    return np.asarray(vector, dtype="float32")


def embed_video_chunks(video_id: str, force: bool = False) -> str:
    """
    Reads data/metadata/<video_id>_chunks.json, embeds every chunk's text,
    and saves the resulting vectors + aligned metadata.

    Saves:
        data/embeddings/<video_id>_text.npy      -- (num_chunks, 384) float32 array
        data/embeddings/<video_id>_text_meta.json -- chunk metadata, same order as rows

    Idempotent unless force=True.
    Returns the path to the saved .npy file.
    """
    chunks_path = os.path.join(METADATA_DIR, f"{video_id}_chunks.json")
    if not os.path.isfile(chunks_path):
        raise TextEmbeddingError(f"Chunks file not found for '{video_id}': {chunks_path}")

    vectors_path = os.path.join(EMBEDDINGS_DIR, f"{video_id}_text.npy")
    meta_path = os.path.join(EMBEDDINGS_DIR, f"{video_id}_text_meta.json")

    if os.path.isfile(vectors_path) and os.path.isfile(meta_path) and not force:
        return vectors_path

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    if len(chunks) == 0:
        raise TextEmbeddingError(f"No chunks to embed for '{video_id}'.")

    model = _get_model()
    texts = [c["text"] for c in chunks]

    try:
        # normalize_embeddings=True makes vectors unit-length, so cosine
        # similarity becomes equivalent to a simple dot product / inner
        # product -- which is what FAISS's IndexFlatIP computes directly.
        vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    except Exception as e:
        raise TextEmbeddingError(f"Embedding failed for '{video_id}': {e}")

    vectors = np.asarray(vectors, dtype="float32")

    np.save(vectors_path, vectors)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    return vectors_path
