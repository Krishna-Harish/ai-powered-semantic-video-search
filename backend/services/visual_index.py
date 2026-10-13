"""
visual_index.py  (Milestone 6/7)

Builds and queries a FAISS index over all videos' visual (frame) embeddings.
Same principle as text_index.py, but the vectors come from CLIP's image
encoder, and queries come from CLIP's text encoder -- both live in the
same shared vector space, which is what lets a text query find matching
images.
"""

import os
import json
import glob
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.config import EMBEDDINGS_DIR, INDEXES_DIR, CLIP_EMBEDDING_DIM

INDEX_PATH = os.path.join(INDEXES_DIR, "visual.index")
METADATA_PATH = os.path.join(INDEXES_DIR, "visual_metadata.json")


class VisualIndexError(Exception):
    pass


def build_visual_index() -> str:
    """
    Scans data/embeddings/*_visual.npy for every processed video, stacks
    all frame vectors into one FAISS index, and saves aligned metadata.
    """
    try:
        import faiss
    except ImportError:
        raise VisualIndexError("faiss is not installed. Run: pip install faiss-cpu")

    vector_files = sorted(glob.glob(os.path.join(EMBEDDINGS_DIR, "*_visual.npy")))
    if len(vector_files) == 0:
        raise VisualIndexError(
            "No visual embedding files found. Run embed_video_frames() for your videos first."
        )

    all_vectors = []
    all_metadata = []

    for vec_path in vector_files:
        video_id = os.path.basename(vec_path).replace("_visual.npy", "")
        meta_path = os.path.join(EMBEDDINGS_DIR, f"{video_id}_visual_meta.json")
        if not os.path.isfile(meta_path):
            continue

        vectors = np.load(vec_path)
        with open(meta_path, "r") as f:
            frames = json.load(f)

        if len(vectors) != len(frames):
            raise VisualIndexError(
                f"Vector/metadata count mismatch for '{video_id}': "
                f"{len(vectors)} vectors vs {len(frames)} frames."
            )

        for frame in frames:
            frame["video_id"] = video_id

        all_vectors.append(vectors)
        all_metadata.extend(frames)

    if len(all_vectors) == 0:
        raise VisualIndexError("No valid visual embeddings found to index.")

    matrix = np.vstack(all_vectors).astype("float32")

    index = faiss.IndexFlatIP(CLIP_EMBEDDING_DIM)
    index.add(matrix)

    faiss.write_index(index, INDEX_PATH)
    with open(METADATA_PATH, "w") as f:
        json.dump(all_metadata, f, indent=2)

    return INDEX_PATH


def require_visual_index():
    """Raises VisualIndexError if no visual index has been built yet (checked before
    loading any model, so the error names the real problem)."""
    if not os.path.isfile(INDEX_PATH) or not os.path.isfile(METADATA_PATH):
        raise VisualIndexError(
            "No video has been indexed for visual search yet. "
            "Select the video and click 'Index now', or run scripts/build_visual_index.py."
        )


def search_visual_index(query_vector: np.ndarray, top_k: int = 5) -> list:
    """
    Searches the visual FAISS index with a CLIP text-query vector.
    Returns ranked results: [{"video_id", "timestamp", "frame_path", "score"}, ...]
    """
    try:
        import faiss
    except ImportError:
        raise VisualIndexError("faiss is not installed. Run: pip install faiss-cpu")

    require_visual_index()

    index = faiss.read_index(INDEX_PATH)
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)

    query = np.asarray(query_vector, dtype="float32").reshape(1, -1)
    scores, indices = index.search(query, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1 or idx >= len(metadata):
            continue
        frame = metadata[idx]
        results.append({
            "video_id": frame["video_id"],
            "timestamp": frame["timestamp"],
            "frame_path": frame["path"],
            "score": float(score),
        })

    return results
