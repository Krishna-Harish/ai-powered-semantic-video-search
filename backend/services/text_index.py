"""
text_index.py  (Milestone 5)

Builds and queries a FAISS index over all videos' text (transcript-chunk)
embeddings. FAISS is NOT an AI model -- it's a fast vector similarity
search library. It does not "understand" anything; it just finds which
stored vectors are closest to a query vector, extremely quickly.

Since our text embeddings are already L2-normalized (see text_embeddings.py),
we use IndexFlatIP (inner product). For normalized vectors, inner product
is mathematically equivalent to cosine similarity -- this is the standard,
correct way to do cosine-similarity search with FAISS.
"""

import os
import json
import glob
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.config import EMBEDDINGS_DIR, INDEXES_DIR, TEXT_EMBEDDING_DIM

INDEX_PATH = os.path.join(INDEXES_DIR, "text.index")
METADATA_PATH = os.path.join(INDEXES_DIR, "text_metadata.json")


class TextIndexError(Exception):
    pass


def build_text_index() -> str:
    """
    Scans data/embeddings/*_text.npy for every processed video, stacks
    all their vectors into one FAISS index, and saves a metadata list
    where metadata[i] describes exactly which chunk FAISS vector i is.

    Returns the path to the saved FAISS index file.
    """
    try:
        import faiss
    except ImportError:
        raise TextIndexError("faiss is not installed. Run: pip install faiss-cpu")

    vector_files = sorted(glob.glob(os.path.join(EMBEDDINGS_DIR, "*_text.npy")))
    if len(vector_files) == 0:
        raise TextIndexError(
            "No text embedding files found. Run embed_video_chunks() for your videos first."
        )

    all_vectors = []
    all_metadata = []

    for vec_path in vector_files:
        video_id = os.path.basename(vec_path).replace("_text.npy", "")
        meta_path = os.path.join(EMBEDDINGS_DIR, f"{video_id}_text_meta.json")
        if not os.path.isfile(meta_path):
            continue  # skip if metadata is missing/inconsistent

        vectors = np.load(vec_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        if len(vectors) != len(chunks):
            raise TextIndexError(
                f"Vector/metadata count mismatch for '{video_id}': "
                f"{len(vectors)} vectors vs {len(chunks)} chunks."
            )

        all_vectors.append(vectors)
        all_metadata.extend(chunks)

    if len(all_vectors) == 0:
        raise TextIndexError("No valid video embeddings found to index.")

    matrix = np.vstack(all_vectors).astype("float32")

    index = faiss.IndexFlatIP(TEXT_EMBEDDING_DIM)
    index.add(matrix)

    faiss.write_index(index, INDEX_PATH)
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, indent=2, ensure_ascii=False)

    return INDEX_PATH


def search_text_index(query_vector: np.ndarray, top_k: int = 5) -> list:
    """
    Searches the text FAISS index with a single query vector
    (already embedded and normalized -- see text_embeddings.embed_text_query).

    Returns a list of result dicts, ranked best-first:
    [{"video_id", "start", "end", "text", "score"}, ...]
    """
    try:
        import faiss
    except ImportError:
        raise TextIndexError("faiss is not installed. Run: pip install faiss-cpu")

    if not os.path.isfile(INDEX_PATH) or not os.path.isfile(METADATA_PATH):
        raise TextIndexError(
            "Text index not found. Run build_text_index() before searching."
        )

    index = faiss.read_index(INDEX_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    query = np.asarray(query_vector, dtype="float32").reshape(1, -1)
    scores, indices = index.search(query, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1 or idx >= len(metadata):
            continue
        chunk = metadata[idx]
        results.append({
            "video_id": chunk["video_id"],
            "start": chunk["start"],
            "end": chunk["end"],
            "text": chunk["text"],
            "score": float(score),
        })

    return results
