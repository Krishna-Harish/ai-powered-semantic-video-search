"""
search.py

Ties together: user query -> query embedding -> FAISS search -> ranked results.
This is the module the FastAPI routes call directly. Keeping this as its
own layer (instead of putting logic in main.py) means the search logic is
testable independently of the web framework.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.config import DEFAULT_TOP_K, MIN_SIMILARITY_SCORE
from backend.services.text_embeddings import embed_text_query
from backend.services.text_index import search_text_index, TextIndexError
from backend.services.visual_embeddings import embed_text_query_clip
from backend.services.visual_index import search_visual_index, VisualIndexError


def _format_timestamp(seconds: float) -> str:
    """143.2 -> '02:23' -- used for human-readable display, not for logic."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def text_search(query: str, top_k: int = None) -> dict:
    """
    Full text-search pipeline: embed the query with the sentence-transformer,
    search the text FAISS index, filter weak matches, return ranked results.
    """
    if not query or not query.strip():
        raise ValueError("Query must not be empty.")

    top_k = top_k or DEFAULT_TOP_K

    query_vector = embed_text_query(query)
    raw_results = search_text_index(query_vector, top_k=top_k)

    results = [r for r in raw_results if r["score"] >= MIN_SIMILARITY_SCORE]
    for r in results:
        r["timestamp_display"] = _format_timestamp(r["start"])

    return {"query": query, "results": results}


def visual_search(query: str, top_k: int = None) -> dict:
    """
    Full visual-search pipeline: embed the query with CLIP's text encoder,
    search the visual FAISS index, filter weak matches, return ranked results.
    """
    if not query or not query.strip():
        raise ValueError("Query must not be empty.")

    top_k = top_k or DEFAULT_TOP_K

    query_vector = embed_text_query_clip(query)
    raw_results = search_visual_index(query_vector, top_k=top_k)

    results = [r for r in raw_results if r["score"] >= MIN_SIMILARITY_SCORE]
    for r in results:
        r["timestamp_display"] = _format_timestamp(r["timestamp"])

    return {"query": query, "results": results}
