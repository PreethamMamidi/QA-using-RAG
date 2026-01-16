"""Cross-encoder re-ranking for retrieved chunks (CPU-only)."""
from typing import Dict, List, Optional, Tuple

from sentence_transformers import CrossEncoder


_MODEL_CACHE: Optional[CrossEncoder] = None
_DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def get_reranker() -> CrossEncoder:
    """Lazily load and cache the CrossEncoder model.

    Returns
    -------
    CrossEncoder
        Loaded cross-encoder instance (CPU-only).
    """
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        _MODEL_CACHE = CrossEncoder(_DEFAULT_MODEL)
    return _MODEL_CACHE


def rerank_chunks(
    query: str,
    chunks: List[Dict[str, str]],
    top_k: int = 5,
) -> List[Dict[str, str]]:
    """Rerank retrieved chunks with a cross-encoder.

    Parameters
    ----------
    query : str
        User query.
    chunks : List[Dict[str, str]]
        Retrieved chunks containing at least a "text" field.
    top_k : int, optional
        Number of reranked results to return, by default 5.

    Returns
    -------
    List[Dict[str, str]]
        Reranked chunks with added "rerank_score" and "retrieval" fields.
    """
    if not query or not query.strip():
        return []
    if not chunks:
        return []

    filtered: List[Dict[str, str]] = []
    for ch in chunks:
        text = (ch.get("text") or "").strip()
        if text:
            filtered.append(ch)

    if not filtered:
        return []

    model = get_reranker()
    pairs = [[query, c.get("text", "")] for c in filtered]
    scores = model.predict(pairs)

    rescored: List[Tuple[float, Dict[str, str]]] = []
    for score, ch in zip(scores.tolist(), filtered):
        item = dict(ch)
        item["rerank_score"] = float(score)
        item["retrieval"] = "reranked"
        rescored.append((item["rerank_score"], item))

    rescored.sort(key=lambda t: t[0], reverse=True)
    return [item for _, item in rescored[:top_k]]
