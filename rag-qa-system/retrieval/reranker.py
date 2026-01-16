"""Cross-encoder re-ranking for retrieved chunks."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from sentence_transformers import CrossEncoder

_MODEL_CACHE: Optional[CrossEncoder] = None
_DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _get_model(model_name: str = _DEFAULT_MODEL) -> CrossEncoder:
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        _MODEL_CACHE = CrossEncoder(model_name)
    return _MODEL_CACHE


def rerank_chunks(
    query: str,
    chunks: List[Dict[str, str]],
    top_k: int = 8,
    model_name: str = _DEFAULT_MODEL,
) -> List[Dict[str, str]]:
    """Rerank retrieved chunks with a cross-encoder.

    Parameters
    ----------
    query : str
        User query.
    chunks : List[Dict[str, str]]
        Retrieved chunks (with at least a "text" field).
    top_k : int, optional
        Number of reranked results to return, by default 8.
    model_name : str, optional
        Cross-encoder model name, by default ms-marco MiniLM.

    Returns
    -------
    List[Dict[str, str]]
        Reranked chunks with updated "score".
    """
    if not query or not query.strip():
        return []
    if not chunks:
        return []

    model = _get_model(model_name)
    pairs = [[query, c.get("text", "")] for c in chunks]
    scores = model.predict(pairs)

    rescored: List[Tuple[float, Dict[str, str]]] = []
    for score, ch in zip(scores.tolist(), chunks):
        item = dict(ch)
        item["score"] = float(score)
        rescored.append((item["score"], item))

    rescored.sort(key=lambda t: t[0], reverse=True)
    return [item for _, item in rescored[:top_k]]
