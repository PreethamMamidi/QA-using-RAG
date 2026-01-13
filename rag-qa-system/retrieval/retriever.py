"""
Retrieval utilities for the RAG QA system.

Implements a simple function to embed a query, search a FAISS index built on
L2-normalized chunk embeddings, and return the top matching chunks.
"""
from typing import Dict, List

import numpy as np
import faiss

from embeddings.embedder import embed_texts
from vector_store.faiss_index import search_index


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
	"""L2-normalize rows of a 2D array, safely handling zeros."""
	if x.ndim == 1:
		x = x.reshape(1, -1)
	norms = np.linalg.norm(x, axis=1, keepdims=True)
	norms = np.maximum(norms, eps)
	return x / norms


def retrieve_chunks(
	query: str,
	index: faiss.Index,
	chunks: List[Dict[str, str]],
	top_k: int = 5,
) -> List[Dict[str, str]]:
	"""
	Retrieve the most relevant text chunks for a query using FAISS.

	Parameters
	----------
	query : str
		User query string to embed and search.
	index : faiss.Index
		FAISS index built on L2-normalized chunk embeddings (e.g., IndexFlatIP).
	chunks : List[Dict[str, str]]
		In-memory list of chunk metadata dicts, each containing at least
		{"chunk_id", "document_id", "text"}. Order must correspond to the
		embeddings added to the index.
	top_k : int, optional
		Number of results to return, by default 5.

	Returns
	-------
	List[Dict[str, str]]
		Top matching chunks sorted by similarity (descending). Each element is
		the original chunk dictionary augmented with a "score" field.

	Raises
	------
	ValueError
		If inputs are invalid (e.g., empty query, index/chunks mismatch).
	"""
	if index is None:
		raise ValueError("FAISS index is None.")
	if not isinstance(chunks, list) or len(chunks) == 0:
		raise ValueError("Chunks list is empty or invalid.")
	if not query or not query.strip():
		return []

	# Validate that index and chunks agree on count where possible
	ntotal = getattr(index, "ntotal", None)
	if ntotal is not None and ntotal != len(chunks):
		raise ValueError(
			f"Index/chunks size mismatch: index.ntotal={ntotal}, len(chunks)={len(chunks)}"
		)

	# Embed and L2-normalize the query for cosine via inner product
	q_vec = embed_texts([query], batch_size=1)
	q_vec = _l2_normalize(q_vec)

	# Search the index
	scores, inds = search_index(index, q_vec, top_k=top_k)
	scores = scores[0]
	inds = inds[0]

	# Map indices back to chunks; filter invalid indices (-1) if any
	results: List[Dict[str, str]] = []
	for idx, score in zip(inds.tolist(), scores.tolist()):
		if idx is None or idx < 0 or idx >= len(chunks):
			continue
		item = dict(chunks[idx])
		item["score"] = float(score)
		results.append(item)

	# Ensure descending by score (FAISS typically returns sorted, but be explicit)
	results.sort(key=lambda d: d.get("score", 0.0), reverse=True)
	return results

