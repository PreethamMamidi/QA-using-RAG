"""
FAISS index utilities for the RAG QA system (CPU-only).

Implements a minimal cosine-similarity index using inner product on
L2-normalized embeddings.
"""
from typing import Tuple

import numpy as np
import faiss


def _to_float32_contiguous(x: np.ndarray) -> np.ndarray:
	"""Ensure array is 2D, float32, and contiguous in memory."""
	if x.ndim != 2:
		raise ValueError(f"Embeddings must be 2D (got shape {x.shape}).")
	return np.ascontiguousarray(x.astype(np.float32, copy=False))


def build_index(embeddings: np.ndarray) -> faiss.Index:
	"""Build a FAISS inner-product index from L2-normalized embeddings.

	Parameters
	----------
	embeddings : np.ndarray
		2D array of shape (N, D) containing L2-normalized float vectors.

	Returns
	-------
	faiss.Index
		A FAISS IndexFlatIP with the embeddings added.

	Raises
	------
	ValueError
		If `embeddings` are empty or invalid shape.
	"""
	if embeddings is None:
		raise ValueError("Embeddings array is None.")
	if embeddings.size == 0:
		raise ValueError("Embeddings array is empty.")

	embs = _to_float32_contiguous(embeddings)
	n, d = embs.shape
	index = faiss.IndexFlatIP(d)  # inner product (cosine when L2-normalized)
	index.add(embs)
	return index


def save_index(index: faiss.Index, index_path: str) -> None:
	"""Persist FAISS index to disk.

	Parameters
	----------
	index : faiss.Index
		FAISS index to save.
	index_path : str
		File path where the index will be written.

	Raises
	------
	ValueError
		If `index` is None.
	"""
	if index is None:
		raise ValueError("FAISS index is None; nothing to save.")
	faiss.write_index(index, index_path)


def load_index(index_path: str) -> faiss.Index:
	"""Load a FAISS index from disk.

	Parameters
	----------
	index_path : str
		Path to a previously saved FAISS index.

	Returns
	-------
	faiss.Index
		Loaded FAISS index.

	Raises
	------
	FileNotFoundError
		If the path cannot be read.
	"""
	try:
		return faiss.read_index(index_path)
	except Exception as e:
		# faiss throws generic exceptions; translate to a clearer error
		raise FileNotFoundError(f"Unable to load FAISS index from: {index_path}") from e


def search_index(
	index: faiss.Index,
	query_embeddings: np.ndarray,
	top_k: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
	"""Search the index with L2-normalized query embeddings.

	Parameters
	----------
	index : faiss.Index
		FAISS index (IndexFlatIP recommended).
	query_embeddings : np.ndarray
		2D array of shape (Q, D) with L2-normalized float vectors.
	top_k : int, optional
		Number of nearest neighbors to return, by default 5.

	Returns
	-------
	Tuple[np.ndarray, np.ndarray]
		(scores, indices) arrays of shape (Q, K).

	Raises
	------
	ValueError
		If inputs are invalid.
	"""
	if index is None:
		raise ValueError("FAISS index is None.")
	if query_embeddings is None:
		raise ValueError("Query embeddings are None.")
	if query_embeddings.size == 0:
		raise ValueError("Query embeddings array is empty.")
	if top_k <= 0:
		raise ValueError("top_k must be a positive integer.")

	q = _to_float32_contiguous(query_embeddings)
	qn, qd = q.shape

	# Validate dimensionality
	# IndexFlatIP exposes `d` (dimension); other index types also provide it.
	index_dim = getattr(index, "d", None)
	if index_dim is not None and index_dim != qd:
		raise ValueError(f"Query dim ({qd}) does not match index dim ({index_dim}).")

	# Avoid -1 results when requesting more neighbors than there are vectors
	ntotal = getattr(index, "ntotal", 0)
	k = min(top_k, ntotal) if ntotal > 0 else top_k

	scores, inds = index.search(q, k)
	return scores, inds

