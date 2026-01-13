"""
Embedding utilities for the RAG QA system.

Provides a lazily loaded sentence-transformer model (CPU-only) and a
batch embedding function that returns NumPy arrays.
"""
from typing import List

import numpy as np

_EMBEDDER = None  # Lazy-loaded SentenceTransformer instance


def get_embedder():
	"""Return a lazily loaded sentence-transformer model on CPU.

	Model: `sentence-transformers/all-MiniLM-L6-v2`

	Returns
	-------
	SentenceTransformer
		Loaded embedding model instance (cached after first call).
	"""
	global _EMBEDDER
	if _EMBEDDER is None:
		from sentence_transformers import SentenceTransformer
		_EMBEDDER = SentenceTransformer(
			"sentence-transformers/all-MiniLM-L6-v2", device="cpu"
		)
	return _EMBEDDER


def embed_texts(texts: List[str], batch_size: int = 32) -> np.ndarray:
	"""Embed a list of text strings into dense vectors.

	Parameters
	----------
	texts : List[str]
		List of text chunks to embed.
	batch_size : int, optional
		Batch size for encoding, by default 32.

	Returns
	-------
	numpy.ndarray
		2D array of shape (N, D) with dtype float32, where N is len(texts)
		and D is the embedding dimension.

	Notes
	-----
	- Uses CPU-only inference.
	- Returns an empty array with shape (0, D) when `texts` is empty.
	"""
	model = get_embedder()

	if not texts:
		dim = getattr(model, "get_sentence_embedding_dimension", lambda: 0)()
		return np.zeros((0, dim), dtype=np.float32)

	vectors = model.encode(
		texts,
		batch_size=batch_size,
		show_progress_bar=False,
		convert_to_numpy=True,
		normalize_embeddings=False,
	)

	# Ensure float32 for consistency
	if vectors.dtype != np.float32:
		vectors = vectors.astype(np.float32)

	return vectors

"""
Embedding utilities for the RAG QA system.

Provides lazily-loaded sentence-transformer model (CPU-only) and a helper
to embed batches of texts as a 2D NumPy array.
"""
from typing import List, Optional

import numpy as np

try:
	from sentence_transformers import SentenceTransformer
except Exception as e:  # pragma: no cover
	SentenceTransformer = None  # type: ignore


_EMBEDDER: Optional["SentenceTransformer"] = None
_MODEL_NAME: str = "all-MiniLM-L6-v2"


def get_embedder() -> "SentenceTransformer":
	"""Return a lazily-initialized sentence-transformer embedder (CPU-only).

	Loads the model only once and reuses it for subsequent calls.

	Returns
	-------
	SentenceTransformer
		The loaded sentence-transformer model.

	Raises
	------
	ImportError
		If `sentence-transformers` is not installed.
	"""
	global _EMBEDDER
	if SentenceTransformer is None:
		raise ImportError(
			"sentence-transformers is required. Install with 'pip install sentence-transformers'."
		)
	if _EMBEDDER is None:
		_EMBEDDER = SentenceTransformer(_MODEL_NAME, device="cpu")
	return _EMBEDDER


def embed_texts(texts: List[str], batch_size: int = 32) -> np.ndarray:
	"""Embed a list of texts into dense vectors using the embedder.

	Parameters
	----------
	texts : List[str]
		Input text strings to embed.
	batch_size : int, optional
		Batch size for model encoding, by default 32.

	Returns
	-------
	np.ndarray
		A 2D NumPy array of shape (N, D) where N is the number of inputs
		and D is the embedding dimension.
	"""
	model = get_embedder()
	if not texts:
		dim = model.get_sentence_embedding_dimension()
		return np.zeros((0, dim), dtype=np.float32)

	vectors: np.ndarray = model.encode(
		texts,
		batch_size=batch_size,
		device="cpu",
		convert_to_numpy=True,
		show_progress_bar=False,
	)
	# Ensure float32 for compatibility with FAISS
	if vectors.dtype != np.float32:
		vectors = vectors.astype(np.float32, copy=False)
	return vectors

