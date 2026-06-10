"""BM25 sparse retrieval utilities for the RAG QA system."""
from __future__ import annotations

import logging
from typing import Dict, List, Sequence

from rank_bm25 import BM25Okapi


logger = logging.getLogger(__name__)


def tokenize_whitespace_lower(text: str) -> List[str]:
	"""Tokenize text using lowercase whitespace splitting."""
	if not text:
		return []
	return [token for token in text.lower().split() if token]


class BM25Retriever:
	"""Reusable BM25 retriever built from chunk texts.

	The retriever is initialized once for a fixed chunk corpus and can then be
	reused for multiple queries without rebuilding the BM25 index.
	"""

	def __init__(self, chunks: Sequence[Dict[str, str]]):
		self.chunks = [dict(chunk) for chunk in chunks or []]
		self.tokenized_corpus = [
			tokenize_whitespace_lower((chunk.get("text") or ""))
			for chunk in self.chunks
		]
		self.index = BM25Okapi(self.tokenized_corpus) if self.tokenized_corpus else None

	@property
	def ready(self) -> bool:
		"""Return True when the BM25 index is available."""
		return self.index is not None and bool(self.chunks)

	def retrieve(
		self,
		query: str,
		top_k: int = 5,
		debug: bool = False,
	) -> List[Dict[str, str]]:
		"""Return the top BM25-matched chunks for a query.

		Parameters
		----------
		query : str
			User query string.
		top_k : int, optional
			Maximum number of chunks to return, by default 5.
		debug : bool, optional
			If True, emit debug logging.
		"""
		if not query or not query.strip():
			return []
		if not self.ready:
			return []
		if top_k <= 0:
			return []

		query_tokens = tokenize_whitespace_lower(query)
		if not query_tokens:
			return []

		scores = self.index.get_scores(query_tokens)
		indexed_scores = sorted(
			enumerate(scores.tolist() if hasattr(scores, "tolist") else scores),
			key=lambda item: item[1],
			reverse=True,
		)

		results: List[Dict[str, str]] = []
		for rank, (idx, score) in enumerate(indexed_scores[: min(top_k, len(self.chunks))], start=1):
			item = dict(self.chunks[idx])
			item["score"] = float(score)
			item["bm25_score"] = float(score)
			item["retrieval"] = "bm25"
			item["bm25_rank"] = rank
			results.append(item)

		if debug:
			logger.debug("BM25 results for query '%s': %s", query, [
				{
					"chunk_id": item.get("chunk_id"),
					"document_id": item.get("document_id"),
					"score": item.get("score"),
				}
				for item in results
			])

		return results
