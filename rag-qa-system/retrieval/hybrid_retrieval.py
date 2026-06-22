"""Hybrid retrieval combining FAISS dense retrieval with BM25 sparse retrieval."""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import faiss

from retrieval.bm25_index import BM25Retriever
from retrieval.fusion import reciprocal_rank_fusion
from retrieval.retriever import materialize_chunk_records, retrieve_chunks


logger = logging.getLogger(__name__)


def _preview_rankings(results: List[Dict[str, str]]) -> List[Dict[str, str]]:
	"""Create a compact debug preview of ranked chunks."""
	preview: List[Dict[str, str]] = []
	for rank, item in enumerate(results, start=1):
		preview.append(
			{
				"rank": rank,
				"chunk_id": item.get("chunk_id"),
				"document_id": item.get("document_id"),
				"page": item.get("page"),
				"score": item.get("score"),
				"dense_score": item.get("dense_score"),
				"bm25_score": item.get("bm25_score"),
				"rrf_score": item.get("rrf_score"),
				"retrieval": item.get("retrieval"),
			}
		)
	return preview


def _log_rankings(label: str, results: List[Dict[str, str]]) -> None:
	logger.debug("%s: %s", label, _preview_rankings(results))


class HybridRetriever:
	"""Hybrid retriever that fuses dense and sparse search results once per corpus."""

	def __init__(
		self,
		index: faiss.Index,
		chunks: List[Dict[str, str]],
		bm25_retriever: Optional[BM25Retriever] = None,
	) -> None:
		self.index = index
		self.chunks = materialize_chunk_records(chunks)
		self.bm25_retriever = bm25_retriever or BM25Retriever(self.chunks)

	@classmethod
	def from_chunks(
		cls,
		index: faiss.Index,
		chunks: List[Dict[str, str]],
	) -> "HybridRetriever":
		"""Build a hybrid retriever from a FAISS index and chunk list."""
		return cls(index=index, chunks=chunks, bm25_retriever=BM25Retriever(chunks))

	def retrieve(
		self,
		query: str,
		top_k_dense: int = 10,
		top_k_sparse: int = 10,
		top_k_fused: int = 10,
		rrf_k: int = 60,
		debug: bool = False,
		return_debug: bool = False,
	) -> List[Dict[str, str]] | Tuple[List[Dict[str, str]], Dict[str, List[Dict[str, str]]]]:
		"""Retrieve chunks via dense, sparse, and RRF fusion.

		The returned chunk objects remain compatible with the existing reranker.
		"""
		if not query or not query.strip():
			return ([], {"dense_results": [], "sparse_results": [], "fused_results": []}) if return_debug else []

		dense_results = retrieve_chunks(query, self.index, self.chunks, top_k=top_k_dense)
		sparse_results = self.bm25_retriever.retrieve(query, top_k=top_k_sparse, debug=debug)
		fused_results = reciprocal_rank_fusion(
			[dense_results, sparse_results],
			k=rrf_k,
			top_k=top_k_fused,
			debug=debug,
		)

		# Preserve compatibility: downstream reranker expects a list of chunk dicts.
		if debug:
			_log_rankings("Dense retrieval", dense_results)
			_log_rankings("Sparse retrieval", sparse_results)
			_log_rankings("Fused retrieval", fused_results)

		if return_debug:
			return fused_results, {
				"dense_results": dense_results,
				"sparse_results": sparse_results,
				"fused_results": fused_results,
			}

		return fused_results


def hybrid_retrieve_chunks(
	query: str,
	index: faiss.Index,
	chunks: List[Dict[str, str]],
	bm25_retriever: Optional[BM25Retriever] = None,
	top_k_dense: int = 10,
	top_k_sparse: int = 10,
	top_k_fused: int = 10,
	rrf_k: int = 60,
	debug: bool = False,
	return_debug: bool = False,
) -> List[Dict[str, str]] | Tuple[List[Dict[str, str]], Dict[str, List[Dict[str, str]]]]:
	"""Convenience wrapper for hybrid retrieval."""
	retriever = HybridRetriever(index=index, chunks=chunks, bm25_retriever=bm25_retriever)
	return retriever.retrieve(
		query,
		top_k_dense=top_k_dense,
		top_k_sparse=top_k_sparse,
		top_k_fused=top_k_fused,
		rrf_k=rrf_k,
		debug=debug,
		return_debug=return_debug,
	)
