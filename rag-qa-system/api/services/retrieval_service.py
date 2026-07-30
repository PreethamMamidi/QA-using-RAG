"""Retrieval orchestration for the API (mirrors Streamlit chat retrieval)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from api.services.index_service import IndexService
from retrieval.query_rewrite import rewrite_query_groq
from retrieval.reranker import rerank_chunks
from retrieval.retriever import retrieve_chunks


@dataclass
class RetrievalResult:
	chunks: List[Dict[str, Any]]
	rewritten_query: str
	debug: Optional[Dict[str, Any]] = None
	error: Optional[str] = None
	active_filters: Optional[List[str]] = None


class RetrievalService:
	"""Hybrid / dense retrieval with optional document filters and reranking."""

	def __init__(self, index_service: IndexService) -> None:
		self.index_service = index_service

	def resolve_scope(
		self,
		document_ids: Optional[Sequence[str]],
	) -> tuple[Optional[List[Dict[str, Any]]], Optional[List[int]], List[str], Optional[str]]:
		"""Return filtered chunks, candidate FAISS indices, labels, or an error."""
		if document_ids is None:
			return None, None, [], None
		ids = [doc_id for doc_id in document_ids if doc_id]
		if not ids:
			return [], [], [], "Select at least one document to apply a document filter."

		repo = self.index_service.repository()
		sqlite_chunks = repo.get_chunks_by_documents(ids)
		all_chunks = self.index_service.chunks
		chunk_id_to_index = {
			chunk["chunk_id"]: idx for idx, chunk in enumerate(all_chunks or [])
		}

		candidate_indices: List[int] = []
		retrieval_chunks: List[Dict[str, Any]] = []
		for chunk in sqlite_chunks:
			idx = chunk_id_to_index.get(chunk["chunk_id"])
			if idx is not None:
				candidate_indices.append(idx)
				retrieval_chunks.append(all_chunks[idx])

		documents = repo.list_documents()
		filename_by_id = {doc["document_id"]: doc["filename"] for doc in documents}
		active_labels = [filename_by_id.get(document_id, document_id) for document_id in ids]
		return retrieval_chunks, candidate_indices, active_labels, None

	def retrieve(
		self,
		query: str,
		*,
		rewrite_mode: str = "general",
		enable_hybrid: bool = True,
		use_reranker: bool = False,
		top_k_dense: int = 8,
		top_k_sparse: int = 8,
		top_k_fused: int = 8,
		rrf_k: int = 60,
		document_ids: Optional[Sequence[str]] = None,
		return_debug: bool = False,
	) -> RetrievalResult:
		if not self.index_service.ready:
			return RetrievalResult(
				chunks=[],
				rewritten_query=query,
				error="Knowledge base is not loaded. Ingest documents first.",
			)

		filtered_chunks, candidate_indices, labels, scope_error = self.resolve_scope(document_ids)
		if scope_error:
			return RetrievalResult(chunks=[], rewritten_query=query, error=scope_error)

		rewritten_query = rewrite_query_groq(query, mode=rewrite_mode) or query
		retrieval_query = rewritten_query
		retrieval_debug: Optional[Dict[str, Any]] = None

		retrieval_kwargs: Dict[str, Any] = {}
		if filtered_chunks is not None:
			retrieval_kwargs = {
				"chunks": filtered_chunks,
				"candidate_indices": candidate_indices,
			}

		if enable_hybrid:
			if self.index_service.hybrid_retriever is None:
				self.index_service.refresh_hybrid_retriever()
			hybrid = self.index_service.hybrid_retriever
			if hybrid is None:
				return RetrievalResult(
					chunks=[],
					rewritten_query=rewritten_query,
					error="Hybrid retriever is not available.",
				)
			if return_debug:
				initial, retrieval_debug = hybrid.retrieve(
					retrieval_query,
					top_k_dense=top_k_dense,
					top_k_sparse=top_k_sparse,
					top_k_fused=top_k_fused,
					rrf_k=rrf_k,
					debug=True,
					return_debug=True,
					**retrieval_kwargs,
				)
			else:
				initial = hybrid.retrieve(
					retrieval_query,
					top_k_dense=top_k_dense,
					top_k_sparse=top_k_sparse,
					top_k_fused=top_k_fused,
					rrf_k=rrf_k,
					debug=False,
					return_debug=False,
					**retrieval_kwargs,
				)
		else:
			initial = retrieve_chunks(
				retrieval_query,
				self.index_service.index,
				self.index_service.chunks,
				top_k=top_k_dense,
				candidate_indices=candidate_indices if filtered_chunks is not None else None,
			)
			if return_debug:
				retrieval_debug = {
					"dense_results": initial,
					"sparse_results": [],
					"fused_results": initial,
				}

		retrieved = rerank_chunks(query, initial, top_k=5) if use_reranker else initial[:8]
		if return_debug and retrieval_debug is not None:
			retrieval_debug["reranked_results"] = retrieved
			if labels:
				retrieval_debug["active_filters"] = labels

		return RetrievalResult(
			chunks=retrieved,
			rewritten_query=rewritten_query,
			debug=retrieval_debug,
			active_filters=labels or None,
		)
