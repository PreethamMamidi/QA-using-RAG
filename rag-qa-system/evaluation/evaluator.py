"""Retrieval evaluation orchestration."""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence

import faiss

from evaluation.datasets import EvaluationSample
from evaluation.metrics import (
	EvaluationSummary,
	LatencyBreakdown,
	QueryEvaluationResult,
	average,
	hit_rate_at_k,
	mean_reciprocal_rank,
	recall_at_k,
)
from retrieval.bm25_index import BM25Retriever
from retrieval.fusion import reciprocal_rank_fusion
from retrieval.hybrid_retrieval import HybridRetriever
from retrieval.reranker import rerank_chunks
from retrieval.retriever import retrieve_chunks


class RetrievalEvaluator:
	"""Evaluate hybrid retrieval quality and latency against labeled queries."""

	def __init__(
		self,
		index: faiss.Index,
		chunks: List[Dict[str, str]],
		*,
		use_reranker: bool = False,
		top_k_dense: int = 20,
		top_k_sparse: int = 20,
		top_k_fused: int = 10,
		rrf_k: int = 60,
		rerank_top_k: int = 5,
	) -> None:
		self.index = index
		self.chunks = chunks
		self.use_reranker = use_reranker
		self.top_k_dense = top_k_dense
		self.top_k_sparse = top_k_sparse
		self.top_k_fused = top_k_fused
		self.rrf_k = rrf_k
		self.rerank_top_k = rerank_top_k
		self._hybrid = HybridRetriever.from_chunks(index, chunks)

	def _resolve_relevant_ids(self, sample: EvaluationSample) -> set[str]:
		chunk_ids = sample.relevant_chunk_ids
		if chunk_ids:
			return set(chunk_ids)
		return set(sample.relevant_document_ids)

	def _extract_retrieved_ids(self, results: Sequence[Dict[str, str]], sample: EvaluationSample) -> List[str]:
		if sample.relevant_chunk_ids:
			return [str(item["chunk_id"]) for item in results if item.get("chunk_id")]
		return [str(item.get("document_id")) for item in results if item.get("document_id")]

	def retrieve_with_timings(self, query: str) -> tuple[List[Dict[str, str]], LatencyBreakdown]:
		"""Run hybrid retrieval with per-stage latency measurement."""
		latency = LatencyBreakdown()
		if not query or not query.strip():
			return [], latency

		total_start = time.perf_counter()

		bm25_start = time.perf_counter()
		sparse_results = self._hybrid.bm25_retriever.retrieve(query, top_k=self.top_k_sparse)
		latency.bm25_ms = (time.perf_counter() - bm25_start) * 1000

		faiss_start = time.perf_counter()
		dense_results = retrieve_chunks(
			query,
			self.index,
			self.chunks,
			top_k=self.top_k_dense,
		)
		latency.faiss_ms = (time.perf_counter() - faiss_start) * 1000

		fusion_start = time.perf_counter()
		fused_results = reciprocal_rank_fusion(
			[dense_results, sparse_results],
			k=self.rrf_k,
			top_k=self.top_k_fused,
		)
		latency.fusion_ms = (time.perf_counter() - fusion_start) * 1000

		final_results = fused_results
		if self.use_reranker:
			rerank_start = time.perf_counter()
			final_results = rerank_chunks(query, fused_results, top_k=self.rerank_top_k)
			latency.rerank_ms = (time.perf_counter() - rerank_start) * 1000

		latency.total_ms = (time.perf_counter() - total_start) * 1000
		return final_results, latency

	def evaluate_query(self, sample: EvaluationSample) -> QueryEvaluationResult:
		"""Evaluate retrieval for a single labeled query."""
		results, latency = self.retrieve_with_timings(sample.question)
		relevant_ids = self._resolve_relevant_ids(sample)
		retrieved_ids = self._extract_retrieved_ids(results, sample)

		return QueryEvaluationResult(
			question=sample.question,
			recall_at_5=recall_at_k(retrieved_ids, relevant_ids, 5),
			recall_at_10=recall_at_k(retrieved_ids, relevant_ids, 10),
			hit_rate_at_5=hit_rate_at_k(retrieved_ids, relevant_ids, 5),
			hit_rate_at_10=hit_rate_at_k(retrieved_ids, relevant_ids, 10),
			mrr=mean_reciprocal_rank(retrieved_ids, relevant_ids),
			latency=latency,
			retrieved_chunk_ids=retrieved_ids,
		)

	def evaluate(self, samples: Sequence[EvaluationSample]) -> EvaluationSummary:
		"""Evaluate all samples and aggregate metrics."""
		per_query = [self.evaluate_query(sample) for sample in samples]
		if not per_query:
			return EvaluationSummary(
				queries_evaluated=0,
				recall_at_5=0.0,
				recall_at_10=0.0,
				hit_rate_at_5=0.0,
				hit_rate_at_10=0.0,
				mrr=0.0,
			)

		avg_latency = LatencyBreakdown(
			bm25_ms=average(result.latency.bm25_ms for result in per_query),
			faiss_ms=average(result.latency.faiss_ms for result in per_query),
			fusion_ms=average(result.latency.fusion_ms for result in per_query),
			rerank_ms=average(result.latency.rerank_ms for result in per_query),
			total_ms=average(result.latency.total_ms for result in per_query),
		)

		return EvaluationSummary(
			queries_evaluated=len(per_query),
			recall_at_5=average(result.recall_at_5 for result in per_query),
			recall_at_10=average(result.recall_at_10 for result in per_query),
			hit_rate_at_5=average(result.hit_rate_at_5 for result in per_query),
			hit_rate_at_10=average(result.hit_rate_at_10 for result in per_query),
			mrr=average(result.mrr for result in per_query),
			average_latency=avg_latency,
			per_query=per_query,
		)
