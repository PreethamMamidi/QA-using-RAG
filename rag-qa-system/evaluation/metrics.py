"""Retrieval evaluation metrics."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Sequence, Set


def recall_at_k(retrieved_ids: Sequence[str], relevant_ids: Set[str], k: int) -> float:
	"""Recall@K = |relevant ∩ top-K| / |relevant|."""
	if k <= 0 or not relevant_ids:
		return 0.0
	top_k = set(retrieved_ids[:k])
	hits = len(relevant_ids & top_k)
	return hits / len(relevant_ids)


def hit_rate_at_k(retrieved_ids: Sequence[str], relevant_ids: Set[str], k: int) -> float:
	"""HitRate@K = 1 if any relevant item appears in top-K, else 0."""
	if k <= 0 or not relevant_ids:
		return 0.0
	top_k = set(retrieved_ids[:k])
	return 1.0 if relevant_ids & top_k else 0.0


def mean_reciprocal_rank(retrieved_ids: Sequence[str], relevant_ids: Set[str]) -> float:
	"""MRR = 1 / rank of first relevant item (0 if none found)."""
	if not relevant_ids:
		return 0.0
	for rank, item_id in enumerate(retrieved_ids, start=1):
		if item_id in relevant_ids:
			return 1.0 / rank
	return 0.0


def average(values: Iterable[float]) -> float:
	"""Compute arithmetic mean; returns 0.0 for empty input."""
	items = list(values)
	if not items:
		return 0.0
	return sum(items) / len(items)


@dataclass
class LatencyBreakdown:
	"""Per-stage retrieval latency in milliseconds."""

	bm25_ms: float = 0.0
	faiss_ms: float = 0.0
	fusion_ms: float = 0.0
	rerank_ms: float = 0.0
	total_ms: float = 0.0

	def to_dict(self) -> dict[str, float]:
		return {
			"bm25_ms": self.bm25_ms,
			"faiss_ms": self.faiss_ms,
			"fusion_ms": self.fusion_ms,
			"rerank_ms": self.rerank_ms,
			"total_ms": self.total_ms,
		}


@dataclass
class QueryEvaluationResult:
	"""Metrics for a single evaluation query."""

	question: str
	recall_at_5: float
	recall_at_10: float
	hit_rate_at_5: float
	hit_rate_at_10: float
	mrr: float
	latency: LatencyBreakdown = field(default_factory=LatencyBreakdown)
	retrieved_chunk_ids: List[str] = field(default_factory=list)


@dataclass
class EvaluationSummary:
	"""Aggregated retrieval evaluation report."""

	queries_evaluated: int
	recall_at_5: float
	recall_at_10: float
	hit_rate_at_5: float
	hit_rate_at_10: float
	mrr: float
	average_latency: LatencyBreakdown = field(default_factory=LatencyBreakdown)
	per_query: List[QueryEvaluationResult] = field(default_factory=list)

	def format_report(self) -> str:
		"""Render a human-readable evaluation report."""
		lines = [
			"=================================",
			"Retrieval Evaluation Report",
			"=================================",
			"",
			f"Queries Evaluated: {self.queries_evaluated}",
			"",
			f"Recall@5:  {self.recall_at_5:.2f}",
			f"Recall@10: {self.recall_at_10:.2f}",
			"",
			f"HitRate@5:  {self.hit_rate_at_5:.2f}",
			f"HitRate@10: {self.hit_rate_at_10:.2f}",
			"",
			f"MRR: {self.mrr:.2f}",
			"",
			"Average Latencies:",
			f"BM25:    {self.average_latency.bm25_ms:.0f} ms",
			f"FAISS:   {self.average_latency.faiss_ms:.0f} ms",
			f"Fusion:  {self.average_latency.fusion_ms:.0f} ms",
			f"Reranker:{self.average_latency.rerank_ms:.0f} ms",
			"",
			f"Total Retrieval: {self.average_latency.total_ms:.0f} ms",
		]
		return "\n".join(lines)

	def to_dict(self) -> dict:
		return {
			"queries_evaluated": self.queries_evaluated,
			"recall_at_5": self.recall_at_5,
			"recall_at_10": self.recall_at_10,
			"hit_rate_at_5": self.hit_rate_at_5,
			"hit_rate_at_10": self.hit_rate_at_10,
			"mrr": self.mrr,
			"average_latency_ms": self.average_latency.to_dict(),
		}
