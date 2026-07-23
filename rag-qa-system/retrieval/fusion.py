"""Ranking fusion utilities for hybrid retrieval."""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence


from retrieval.citations import merge_chunk_metadata

logger = logging.getLogger(__name__)


def _chunk_key(item: Dict[str, str]) -> str:
	chunk_id = item.get("chunk_id")
	if chunk_id:
		return str(chunk_id)
	document_id = item.get("document_id", "unknown")
	text = item.get("text", "")
	return f"{document_id}::{text}"


def reciprocal_rank_fusion(
	result_sets: Sequence[Sequence[Dict[str, str]]],
	k: int = 60,
	top_k: Optional[int] = None,
	debug: bool = False,
) -> List[Dict[str, str]]:
	"""Fuse multiple ranked lists using Reciprocal Rank Fusion (RRF).

	Parameters
	----------
	result_sets : Sequence[Sequence[Dict[str, str]]]
		Ranked result lists from dense and sparse retrieval.
	k : int, optional
		RRF constant, by default 60.
	top_k : Optional[int], optional
		Maximum number of fused results to return.
	debug : bool, optional
		If True, emit debug logging.
	"""
	if k < 0:
		raise ValueError("RRF constant k must be non-negative.")

	aggregated: Dict[str, Dict[str, str]] = {}

	for source_index, results in enumerate(result_sets):
		for rank, item in enumerate(results or [], start=1):
			key = _chunk_key(item)
			contribution = 1.0 / (k + rank)

			if key not in aggregated:
				aggregated[key] = merge_chunk_metadata({}, item)
				aggregated[key]["rrf_score"] = 0.0
				aggregated[key]["score"] = 0.0
				aggregated[key]["rrf_sources"] = []
				aggregated[key]["rrf_details"] = []

			entry = aggregated[key]
			entry = merge_chunk_metadata(entry, item)
			aggregated[key] = entry
			entry["rrf_score"] = float(entry.get("rrf_score", 0.0) + contribution)
			entry["score"] = float(entry["rrf_score"])
			entry["rrf_details"].append(
				{
					"source_index": source_index,
					"rank": rank,
					"contribution": contribution,
				}
			)

			retrieval_name = item.get("retrieval")
			if retrieval_name and retrieval_name not in entry["rrf_sources"]:
				entry["rrf_sources"].append(retrieval_name)

			entry["retrieval"] = "rrf"

	results = sorted(aggregated.values(), key=lambda item: item.get("rrf_score", 0.0), reverse=True)
	if top_k is not None:
		results = results[:top_k]

	if debug:
		logger.debug("RRF fused results: %s", [
			{
				"chunk_id": item.get("chunk_id"),
				"document_id": item.get("document_id"),
				"score": item.get("score"),
				"sources": item.get("rrf_sources"),
			}
			for item in results
		])

	return results
