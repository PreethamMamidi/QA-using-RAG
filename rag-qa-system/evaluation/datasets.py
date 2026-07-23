"""Evaluation dataset loading utilities."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, List, Sequence, Set


@dataclass(frozen=True)
class EvaluationSample:
	"""Single retrieval evaluation example."""

	question: str
	relevant_document_ids: Set[str] = field(default_factory=set)
	relevant_chunk_ids: Set[str] = field(default_factory=set)

	@classmethod
	def from_dict(cls, payload: dict[str, Any]) -> "EvaluationSample":
		"""Parse a sample from JSON, supporting legacy field names."""
		question = str(payload.get("question", "")).strip()
		if not question:
			raise ValueError("Evaluation sample missing 'question'.")

		doc_ids: Set[str] = set()
		for key in ("relevant_document_ids", "relevant_docs", "relevant_documents"):
			values = payload.get(key)
			if values:
				doc_ids = {str(value) for value in values}

		chunk_ids: Set[str] = set()
		raw_chunks = payload.get("relevant_chunk_ids") or payload.get("relevant_chunks")
		if raw_chunks:
			chunk_ids = {str(value) for value in raw_chunks}

		return cls(
			question=question,
			relevant_document_ids=doc_ids,
			relevant_chunk_ids=chunk_ids,
		)

	def relevant_ids(self, *, prefer_chunks: bool = True) -> Set[str]:
		"""Return chunk IDs when available, otherwise document IDs."""
		if prefer_chunks and self.relevant_chunk_ids:
			return set(self.relevant_chunk_ids)
		return set(self.relevant_document_ids)


def load_evaluation_dataset(path: str | Path) -> List[EvaluationSample]:
	"""Load evaluation samples from a JSON file."""
	dataset_path = Path(path)
	if not dataset_path.exists():
		raise FileNotFoundError(f"Evaluation dataset not found: {dataset_path}")

	with dataset_path.open("r", encoding="utf-8") as handle:
		payload = json.load(handle)

	if not isinstance(payload, list):
		raise ValueError("Evaluation dataset must be a JSON array.")

	return [EvaluationSample.from_dict(item) for item in payload]


def save_evaluation_dataset(samples: Sequence[EvaluationSample], path: str | Path) -> None:
	"""Persist evaluation samples to JSON."""
	dataset_path = Path(path)
	records = [
		{
			"question": sample.question,
			"relevant_document_ids": sorted(sample.relevant_document_ids),
			"relevant_chunk_ids": sorted(sample.relevant_chunk_ids),
		}
		for sample in samples
	]
	dataset_path.parent.mkdir(parents=True, exist_ok=True)
	with dataset_path.open("w", encoding="utf-8") as handle:
		json.dump(records, handle, indent=2)
