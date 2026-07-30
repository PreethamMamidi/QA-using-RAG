"""Evaluation API wrappers."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from api.services.index_service import IndexService
from evaluation.datasets import load_evaluation_dataset
from evaluation.evaluator import RetrievalEvaluator


class EvaluationService:
	"""Run retrieval evaluation against the loaded knowledge base."""

	def __init__(self, index_service: IndexService, evaluation_dir: Path) -> None:
		self.index_service = index_service
		self.evaluation_dir = Path(evaluation_dir)

	def list_datasets(self) -> List[Dict[str, str]]:
		if not self.evaluation_dir.exists():
			return []
		datasets = []
		for path in sorted(self.evaluation_dir.glob("*.json")):
			# Skip non-dataset JSON if needed; gold/sample are the known ones.
			datasets.append(
				{
					"name": path.name,
					"path": str(path.resolve()),
				}
			)
		return datasets

	def run(
		self,
		dataset_path: str | Path,
		*,
		use_reranker: bool = False,
	) -> Dict[str, Any]:
		if not self.index_service.ready:
			raise RuntimeError("Knowledge base is not loaded. Ingest documents first.")

		path = Path(dataset_path)
		if not path.is_absolute():
			candidate = self.evaluation_dir / path
			path = candidate if candidate.exists() else Path(dataset_path)
		if not path.exists():
			raise FileNotFoundError(f"Evaluation dataset not found: {dataset_path}")

		samples = load_evaluation_dataset(path)
		evaluator = RetrievalEvaluator(
			self.index_service.index,
			self.index_service.chunks,
			use_reranker=use_reranker,
		)
		summary = evaluator.evaluate(samples)
		payload = summary.to_dict()
		payload["report"] = summary.format_report()
		return payload
