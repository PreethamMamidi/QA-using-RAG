"""CLI runner for retrieval evaluation."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
	sys.path.insert(0, str(ROOT_DIR))

import faiss

from database.sqlite_repository import SQLiteMetadataRepository
from evaluation.datasets import load_evaluation_dataset
from evaluation.evaluator import RetrievalEvaluator


def _load_index_and_chunks(storage_dir: Path):
	index_path = storage_dir / "faiss.index"
	db_path = storage_dir / "metadata.db"
	if not index_path.exists() or not db_path.exists():
		raise FileNotFoundError(
			f"Missing index or metadata in {storage_dir}. Process documents first."
		)

	repository = SQLiteMetadataRepository(db_path)
	chunks = repository.get_all_chunks()
	index = faiss.read_index(str(index_path))
	return index, chunks


def main() -> None:
	parser = argparse.ArgumentParser(description="Run retrieval evaluation.")
	parser.add_argument(
		"--dataset",
		default=str(ROOT_DIR / "evaluation" / "gold.json"),
		help="Path to evaluation JSON dataset.",
	)
	parser.add_argument(
		"--storage",
		default=str(ROOT_DIR / "storage"),
		help="Path to storage directory containing faiss.index and metadata.db.",
	)
	parser.add_argument("--reranker", action="store_true", help="Enable cross-encoder reranking.")
	args = parser.parse_args()

	samples = load_evaluation_dataset(args.dataset)
	index, chunks = _load_index_and_chunks(Path(args.storage))

	evaluator = RetrievalEvaluator(index, chunks, use_reranker=args.reranker)
	summary = evaluator.evaluate(samples)
	print(summary.format_report())


if __name__ == "__main__":
	main()
