"""
Document loading utilities for the RAG QA system.

Reads files from a directory (non-recursive) and returns normalized entries
via Docling + RapidOCR, with a fast path for plain TXT files.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

from ingestion.docling_converter import convert_path, is_supported_extension
from ingestion.types import LoadedDocument

logger = logging.getLogger(__name__)


def load_file(path: str | Path) -> List[LoadedDocument]:
	"""Load a single supported file."""
	return convert_path(Path(path))


def load_documents(data_dir: str) -> List[Dict[str, str]]:
	"""
	Load documents from a directory and return a list of entries.

	Returns dicts compatible with the existing pipeline (`document_id`, `text`,
	plus optional metadata fields).
	"""
	root = Path(data_dir)
	if not root.exists() or not root.is_dir():
		raise FileNotFoundError(f"Directory not found or not a directory: {data_dir}")

	entries: List[Dict[str, str]] = []
	skipped: List[str] = []

	for path in sorted(root.iterdir()):
		if not path.is_file():
			continue

		suffix = path.suffix.lower()
		if not is_supported_extension(suffix):
			skipped.append(path.name)
			continue

		try:
			loaded = convert_path(path)
			entries.extend(loaded)
		except Exception as exc:
			logger.exception("Failed to load %s: %s", path.name, exc)
			skipped.append(path.name)

	if skipped:
		logger.warning("Skipped unsupported or failed files: %s", ", ".join(skipped))

	return entries
