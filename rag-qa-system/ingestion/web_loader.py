"""Single-URL web page ingestion via Docling."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List
from urllib.parse import urlparse

from ingestion.docling_converter import convert_source
from ingestion.types import LoadedDocument

_BLOCKED_SCHEMES = {"file", "ftp", "javascript", "data"}


def validate_web_url(url: str) -> str:
	"""Validate and normalize a web URL for ingestion."""
	normalized = (url or "").strip()
	if not normalized:
		raise ValueError("URL is empty.")

	parsed = urlparse(normalized)
	if parsed.scheme in _BLOCKED_SCHEMES:
		raise ValueError(f"Unsupported URL scheme: {parsed.scheme}")
	if parsed.scheme not in {"http", "https"}:
		raise ValueError("Only http and https URLs are supported.")
	if not parsed.netloc:
		raise ValueError("URL must include a host name.")

	return normalized


def _display_name_from_url(url: str) -> str:
	parsed = urlparse(url)
	path = parsed.path.strip("/").replace("/", "_") or "index"
	safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", path)
	return f"{parsed.netloc}_{safe}.html"


def load_web_page(url: str) -> List[LoadedDocument]:
	"""Load a single web page by URL."""
	valid_url = validate_web_url(url)
	display_name = _display_name_from_url(valid_url)
	return convert_source(
		valid_url,
		source_type="url",
		filename=display_name,
		source_url=valid_url,
	)


def save_web_snapshot(url: str, output_dir: str | Path) -> Path:
	"""Persist URL metadata snapshot for auditability."""
	output_dir = Path(output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	valid_url = validate_web_url(url)
	snapshot_path = output_dir / f"{_display_name_from_url(valid_url)}.meta.json"
	snapshot_path.write_text(
		json.dumps({"url": valid_url, "filename": snapshot_path.stem}, indent=2),
		encoding="utf-8",
	)
	return snapshot_path
