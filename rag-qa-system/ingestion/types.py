"""Shared types for document ingestion."""
from __future__ import annotations

from typing import Literal, TypedDict

SourceType = Literal["file", "url", "image"]


class LoadedDocument(TypedDict, total=False):
	"""Normalized document record produced by loaders."""

	document_id: str
	text: str
	page: int | None
	source_type: SourceType
	source_url: str | None
	filename: str
	section_title: str | None
	parser: str
