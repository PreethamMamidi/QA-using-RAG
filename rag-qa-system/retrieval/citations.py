"""Source citation utilities for the retrieval pipeline."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

# Metadata fields preserved across retrieval stages.
CHUNK_METADATA_FIELDS: tuple[str, ...] = (
	"chunk_id",
	"document_id",
	"filename",
	"page",
	"chunk_index",
	"source_type",
	"source_url",
	"section_title",
	"text",
)

SCORE_FIELDS: frozenset[str] = frozenset(
	{
		"score",
		"dense_score",
		"bm25_score",
		"rrf_score",
		"rerank_score",
		"bm25_rank",
		"retrieval",
		"rrf_sources",
		"rrf_details",
	}
)


def preserve_chunk_metadata(chunk: Dict[str, Any]) -> Dict[str, Any]:
	"""Return a copy of a chunk dict with core metadata fields intact."""
	return dict(chunk or {})


def merge_chunk_metadata(base: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
	"""Merge metadata from an incoming chunk into a base chunk record."""
	merged = preserve_chunk_metadata(base)
	for key, value in (incoming or {}).items():
		if key in SCORE_FIELDS:
			continue
		if value is None or value == "":
			continue
		if key not in merged or merged.get(key) in (None, ""):
			merged[key] = value
	return merged


@dataclass(frozen=True)
class SourceCitation:
	"""Normalized source citation for UI and report export."""

	document_id: str
	filename: str
	page: Optional[int] = None
	source_type: str = "file"
	source_url: Optional[str] = None
	chunk_ids: tuple[str, ...] = field(default_factory=tuple)

	def label(self) -> str:
		"""Human-readable citation label."""
		name = self.filename or self.document_id
		if self.page is not None:
			try:
				page_num = int(self.page)
				if page_num > 0:
					return f"{name} (Page {page_num})"
			except (TypeError, ValueError):
				pass
		return name

	def to_dict(self) -> Dict[str, Any]:
		"""Serialize for JSON reports."""
		payload = asdict(self)
		payload["label"] = self.label()
		return payload


def _citation_key(chunk: Dict[str, Any]) -> tuple[str, Optional[int]]:
	filename = str(chunk.get("filename") or chunk.get("document_id") or "unknown")
	page = chunk.get("page")
	try:
		page_val = int(page) if page is not None else None
	except (TypeError, ValueError):
		page_val = None
	return filename, page_val


def citations_from_chunks(chunks: Sequence[Dict[str, Any]]) -> List[SourceCitation]:
	"""Build deduplicated source citations from retrieved chunks (rank order preserved)."""
	ordered: List[SourceCitation] = []
	seen: set[tuple[str, Optional[int]]] = set()
	chunk_ids_by_key: Dict[tuple[str, Optional[int]], List[str]] = {}

	for chunk in chunks or []:
		key = _citation_key(chunk)
		chunk_id = chunk.get("chunk_id")
		if chunk_id:
			chunk_ids_by_key.setdefault(key, [])
			if chunk_id not in chunk_ids_by_key[key]:
				chunk_ids_by_key[key].append(str(chunk_id))

		if key in seen:
			continue
		seen.add(key)
		ordered.append(
			SourceCitation(
				document_id=str(chunk.get("document_id") or "unknown"),
				filename=str(chunk.get("filename") or chunk.get("document_id") or "unknown"),
				page=chunk.get("page"),
				source_type=str(chunk.get("source_type") or "file"),
				source_url=chunk.get("source_url"),
				chunk_ids=tuple(),
			)
		)

	result: List[SourceCitation] = []
	for citation in ordered:
		key = (citation.filename, citation.page if isinstance(citation.page, int) else None)
		try:
			page_val = int(citation.page) if citation.page is not None else None
			key = (citation.filename, page_val)
		except (TypeError, ValueError):
			key = (citation.filename, None)
		result.append(
			SourceCitation(
				document_id=citation.document_id,
				filename=citation.filename,
				page=citation.page,
				source_type=citation.source_type,
				source_url=citation.source_url,
				chunk_ids=tuple(chunk_ids_by_key.get(key, ())),
			)
		)
	return result


def format_citations_markdown(citations: Sequence[SourceCitation]) -> str:
	"""Format citations as a markdown bullet list."""
	if not citations:
		return "_No sources available._"
	return "\n".join(f"- {citation.label()}" for citation in citations)


def citations_report_payload(
	chunks: Sequence[Dict[str, Any]],
	*,
	question: Optional[str] = None,
	answer: Optional[str] = None,
) -> Dict[str, Any]:
	"""Build a JSON-serializable citation report for export."""
	citations = citations_from_chunks(chunks)
	return {
		"question": question,
		"answer": answer,
		"sources": [citation.to_dict() for citation in citations],
		"chunks": [
			{
				field: chunk.get(field)
				for field in (*CHUNK_METADATA_FIELDS, "score", "retrieval")
				if chunk.get(field) is not None
			}
			for chunk in (chunks or [])
		],
	}
