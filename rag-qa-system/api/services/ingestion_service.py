"""Ingestion orchestration extracted from Streamlit (full KB rebuild V1)."""
from __future__ import annotations

import datetime
import logging
import os
import re
import shutil
import stat
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import faiss

from api.services.index_service import IndexService
from embeddings.embedder import embed_texts
from ingestion.chunker import chunk_markdown, chunk_text, get_chunk_params
from ingestion.cleaner import clean_text
from ingestion.docling_converter import IMAGE_EXTENSIONS
from ingestion.loader import load_documents
from ingestion.web_loader import load_web_page, save_web_snapshot, validate_web_url
from vector_store.faiss_index import build_index

logger = logging.getLogger(__name__)


def _slugify_document_name(filename: str) -> str:
	stem = os.path.splitext(filename)[0].lower()
	slug = re.sub(r"[^a-z0-9]+", "", stem)
	return slug or "document"


def _split_source_document_id(source_document_id: str) -> tuple[str, int | None]:
	match = re.match(r"^(.*)_page_(\d+)$", source_document_id)
	if not match:
		return source_document_id, None
	return match.group(1), int(match.group(2))


def _extract_chunk_index(chunk: Dict[str, Any]) -> Optional[int]:
	chunk_id = chunk.get("chunk_id", "")
	match = re.search(r"_c(\d+)$", chunk_id)
	if match:
		return int(match.group(1))
	return None


def _remove_tree_safely(path: Path) -> bool:
	def _handle_remove_error(func, target_path, exc_info):
		try:
			os.chmod(target_path, stat.S_IWRITE)
			func(target_path)
		except OSError:
			raise exc_info[1]

	try:
		shutil.rmtree(path, onerror=_handle_remove_error)
		return True
	except OSError as exc:
		logger.exception("Failed to remove directory tree %s: %s", path, exc)
		return False


class IngestionService:
	"""Full-rebuild ingestion matching Streamlit V1 semantics."""

	def __init__(self, index_service: IndexService) -> None:
		self.index_service = index_service

	def build_schema(
		self,
		sources: Sequence[Dict[str, Any]],
		docs: Sequence[Dict[str, Any]],
	) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
		uploaded_at = datetime.date.today().isoformat()
		docs_by_filename: Dict[str, List[Tuple[Optional[int], Dict[str, Any]]]] = {}
		for item in docs:
			key = item.get("filename") or _split_source_document_id(item["document_id"])[0]
			_, page_number = _split_source_document_id(item["document_id"])
			if page_number is None:
				page_number = item.get("page")
			docs_by_filename.setdefault(str(key), []).append((page_number, item))

		used_ids: set[str] = set()
		documents: List[Dict[str, Any]] = []
		chunks: List[Dict[str, Any]] = []

		for source in sources:
			filename = source["filename"]
			source_type = source.get("source_type", "file")
			source_url = source.get("source_url")
			base_id = _slugify_document_name(filename)
			document_id = base_id
			suffix = 2
			while document_id in used_ids:
				document_id = f"{base_id}_{suffix}"
				suffix += 1
			used_ids.add(document_id)

			source_pages = sorted(
				docs_by_filename.get(filename, []),
				key=lambda pair: pair[0] if pair[0] is not None else 0,
			)
			total_pages = len(source_pages) if source_pages else 1
			documents.append(
				{
					"document_id": document_id,
					"filename": filename,
					"uploaded_at": uploaded_at,
					"total_pages": total_pages,
					"source_type": source_type,
					"source_url": source_url,
				}
			)

			if not source_pages:
				continue

			chunk_size, overlap = get_chunk_params(source_type, filename)
			use_markdown = source_type == "url" or filename.lower().endswith(
				(".md", ".markdown", ".docx", ".html", ".htm")
			)

			for page_number, source_doc in source_pages:
				page_label = page_number if page_number is not None else 1
				cleaned = clean_text(source_doc["text"])
				if use_markdown:
					page_chunks = chunk_markdown(
						cleaned,
						document_id=document_id,
						chunk_size=chunk_size,
						overlap=overlap,
					)
				else:
					page_chunks = chunk_text(
						cleaned,
						document_id=document_id,
						chunk_size=chunk_size,
						overlap=overlap,
					)
				for chunk_index, chunk in enumerate(page_chunks):
					chunk["chunk_id"] = f"{document_id}_p{page_label}_c{chunk_index}"
					chunk["document_id"] = document_id
					chunk["filename"] = filename
					chunk["page"] = page_label
					chunks.append(chunk)

		return documents, chunks

	def persist_metadata(
		self,
		document_records: Sequence[Dict[str, Any]],
		chunks: Sequence[Dict[str, Any]],
	) -> None:
		repository = self.index_service.repository()
		repository.clear_database()

		chunk_totals: Dict[str, int] = {}
		for chunk in chunks:
			document_id = chunk.get("document_id")
			if not document_id:
				continue
			chunk_totals[document_id] = chunk_totals.get(document_id, 0) + 1

		for document in document_records:
			repository.insert_document(
				document_id=document["document_id"],
				filename=document["filename"],
				upload_time=document["uploaded_at"],
				total_chunks=chunk_totals.get(document["document_id"], 0),
				source_type=document.get("source_type", "file"),
				source_url=document.get("source_url"),
			)

		repository.insert_chunks(
			[
				{
					"chunk_id": chunk["chunk_id"],
					"document_id": chunk["document_id"],
					"page": chunk.get("page"),
					"chunk_index": _extract_chunk_index(chunk),
					"text": chunk["text"],
					"section_title": chunk.get("section_title"),
				}
				for chunk in chunks
			]
		)

	def ingest_and_rebuild(
		self,
		sources: Sequence[Dict[str, Any]],
		docs: Sequence[Dict[str, Any]],
		*,
		stats_label: str = "ingest",
	) -> Dict[str, Any]:
		"""Chunk, embed, persist FAISS+SQLite, refresh runtime index (full rebuild)."""
		if not docs:
			raise ValueError("No text could be extracted from the provided source(s).")

		document_records, all_chunks = self.build_schema(sources, docs)
		if not all_chunks:
			raise ValueError("No chunks were created. The source may be empty or OCR failed.")

		with self.index_service.write_lock:
			embeddings = embed_texts([c["text"] for c in all_chunks])
			index = build_index(embeddings)

			self.index_service.storage_dir.mkdir(parents=True, exist_ok=True)
			faiss.write_index(index, str(self.index_service.faiss_index_path))
			self.persist_metadata(document_records, all_chunks)

			documents, chunks = (
				self.index_service.repository().list_documents(),
				self.index_service.repository().get_all_chunks(),
			)
			self.index_service.set_runtime(index=index, chunks=chunks, documents=documents)

		return {
			"stats_label": stats_label,
			"sources_processed": len(sources),
			"docs_loaded": len(docs),
			"chunks_created": len(all_chunks),
			"documents": len(documents),
		}

	def ingest_files(
		self,
		files: Sequence[Tuple[str, bytes]],
		*,
		replace: bool = True,
	) -> Dict[str, Any]:
		"""Ingest uploaded filename/bytes pairs and rebuild the KB.

		When ``replace`` is True (V1 default), ``raw_docs`` is wiped before writing
		the new uploads so disk state matches the rebuilt index.
		"""
		if not files:
			raise ValueError("No files provided.")

		raw_dir = self.index_service.raw_docs_dir
		if replace and raw_dir.exists():
			_remove_tree_safely(raw_dir)
		raw_dir.mkdir(parents=True, exist_ok=True)

		with tempfile.TemporaryDirectory() as tmpdir:
			sources: List[Dict[str, Any]] = []
			for filename, data in files:
				safe_name = Path(filename).name
				tmp_path = Path(tmpdir) / safe_name
				tmp_path.write_bytes(data)
				(raw_dir / safe_name).write_bytes(data)
				suffix = Path(safe_name).suffix.lower()
				sources.append(
					{
						"filename": safe_name,
						"source_type": "image" if suffix in IMAGE_EXTENSIONS else "file",
						"source_url": None,
					}
				)
			docs = load_documents(tmpdir)

		return self.ingest_and_rebuild(
			sources,
			docs,
			stats_label=f"{len(files)} file(s)",
		)

	def ingest_url(self, url: str) -> Dict[str, Any]:
		"""Ingest a single web page and rebuild the KB."""
		valid = validate_web_url(url)
		docs = load_web_page(valid)
		web_dir = self.index_service.raw_docs_dir / "web"
		save_web_snapshot(valid, web_dir)
		filename = docs[0].get("filename") if docs else "web_page.html"
		sources = [
			{
				"filename": filename,
				"source_type": "url",
				"source_url": valid,
			}
		]
		return self.ingest_and_rebuild(sources, docs, stats_label="1 web page")

	def reset_knowledge_base(self) -> Dict[str, Any]:
		"""Clear SQLite, FAISS, raw_docs, and unload in-memory index."""
		with self.index_service.write_lock:
			try:
				self.index_service.repository().clear_database()
			except Exception as exc:
				logger.exception("Failed to clear SQLite during reset: %s", exc)

			if self.index_service.faiss_index_path.exists():
				self.index_service.faiss_index_path.unlink()

			if self.index_service.raw_docs_dir.exists():
				_remove_tree_safely(self.index_service.raw_docs_dir)

			for legacy in ("chunks.json", "documents.json"):
				legacy_path = self.index_service.storage_dir / legacy
				if legacy_path.exists():
					legacy_path.unlink()

			self.index_service.unload()

		return {"reset": True, **self.index_service.stats()}
