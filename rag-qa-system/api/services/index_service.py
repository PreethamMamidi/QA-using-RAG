"""Process-wide FAISS + chunk + HybridRetriever holder."""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss

from database.sqlite_repository import SQLiteMetadataRepository
from retrieval.hybrid_retrieval import HybridRetriever

logger = logging.getLogger(__name__)


class IndexService:
	"""In-memory knowledge-base runtime state shared by API handlers."""

	def __init__(self, storage_dir: Path, metadata_db_path: Path) -> None:
		self.storage_dir = Path(storage_dir)
		self.metadata_db_path = Path(metadata_db_path)
		self.faiss_index_path = self.storage_dir / "faiss.index"
		self.raw_docs_dir = self.storage_dir / "raw_docs"

		self.index: Optional[faiss.Index] = None
		self.chunks: List[Dict[str, Any]] = []
		self.documents: List[Dict[str, Any]] = []
		self.hybrid_retriever: Optional[HybridRetriever] = None
		self.write_lock = threading.Lock()

	@property
	def ready(self) -> bool:
		return self.index is not None and bool(self.chunks)

	def repository(self) -> SQLiteMetadataRepository:
		return SQLiteMetadataRepository(self.metadata_db_path)

	def load_from_disk(self) -> bool:
		"""Load FAISS + SQLite metadata into memory. Returns True if loaded."""
		if not self.faiss_index_path.exists() or not self.metadata_db_path.exists():
			self.unload()
			return False

		repo = self.repository()
		self.documents = repo.list_documents()
		self.chunks = repo.get_all_chunks()
		self.index = faiss.read_index(str(self.faiss_index_path))
		self.refresh_hybrid_retriever()
		logger.info(
			"Loaded KB from disk: %d documents, %d chunks, ntotal=%s",
			len(self.documents),
			len(self.chunks),
			getattr(self.index, "ntotal", None),
		)
		return True

	def set_runtime(
		self,
		*,
		index: faiss.Index,
		chunks: List[Dict[str, Any]],
		documents: Optional[List[Dict[str, Any]]] = None,
	) -> None:
		"""Replace in-memory index state after a rebuild."""
		self.index = index
		self.chunks = list(chunks or [])
		if documents is not None:
			self.documents = list(documents)
		else:
			self.documents = self.repository().list_documents()
		self.refresh_hybrid_retriever()

	def refresh_hybrid_retriever(self) -> None:
		if self.index is None or not self.chunks:
			self.hybrid_retriever = None
			return
		self.hybrid_retriever = HybridRetriever.from_chunks(self.index, self.chunks)

	def unload(self) -> None:
		self.index = None
		self.chunks = []
		self.documents = []
		self.hybrid_retriever = None

	def stats(self) -> Dict[str, Any]:
		repo = self.repository()
		try:
			doc_count = repo.get_document_count()
			chunk_count = repo.get_chunk_count()
		except Exception:
			doc_count = len(self.documents)
			chunk_count = len(self.chunks)
		return {
			"documents": doc_count,
			"chunks": chunk_count,
			"index_loaded": self.index is not None,
			"hybrid_ready": self.hybrid_retriever is not None,
			"faiss_ntotal": int(getattr(self.index, "ntotal", 0) or 0) if self.index else 0,
			"storage_dir": str(self.storage_dir),
		}
