from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence


class MetadataRepository(ABC):
    @abstractmethod
    def insert_document(
        self,
        document_id: str,
        filename: str,
        upload_time: str,
        total_chunks: int = 0,
    ) -> None:
        """Insert or replace a document metadata record."""

    @abstractmethod
    def insert_chunks(self, chunks: Sequence[Dict[str, Any]]) -> None:
        """Insert or replace chunk metadata records."""

    @abstractmethod
    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Return a single document record, or None if missing."""

    @abstractmethod
    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Return all document records."""

    @abstractmethod
    def get_chunks_by_document(self, document_id: str) -> List[Dict[str, Any]]:
        """Return all chunk records for a document."""

    @abstractmethod
    def get_all_chunks(self) -> List[Dict[str, Any]]:
        """Return all chunk records."""

    @abstractmethod
    def delete_document(self, document_id: str) -> None:
        """Delete a document and any dependent chunks."""

    @abstractmethod
    def clear_database(self) -> None:
        """Remove all metadata from the repository."""
