from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from database.repository import MetadataRepository


class SQLiteMetadataRepository(MetadataRepository):
    def __init__(self, db_path: str | Path | None = None) -> None:
        default_path = Path(__file__).resolve().parent.parent / "storage" / "metadata.db"
        self.db_path = Path(db_path) if db_path is not None else default_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()

    def _get_connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode = MEMORY")
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize_database(self) -> None:
        with self._get_connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    upload_time TEXT NOT NULL,
                    total_chunks INTEGER DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    page INTEGER,
                    chunk_index INTEGER,
                    text TEXT NOT NULL,
                    FOREIGN KEY (document_id) REFERENCES documents(document_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_chunks_document_id
                    ON chunks(document_id);

                CREATE INDEX IF NOT EXISTS idx_chunks_page
                    ON chunks(page);
                """
            )

    def insert_document(
        self,
        document_id: str,
        filename: str,
        upload_time: str,
        total_chunks: int = 0,
    ) -> None:
        with self._get_connection() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO documents (
                    document_id,
                    filename,
                    upload_time,
                    total_chunks
                ) VALUES (?, ?, ?, ?)
                """,
                (document_id, filename, upload_time, total_chunks),
            )

    def insert_chunks(self, chunks: Sequence[Dict[str, Any]]) -> None:
        if not chunks:
            return

        payload = [
            (
                chunk["chunk_id"],
                chunk["document_id"],
                chunk.get("page"),
                chunk.get("chunk_index"),
                chunk["text"],
            )
            for chunk in chunks
        ]

        with self._get_connection() as connection:
            connection.executemany(
                """
                INSERT OR REPLACE INTO chunks (
                    chunk_id,
                    document_id,
                    page,
                    chunk_index,
                    text
                ) VALUES (?, ?, ?, ?, ?)
                """,
                payload,
            )

    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as connection:
            row = connection.execute(
                """
                SELECT document_id, filename, upload_time, total_chunks
                FROM documents
                WHERE document_id = ?
                """,
                (document_id,),
            ).fetchone()
        return dict(row) if row is not None else None

    def get_all_documents(self) -> List[Dict[str, Any]]:
        with self._get_connection() as connection:
            rows = connection.execute(
                """
                SELECT document_id, filename, upload_time, total_chunks
                FROM documents
                ORDER BY upload_time DESC, document_id ASC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def get_chunks_by_document(self, document_id: str) -> List[Dict[str, Any]]:
        with self._get_connection() as connection:
            rows = connection.execute(
                """
                SELECT
                    chunks.chunk_id,
                    chunks.document_id,
                    documents.filename,
                    chunks.page,
                    chunks.chunk_index,
                    chunks.text
                FROM chunks
                JOIN documents ON documents.document_id = chunks.document_id
                WHERE chunks.document_id = ?
                ORDER BY
                    COALESCE(chunks.page, 0) ASC,
                    COALESCE(chunks.chunk_index, 0) ASC,
                    chunks.chunk_id ASC
                """,
                (document_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_all_chunks(self) -> List[Dict[str, Any]]:
        with self._get_connection() as connection:
            rows = connection.execute(
                """
                SELECT
                    chunks.chunk_id,
                    chunks.document_id,
                    documents.filename,
                    chunks.page,
                    chunks.chunk_index,
                    chunks.text
                FROM chunks
                JOIN documents ON documents.document_id = chunks.document_id
                ORDER BY
                    chunks.document_id ASC,
                    COALESCE(chunks.page, 0) ASC,
                    COALESCE(chunks.chunk_index, 0) ASC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def delete_document(self, document_id: str) -> None:
        with self._get_connection() as connection:
            connection.execute(
                "DELETE FROM documents WHERE document_id = ?",
                (document_id,),
            )

    def clear_database(self) -> None:
        with self._get_connection() as connection:
            connection.execute("DELETE FROM chunks")
            connection.execute("DELETE FROM documents")
