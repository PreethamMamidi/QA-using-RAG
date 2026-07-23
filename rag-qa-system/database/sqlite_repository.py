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
                    total_chunks INTEGER DEFAULT 0,
                    source_type TEXT DEFAULT 'file',
                    source_url TEXT
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    page INTEGER,
                    chunk_index INTEGER,
                    text TEXT NOT NULL,
                    section_title TEXT,
                    FOREIGN KEY (document_id) REFERENCES documents(document_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_chunks_document_id
                    ON chunks(document_id);

                CREATE INDEX IF NOT EXISTS idx_chunks_page
                    ON chunks(page);
                """
            )
            self._migrate_schema(connection)

    def _migrate_schema(self, connection: sqlite3.Connection) -> None:
        document_columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(documents)").fetchall()
        }
        if "source_type" not in document_columns:
            connection.execute(
                "ALTER TABLE documents ADD COLUMN source_type TEXT DEFAULT 'file'"
            )
        if "source_url" not in document_columns:
            connection.execute("ALTER TABLE documents ADD COLUMN source_url TEXT")

        chunk_columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(chunks)").fetchall()
        }
        if "section_title" not in chunk_columns:
            connection.execute("ALTER TABLE chunks ADD COLUMN section_title TEXT")

    def insert_document(
        self,
        document_id: str,
        filename: str,
        upload_time: str,
        total_chunks: int = 0,
        source_type: str = "file",
        source_url: str | None = None,
    ) -> None:
        with self._get_connection() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO documents (
                    document_id,
                    filename,
                    upload_time,
                    total_chunks,
                    source_type,
                    source_url
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (document_id, filename, upload_time, total_chunks, source_type, source_url),
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
                chunk.get("section_title"),
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
                    text,
                    section_title
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                payload,
            )

    @staticmethod
    def _document_row(row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "document_id": row["document_id"],
            "filename": row["filename"],
            "upload_time": row["upload_time"],
            "total_chunks": row["total_chunks"],
            "source_type": row["source_type"] if "source_type" in row.keys() else "file",
            "source_url": row["source_url"] if "source_url" in row.keys() else None,
        }

    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as connection:
            row = connection.execute(
                """
                SELECT document_id, filename, upload_time, total_chunks, source_type, source_url
                FROM documents
                WHERE document_id = ?
                """,
                (document_id,),
            ).fetchone()
        return self._document_row(row) if row is not None else None

    def list_documents(self) -> List[Dict[str, Any]]:
        with self._get_connection() as connection:
            rows = connection.execute(
                """
                SELECT document_id, filename, upload_time, total_chunks, source_type, source_url
                FROM documents
                ORDER BY upload_time DESC, document_id ASC
                """
            ).fetchall()
        return [self._document_row(row) for row in rows]

    def document_exists(self, document_id: str) -> bool:
        with self._get_connection() as connection:
            row = connection.execute(
                "SELECT 1 FROM documents WHERE document_id = ? LIMIT 1",
                (document_id,),
            ).fetchone()
        return row is not None

    def get_document_count(self) -> int:
        with self._get_connection() as connection:
            row = connection.execute("SELECT COUNT(*) AS count FROM documents").fetchone()
        return int(row["count"]) if row is not None else 0

    def get_chunk_count(self) -> int:
        with self._get_connection() as connection:
            row = connection.execute("SELECT COUNT(*) AS count FROM chunks").fetchone()
        return int(row["count"]) if row is not None else 0

    def get_all_documents(self) -> List[Dict[str, Any]]:
        return self.list_documents()

    def get_chunks_by_document(self, document_id: str) -> List[Dict[str, Any]]:
        return self.get_chunks_by_documents([document_id])

    def get_chunks_by_documents(self, document_ids: Sequence[str]) -> List[Dict[str, Any]]:
        ids = [doc_id for doc_id in document_ids if doc_id]
        if not ids:
            return []

        placeholders = ", ".join("?" * len(ids))
        with self._get_connection() as connection:
            rows = connection.execute(
                f"""
                SELECT
                    chunks.chunk_id,
                    chunks.document_id,
                    documents.filename,
                    chunks.page,
                    chunks.chunk_index,
                    chunks.text,
                    chunks.section_title,
                    documents.source_type,
                    documents.source_url
                FROM chunks
                JOIN documents ON documents.document_id = chunks.document_id
                WHERE chunks.document_id IN ({placeholders})
                ORDER BY
                    chunks.document_id ASC,
                    COALESCE(chunks.page, 0) ASC,
                    COALESCE(chunks.chunk_index, 0) ASC,
                    chunks.chunk_id ASC
                """,
                ids,
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
                    chunks.text,
                    chunks.section_title,
                    documents.source_type,
                    documents.source_url
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
