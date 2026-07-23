"""Ingestion tests."""
from pathlib import Path

import pytest

from ingestion.chunker import chunk_markdown, get_chunk_params
from ingestion.web_loader import validate_web_url


def test_validate_web_url_accepts_https():
    assert validate_web_url("https://example.com/docs") == "https://example.com/docs"


def test_validate_web_url_rejects_file_scheme():
    with pytest.raises(ValueError):
        validate_web_url("file:///etc/passwd")


def test_chunk_markdown_splits_on_headings():
    text = "## Intro\nFirst point.\n\n## Details\nSecond point with more context."
    chunks = chunk_markdown(text, document_id="doc1", chunk_size=50, overlap=10)
    assert chunks
    assert any(chunk.get("section_title") == "Intro" for chunk in chunks)
    assert any(chunk.get("section_title") == "Details" for chunk in chunks)


def test_get_chunk_params_for_pdf_and_web():
    assert get_chunk_params(filename="notes.pdf") == (180, 60)
    assert get_chunk_params(source_type="url", filename="example.com_index.html") == (280, 80)
