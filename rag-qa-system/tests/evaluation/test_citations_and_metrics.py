"""Tests for citations and evaluation metrics."""
from retrieval.citations import SourceCitation, citations_from_chunks, format_citations_markdown
from evaluation.metrics import hit_rate_at_k, mean_reciprocal_rank, recall_at_k


def test_citations_deduplicate_by_filename_and_page():
	chunks = [
		{"chunk_id": "a_c0", "document_id": "doc1", "filename": "Book.pdf", "page": 12, "score": 0.9},
		{"chunk_id": "a_c1", "document_id": "doc1", "filename": "Book.pdf", "page": 12, "score": 0.8},
		{"chunk_id": "b_c0", "document_id": "doc2", "filename": "Notes.docx", "score": 0.7},
	]
	citations = citations_from_chunks(chunks)
	assert len(citations) == 2
	assert citations[0].label() == "Book.pdf (Page 12)"
	assert citations[1].label() == "Notes.docx"
	assert set(citations[0].chunk_ids) == {"a_c0", "a_c1"}


def test_citation_without_page_uses_filename_only():
	citation = SourceCitation(document_id="doc1", filename="RFC793_Webpage")
	assert citation.label() == "RFC793_Webpage"


def test_recall_and_hit_rate():
	relevant = {"c1", "c2"}
	retrieved = ["x", "c2", "y", "c1", "z"]
	assert recall_at_k(retrieved, relevant, 5) == 1.0
	assert hit_rate_at_k(retrieved, relevant, 1) == 0.0
	assert hit_rate_at_k(retrieved, relevant, 2) == 1.0
	assert mean_reciprocal_rank(retrieved, relevant) == 0.5
