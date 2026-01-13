"""
Document loading utilities for the RAG QA system.

Features:
- Reads all files in a given directory (non-recursive).
- Supports UTF-8 TXT files and PDF files via PyMuPDF (fitz).
- Returns a list of dictionaries with `document_id` and `text`.
	- TXT: one document per file (`document_id` = filename)
	- PDF: one document per page (`document_id` = "<filename>_page_<n>")

Notes:
- This module performs no text cleaning; it only loads raw text.
- Unsupported file types are skipped silently.
- Raises FileNotFoundError if the provided directory does not exist.
"""
from pathlib import Path
from typing import Dict, List

import fitz  # PyMuPDF


def _load_txt(path: Path) -> str:
	"""Load a UTF-8 text file and return its contents as a string."""
	with path.open("r", encoding="utf-8", errors="ignore") as f:
		return f.read().strip()


def _load_pdf_pages(path: Path) -> List[Dict[str, str]]:
	"""Load a PDF and return one document entry per page.

	Each entry contains:
	- document_id: "<filename>_page_<n>" (1-based page number)
	- text: extracted page text (skips empty pages)
	"""
	entries: List[Dict[str, str]] = []
	with fitz.open(str(path)) as doc:
		for i in range(doc.page_count):
			page = doc.load_page(i)
			text = (page.get_text("text") or "").strip()
			if not text:
				continue
			entries.append({
				"document_id": f"{path.name}_page_{i+1}",
				"text": text,
			})
	return entries


def load_documents(data_dir: str) -> List[Dict[str, str]]:
	"""
	Load documents from a directory and return a list of entries.

	Parameters
	----------
	data_dir : str
		Path to the directory containing raw documents.

	Returns
	-------
	List[Dict[str, str]]
		A list of dictionaries with keys:
		- "document_id": str
			* TXT: filename
			* PDF: "<filename>_page_<n>"
		- "text": str (raw extracted text)

	Raises
	------
	FileNotFoundError
		If the provided directory does not exist or is not a directory.
	"""
	root = Path(data_dir)
	if not root.exists() or not root.is_dir():
		raise FileNotFoundError(f"Directory not found or not a directory: {data_dir}")

	entries: List[Dict[str, str]] = []
	for p in sorted(root.iterdir()):
		if not p.is_file():
			continue

		suffix = p.suffix.lower()
		if suffix == ".txt":
			text = _load_txt(p)
			entries.append({
				"document_id": p.name,
				"text": text,
			})
		elif suffix == ".pdf":
			page_entries = _load_pdf_pages(p)
			entries.extend(page_entries)
		else:
			# Unsupported type: skip silently
			continue

	return entries

