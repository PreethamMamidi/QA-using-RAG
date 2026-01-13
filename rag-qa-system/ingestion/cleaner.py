"""
Text cleaning utilities for the RAG QA system.

This module focuses on minimal, safe normalization before chunking/embedding.
It preserves casing and punctuation, does not tokenize, and uses only
standard Python libraries.
"""
from typing import Optional
import re


def clean_text(text: Optional[str]) -> str:
	"""Clean raw text by applying minimal, safe normalization.

	Operations performed (in order):
	- If input is None or empty, return an empty string.
	- Normalize line endings and whitespace by collapsing any sequence of
	  whitespace (spaces, tabs, newlines) into a single space.
	- Remove non-ASCII characters.
	- Strip leading and trailing whitespace.

	Notes
	-----
	- Casing and punctuation are preserved.
	- No tokenization is performed.

	Parameters
	----------
	text : Optional[str]
		Input text to clean.

	Returns
	-------
	str
		Cleaned text string.
	"""
	if not text:
		return ""

	# Remove non-ASCII characters by encoding/decoding.
	# This safely strips characters like \u2028 without altering casing/punctuation.
	ascii_text = text.encode("ascii", errors="ignore").decode("ascii")

	# Collapse any whitespace (spaces, tabs, newlines) to a single space.
	normalized = re.sub(r"\s+", " ", ascii_text)
	lower = normalized.lower()
	for marker in ["references", "bibliography"]:
		idx = lower.find(marker)
		if idx != -1:
			normalized = normalized[:idx].strip()
			break

	return normalized.strip()

