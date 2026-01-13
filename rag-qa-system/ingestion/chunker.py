"""
Chunking utilities for the RAG QA system.

Provides sentence-aware, overlapping chunks suitable for embedding/retrieval.
Uses NLTK sentence tokenizer; callers should ensure NLTK's 'punkt' resource
is available (run `import nltk; nltk.download("punkt")` once if needed).
"""
from typing import Dict, List

import nltk
from nltk.tokenize import sent_tokenize



def _ensure_punkt() -> None:
    try:
        nltk.data.find("tokenizers/punkt")
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError as e:
        raise RuntimeError(
            "NLTK sentence tokenizer resources not found.\n"
            "Run:\n"
            "    import nltk\n"
            "    nltk.download('punkt')\n"
            "    nltk.download('punkt_tab')\n"
        ) from e



def _token_count(s: str) -> int:
	"""Approximate token count by whitespace splitting."""
	return len(s.split())


def chunk_text(
	text: str,
	document_id: str,
	chunk_size: int = 400,
	overlap: int = 80,
) -> List[Dict[str, str]]:
	"""
	Split text into sentence-based overlapping chunks, bounded by token size.

	Parameters
	----------
	text : str
		Cleaned input text to split. If empty/whitespace, returns [].
	document_id : str
		Identifier for the source document; used to form unique chunk_ids.
	chunk_size : int, optional
		Maximum tokens per chunk (approximate via whitespace), by default 400.
	overlap : int, optional
		Number of tokens to reuse from the previous chunk for context, by default 80.

	Returns
	-------
	List[Dict[str, str]]
		Chunks as dictionaries: {"chunk_id", "document_id", "text"}.

	Notes
	-----
	- Chunks never exceed `chunk_size` tokens.
	- Overlap is enforced between consecutive chunks.
	- Very short empty chunks are skipped; short documents still yield one chunk.
	"""
	if not text or not text.strip():
		return []

	_ensure_punkt()

	sentences = [s for s in sent_tokenize(text) if s and s.strip()]
	if not sentences:
		# Fallback: treat whole text as one chunk if tokenizer yields nothing
		sentences = [text.strip()]

	chunks_tokens: List[List[str]] = []
	current: List[str] = []

	def finalize_current():
		nonlocal current
		if current:
			chunks_tokens.append(current[:])
			current = []

	# Bound overlap to be less than chunk_size to avoid zero/negative stride
	overlap = max(0, min(overlap, max(0, chunk_size - 1)))

	for s in sentences:
		sent_tokens = s.split()
		if not sent_tokens:
			continue

		# If a single sentence is longer than chunk_size, split it into blocks
		if len(sent_tokens) > chunk_size:
			# Flush any current content first
			finalize_current()
			stride = max(1, chunk_size - overlap)
			j = 0
			while j < len(sent_tokens):
				block = sent_tokens[j:j + chunk_size]
				if block:
					chunks_tokens.append(block)
				j += stride
			# Seed the next chunk with overlap from the last block
			if chunks_tokens:
				last = chunks_tokens[-1]
				current = last[-min(overlap, len(last)) :]
			continue

		# Normal sentence handling
		if len(current) + len(sent_tokens) <= chunk_size:
			current.extend(sent_tokens)
		else:
			# Finalize current and start a new chunk with overlap, then add sentence
			prev_tail: List[str] = current[-min(overlap, len(current)) :] if current else []
			finalize_current()
			# Adjust effective overlap to ensure sentence fits into new chunk
			effective_overlap = min(len(prev_tail), max(0, chunk_size - len(sent_tokens)))
			if effective_overlap > 0:
				current = prev_tail[-effective_overlap:].copy()
			else:
				current = []
			# Now add sentence (guaranteed to fit)
			current.extend(sent_tokens)

	# Finalize any remaining tokens
	finalize_current()

	# Filter empty and extremely short chunks, but keep single short doc chunk
	MIN_TOKENS = 5
	if len(chunks_tokens) > 1:
		chunks_tokens = [t for t in chunks_tokens if len(t) >= MIN_TOKENS]
		if not chunks_tokens:
			return []

	# Materialize dicts
	out: List[Dict[str, str]] = []
	for i, toks in enumerate(chunks_tokens):
		if not toks:
			continue
		text_chunk = " ".join(toks).strip()
		if not text_chunk:
			continue
		out.append({
			"chunk_id": f"{document_id}_{i}",
			"document_id": document_id,
			"text": text_chunk,
		})

	return out

