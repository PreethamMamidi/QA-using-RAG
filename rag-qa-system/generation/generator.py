"""
Answer generation utilities for the RAG QA system.

Uses the `google/flan-t5-small` model locally on CPU to generate answers
based on retrieved context chunks.
"""
from typing import Dict, List, Tuple, Optional

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration


_TOKENIZER: Optional[T5Tokenizer] = None
_MODEL: Optional[T5ForConditionalGeneration] = None
_MODEL_NAME = "google/flan-t5-small"
_DEVICE = torch.device("cpu")


def get_generator() -> Tuple[T5Tokenizer, T5ForConditionalGeneration]:
	"""Lazily load and return the FLAN-T5 tokenizer and model (CPU-only).

	The model is set to evaluation mode. Callers should use `torch.no_grad()`
	during generation to avoid building gradients.

	Returns
	-------
	Tuple[T5Tokenizer, T5ForConditionalGeneration]
		Tokenizer and model instances.
	"""
	global _TOKENIZER, _MODEL
	if _TOKENIZER is None or _MODEL is None:
		_TOKENIZER = T5Tokenizer.from_pretrained(_MODEL_NAME,legacy=True)
		_MODEL = T5ForConditionalGeneration.from_pretrained(_MODEL_NAME)
		_MODEL.to(_DEVICE)
		_MODEL.eval()
	return _TOKENIZER, _MODEL


def build_context(chunks: List[Dict[str, str]], max_tokens: int = 512) -> str:
	"""Concatenate chunk texts into a single context string with a token cap.

	Parameters
	----------
	chunks : List[Dict[str, str]]
		Retrieved chunks (already sorted by relevance). Each chunk must contain
		a "text" field. Empty texts are skipped.
	max_tokens : int, optional
		Maximum number of approximate whitespace tokens in the final context,
		by default 512.

	Returns
	-------
	str
		Concatenated context string respecting the token cap.
	"""
	if max_tokens <= 0:
		return ""

	parts: List[str] = []
	token_count = 0
	for ch in chunks:
		t = (ch.get("text", "") or "").strip()
		if not t:
			continue
		toks = t.split()
		if not toks:
			continue
		remaining = max_tokens - token_count
		if remaining <= 0:
			break
		if len(toks) <= remaining:
			parts.append(t)
			token_count += len(toks)
		else:
			# Truncate to fit remaining tokens
			parts.append(" ".join(toks[:remaining]))
			token_count += remaining
			break

	return "\n\n".join(parts).strip()


def generate_answer(
	question: str,
	chunks: List[Dict[str, str]],
	max_context_tokens: int = 512,
	max_new_tokens: int = 128,
) -> str:
	"""Generate an answer for a question using retrieved context and FLAN-T5.

	Parameters
	----------
	question : str
		User question.
	chunks : List[Dict[str, str]]
		Retrieved chunks (sorted by relevance) used to build context.
	max_context_tokens : int, optional
		Approximate max context token budget (whitespace tokens), by default 512.
	max_new_tokens : int, optional
		Maximum tokens to generate, by default 128.

	Returns
	-------
	str
		Generated answer string.
	"""
	if not question or not question.strip():
		return ""

	tokenizer, model = get_generator()

	context = build_context(chunks, max_tokens=max_context_tokens)
	prompt = (
		"Answer the question based on the context below.\n\n"
		"Context:\n"
		f"{context}\n\n"
		"Question:\n"
		f"{question}\n\n"
		"Answer:"
	)

	inputs = tokenizer(
		prompt,
		return_tensors="pt",
		truncation=True,
		padding=False,
	)
	input_ids = inputs.input_ids.to(_DEVICE)
	attention_mask = inputs.attention_mask.to(_DEVICE)

	with torch.no_grad():
		output_ids = model.generate(
			input_ids=input_ids,
			attention_mask=attention_mask,
			max_new_tokens=max_new_tokens,
			do_sample=False,  # greedy decoding
			num_beams=1,
		)

	answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
	return answer.strip()

