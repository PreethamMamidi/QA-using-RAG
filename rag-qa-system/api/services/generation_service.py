"""Answer generation wrappers for the API."""
from __future__ import annotations

from typing import Any, Dict, Iterator, List, Literal

from generation.generator import stream_answer
from generation.groq_generator import stream_answer_groq

GeneratorName = Literal["groq", "local"]


class GenerationService:
	"""Stream or collect answers from Groq / local FLAN-T5."""

	def stream(
		self,
		question: str,
		chunks: List[Dict[str, Any]],
		*,
		generator: GeneratorName = "groq",
		groq_model: str = "llama-3.3-70b-versatile",
	) -> Iterator[str]:
		if generator == "local":
			yield from stream_answer(question, chunks)
			return
		yield from stream_answer_groq(question, chunks, model=groq_model)

	def generate(
		self,
		question: str,
		chunks: List[Dict[str, Any]],
		*,
		generator: GeneratorName = "groq",
		groq_model: str = "llama-3.3-70b-versatile",
	) -> str:
		return "".join(
			self.stream(
				question,
				chunks,
				generator=generator,
				groq_model=groq_model,
			)
		).strip()
