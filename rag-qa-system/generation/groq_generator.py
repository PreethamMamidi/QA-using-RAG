"""Groq-powered answer generation for the RAG system."""
from typing import Dict, Iterator, List
import os

from groq import Groq

from generation.prompts import GROQ_ANSWER_INSTRUCTIONS


def build_context(chunks: List[Dict[str, str]], max_chars: int = 8000) -> str:
    """Build a context string from retrieved chunks with source metadata.

    Parameters
    ----------
    chunks : List[Dict[str, str]]
        Retrieved chunks that include at least "text" and "document_id".
    max_chars : int, optional
        Character budget for the resulting context, by default 8000.

    Returns
    -------
    str
        Concatenated context with source tags, capped by max_chars.
    """
    if max_chars <= 0:
        return ""

    parts: List[str] = []
    remaining = max_chars

    for idx, ch in enumerate(chunks):
        text = (ch.get("text") or "").strip()
        if not text:
            continue

        doc_id = str(ch.get("document_id", "unknown"))
        chunk_id = str(ch.get("chunk_id", idx))
        score_val = ch.get("score")
        if isinstance(score_val, (int, float)):
            score = f"{score_val:.3f}"
        elif score_val is not None:
            score = str(score_val)
        else:
            score = "n/a"

        header = f"[SOURCE: {doc_id} | chunk_id: {chunk_id} | score: {score}]\n"
        entry = f"{header}{text}\n\n"

        if len(entry) > remaining:
            # Try to fit as much of this chunk as possible within the budget.
            space_for_text = max(0, remaining - len(header) - 2)  # minus \n\n
            if space_for_text <= 0:
                break

            truncated_text = text[:space_for_text]
            parts.append(f"{header}{truncated_text}")
            remaining = 0
            break

        parts.append(entry)
        remaining -= len(entry)

        if remaining <= 0:
            break

    return "".join(parts).strip()


def _prepare_groq_request(
    question: str,
    chunks: List[Dict[str, str]],
) -> tuple[str, str, str] | tuple[None, None, str]:
    """Return (api_key, system_prompt, user_message) or (None, None, error_message)."""
    if not question or not question.strip():
        return None, None, ""

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None, None, "GROQ_API_KEY is not set. Please configure it in your environment."

    context = build_context(chunks)
    if len(context.strip()) < 200:
        return None, None, "I don't know based on the uploaded document."

    user_message = f"Context:\n{context}\n\nQuestion:\n{question}"
    return api_key, GROQ_ANSWER_INSTRUCTIONS, user_message


def stream_answer_groq(
    question: str,
    chunks: List[Dict[str, str]],
    model: str = "llama-3.3-70b-versatile",
    temperature: float = 0.2,
) -> Iterator[str]:
    """Stream an answer token-by-token using Groq chat completions."""
    api_key, system_prompt, user_message = _prepare_groq_request(question, chunks)
    if api_key is None:
        if user_message:
            yield user_message
        return

    client = Groq(api_key=api_key)

    try:
        stream = client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=1024,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            stream=True,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
    except Exception as exc:  # pragma: no cover - network/SDK errors
        yield f"Error calling Groq API: {exc}"


def generate_answer_groq(
    question: str,
    chunks: List[Dict[str, str]],
    model: str = "llama-3.3-70b-versatile",
    temperature: float = 0.2,
) -> str:
    """Generate an answer using Groq chat completions.

    Parameters
    ----------
    question : str
        User question.
    chunks : List[Dict[str, str]]
        Retrieved context chunks to ground the answer.
    model : str, optional
        Groq chat completion model name, by default "llama-3.3-70b-versatile".
    temperature : float, optional
        Sampling temperature, by default 0.2.

    Returns
    -------
    str
        Generated answer text or an informative error string.
    """
    return "".join(
        stream_answer_groq(
            question,
            chunks,
            model=model,
            temperature=temperature,
        )
    ).strip()
