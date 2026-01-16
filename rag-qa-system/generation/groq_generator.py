"""Groq-powered answer generation for the RAG system."""
from typing import Dict, List
import os

from groq import Groq


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
    if not question or not question.strip():
        return ""

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "GROQ_API_KEY is not set. Please configure it in your environment."

    context = build_context(chunks)
    if len(context.strip()) < 200:
        return "I don't know based on the uploaded document."

    system_prompt = (
        "You are a concise assistant. Answer using ONLY the provided context. "
        "Respond in 1 to 3 complete sentences. "
        "If the context lacks the answer, reply: 'I don't know based on the uploaded document.' "
        "Do not copy the context verbatim. "
        "Include citations in the final answer like '(Source: Lec16.pdf_page_4)'. "
        "Cite the best 1-2 sources max. "
        "Do not repeat the question. "
        "Avoid bullet lists unless the user explicitly asks for them."
    )

    client = Groq(api_key=api_key)

    try:
        completion = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion:\n{question}",
                },
            ],
        )
    except Exception as exc:  # pragma: no cover - network/SDK errors
        return f"Error calling Groq API: {exc}"

    message = completion.choices[0].message if completion.choices else None
    answer = getattr(message, "content", "") if message else ""
    return (answer or "").strip()
