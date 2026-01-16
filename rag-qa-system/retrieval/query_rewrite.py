"""Query rewrite utilities using Groq for improved retrieval."""
from typing import Optional
import os

from groq import Groq


def _build_system_prompt(mode: str) -> str:
    """Build the system prompt for query rewriting.

    Parameters
    ----------
    mode : str
        Rewrite mode. Use "medical" to preserve domain terminology.

    Returns
    -------
    str
        System prompt string.
    """
    base = (
        "You are a query rewriter for a retrieval system. "
        "Return ONLY the rewritten query text (no quotes, no explanations). "
        "Keep it under 25 words. Remove filler words. "
        "Expand ambiguous questions into a concrete query."
    )
    if mode.strip().lower() == "medical":
        base += " Preserve medical terminology and abbreviations."
    return base


def rewrite_query_groq(question: str, mode: str = "general") -> str:
    """Rewrite a user question into a better retrieval query using Groq.

    Parameters
    ----------
    question : str
        User question to rewrite.
    mode : str, optional
        Rewrite mode, by default "general". Use "medical" to preserve
        domain terminology.

    Returns
    -------
    str
        Rewritten query text, or the original question if missing API key.
    """
    if not question or not question.strip():
        return ""

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return question

    system_prompt = _build_system_prompt(mode)

    client = Groq(api_key=api_key)
    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            temperature=0.1,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question.strip()},
            ],
        )
    except Exception:
        return question

    message = completion.choices[0].message if completion.choices else None
    rewritten = getattr(message, "content", "") if message else ""
    return (rewritten or "").strip()
