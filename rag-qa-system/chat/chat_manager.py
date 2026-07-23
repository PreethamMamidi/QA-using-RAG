"""Conversation mutation helpers for the chat UI."""
from __future__ import annotations

from typing import Any, Dict, Optional

from chat.chat_state import ChatMessage, clear_chat_history, get_chat_history, init_chat_state


def ensure_chat_ready() -> None:
	"""Initialize chat session state if needed."""
	init_chat_state()


def add_user_message(content: str) -> ChatMessage:
	"""Append a user message and return it."""
	ensure_chat_ready()
	message: ChatMessage = {"role": "user", "content": (content or "").strip()}
	get_chat_history().append(message)
	return message


def add_assistant_message(
	content: str,
	*,
	sources_markdown: str = "",
	citation_report: Optional[Dict[str, Any]] = None,
	retrieval_debug: Optional[Dict[str, Any]] = None,
	rewritten_query: str = "",
) -> ChatMessage:
	"""Append an assistant message (optionally with citation metadata)."""
	ensure_chat_ready()
	message: ChatMessage = {
		"role": "assistant",
		"content": content or "",
	}
	if sources_markdown:
		message["sources_markdown"] = sources_markdown
	if citation_report is not None:
		message["citation_report"] = citation_report
	if retrieval_debug is not None:
		message["retrieval_debug"] = retrieval_debug
	if rewritten_query:
		message["rewritten_query"] = rewritten_query
	get_chat_history().append(message)
	return message


def start_new_chat() -> None:
	"""Clear conversation only; preserve knowledge base / indexes / filters."""
	clear_chat_history()
