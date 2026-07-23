"""Session-state helpers for the conversational chat UI."""
from __future__ import annotations

from typing import Any, Dict, List, Literal, TypedDict

import streamlit as st

Role = Literal["user", "assistant"]


class ChatMessage(TypedDict, total=False):
	"""Single chat turn persisted in session state."""

	role: Role
	content: str
	# Optional metadata for future contextual rewrite / UI expanders.
	sources_markdown: str
	citation_report: Dict[str, Any]
	retrieval_debug: Dict[str, Any]
	rewritten_query: str


CHAT_HISTORY_KEY = "chat_history"


def init_chat_state() -> None:
	"""Ensure conversation session keys exist (idempotent)."""
	if CHAT_HISTORY_KEY not in st.session_state:
		st.session_state[CHAT_HISTORY_KEY] = []


def get_chat_history() -> List[ChatMessage]:
	"""Return the current chat history list (mutates session state in place)."""
	init_chat_state()
	return st.session_state[CHAT_HISTORY_KEY]


def set_chat_history(messages: List[ChatMessage]) -> None:
	"""Replace the full chat history."""
	st.session_state[CHAT_HISTORY_KEY] = list(messages or [])


def clear_chat_history() -> None:
	"""Clear only conversation turns; knowledge base state is untouched."""
	st.session_state[CHAT_HISTORY_KEY] = []


def recent_turns(limit: int = 6) -> List[ChatMessage]:
	"""Return the last ``limit`` turns for a future contextual rewrite module.

	Retrieval currently uses only the latest user query; this helper exists so
	a follow-up rewriter can consume prior turns without changing the UI layer.
	"""
	history = get_chat_history()
	if limit <= 0:
		return []
	return history[-limit:]
