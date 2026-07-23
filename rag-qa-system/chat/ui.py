"""Streamlit chat rendering helpers."""
from __future__ import annotations

import json
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional

import streamlit as st

from chat.chat_manager import start_new_chat
from chat.chat_state import ChatMessage, get_chat_history, init_chat_state


def render_conversation_sidebar() -> bool:
	"""Render Conversation sidebar controls.

	Returns
	-------
	bool
		True when the user clicked **New Chat**.
	"""
	st.sidebar.subheader("Conversation")
	clicked = st.sidebar.button("New Chat", key="new_chat_button", use_container_width=True)
	st.sidebar.caption("Clears chat only. Documents and indexes stay loaded.")
	if clicked:
		start_new_chat()
		st.sidebar.success("Started a new conversation.")
		st.rerun()
	return clicked


def render_chat_history(messages: Optional[List[ChatMessage]] = None) -> None:
	"""Render persisted chat bubbles for prior turns."""
	init_chat_state()
	history = messages if messages is not None else get_chat_history()

	for index, message in enumerate(history):
		role = message.get("role", "assistant")
		with st.chat_message(role):
			st.markdown(message.get("content") or "")
			if role == "assistant":
				_render_assistant_extras(message, key_suffix=str(index))


def _render_assistant_extras(message: ChatMessage, *, key_suffix: str) -> None:
	"""Render sources / debug expanders under an assistant bubble."""
	sources = message.get("sources_markdown") or ""
	if sources:
		with st.expander("Sources", expanded=False):
			st.markdown(sources)

	report = message.get("citation_report")
	if report:
		st.download_button(
			label="Download citation report (JSON)",
			data=json.dumps(report, indent=2),
			file_name=f"citation_report_{key_suffix}.json",
			mime="application/json",
			key=f"citation_download_{key_suffix}",
		)

	rewritten = message.get("rewritten_query")
	if rewritten:
		st.caption(f"Retrieval query: {rewritten}")

	debug = message.get("retrieval_debug")
	if debug:
		with st.expander("Retrieval debug", expanded=False):
			st.json(debug)


def stream_assistant_reply(
	token_stream: Iterable[str] | Iterator[str],
) -> str:
	"""Stream tokens into the current assistant chat bubble and return full text.

	Must be called inside ``st.chat_message("assistant")``.
	"""
	# st.write_stream returns the concatenated string when the iterator completes.
	result = st.write_stream(token_stream)
	if isinstance(result, str):
		return result
	if result is None:
		return ""
	return str(result)


def render_empty_chat_placeholder() -> None:
	"""Friendly empty-state when no messages exist yet."""
	st.markdown(
		"Ask a question about your knowledge base. Follow-ups stay in this chat "
		"until you click **New Chat**."
	)


AnswerStreamer = Callable[..., Iterator[str]]


def build_assistant_payload(
	*,
	answer: str,
	sources_markdown: str = "",
	citation_report: Optional[Dict[str, Any]] = None,
	retrieval_debug: Optional[Dict[str, Any]] = None,
	rewritten_query: str = "",
) -> ChatMessage:
	"""Build an assistant message dict ready for chat_manager.add_assistant_message."""
	payload: ChatMessage = {"role": "assistant", "content": answer}
	if sources_markdown:
		payload["sources_markdown"] = sources_markdown
	if citation_report is not None:
		payload["citation_report"] = citation_report
	if retrieval_debug is not None:
		payload["retrieval_debug"] = retrieval_debug
	if rewritten_query:
		payload["rewritten_query"] = rewritten_query
	return payload
