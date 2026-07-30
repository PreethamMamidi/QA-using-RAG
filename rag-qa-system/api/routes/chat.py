"""Chat session and streaming answer routes."""
from __future__ import annotations

import json
import os
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sse_starlette.sse import EventSourceResponse

from api.deps import ChatSessionsDep, IndexServiceDep, SettingsDep, verify_api_key
from api.schemas import (
	ChatAnswerResponse,
	ChatMessageOut,
	ChatMessageRequest,
	SessionCreateResponse,
)
from api.services.generation_service import GenerationService
from api.services.retrieval_service import RetrievalService
from retrieval.citations import citations_from_chunks, citations_report_payload, format_citations_markdown

router = APIRouter(prefix="/chat", tags=["chat"], dependencies=[Depends(verify_api_key)])


def _session_or_404(chat_sessions: ChatSessionsDep, session_id: str) -> None:
	if not chat_sessions.exists(session_id):
		raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found.")


@router.post("/sessions", response_model=SessionCreateResponse)
def create_session(chat_sessions: ChatSessionsDep) -> SessionCreateResponse:
	return SessionCreateResponse(session_id=chat_sessions.create_session())


@router.get("/sessions/{session_id}/messages", response_model=List[ChatMessageOut])
def list_messages(session_id: str, chat_sessions: ChatSessionsDep) -> List[ChatMessageOut]:
	_session_or_404(chat_sessions, session_id)
	return [ChatMessageOut(**msg) for msg in chat_sessions.get_messages(session_id)]


@router.delete("/sessions/{session_id}", response_model=SessionCreateResponse)
def clear_session(session_id: str, chat_sessions: ChatSessionsDep) -> SessionCreateResponse:
	"""Clear conversation history only (New Chat). Knowledge base is untouched."""
	_session_or_404(chat_sessions, session_id)
	chat_sessions.clear_session(session_id)
	return SessionCreateResponse(session_id=session_id)


@router.post("/sessions/{session_id}/messages", response_model=ChatAnswerResponse)
def post_message(
	session_id: str,
	body: ChatMessageRequest,
	index_service: IndexServiceDep,
	chat_sessions: ChatSessionsDep,
	settings: SettingsDep,
) -> ChatAnswerResponse:
	_session_or_404(chat_sessions, session_id)

	chat_sessions.add_message(session_id, "user", body.content)
	retrieval = RetrievalService(index_service)
	result = retrieval.retrieve(
		body.content,
		rewrite_mode=body.rewrite_mode,
		enable_hybrid=body.enable_hybrid,
		use_reranker=body.use_reranker,
		top_k_dense=body.top_k_dense,
		top_k_sparse=body.top_k_sparse,
		top_k_fused=body.top_k_fused,
		rrf_k=body.rrf_k,
		document_ids=body.document_ids,
		return_debug=body.return_debug,
	)
	if result.error:
		chat_sessions.add_message(session_id, "assistant", result.error)
		raise HTTPException(status_code=400, detail=result.error)

	generator = body.generator or settings.default_generator
	if generator == "groq" and not (settings.groq_api_key or os.getenv("GROQ_API_KEY")):
		detail = "GROQ_API_KEY is not set. Configure it or use generator='local'."
		chat_sessions.add_message(session_id, "assistant", detail)
		raise HTTPException(status_code=400, detail=detail)

	gen = GenerationService()
	answer = gen.generate(
		body.content,
		result.chunks,
		generator=generator,  # type: ignore[arg-type]
		groq_model=body.groq_model or settings.default_groq_model,
	)

	citations = citations_from_chunks(result.chunks)
	sources_md = format_citations_markdown(citations) if citations else ""
	report = citations_report_payload(result.chunks, question=body.content, answer=answer)
	meta = {
		"sources_markdown": sources_md,
		"citation_report": report,
		"rewritten_query": result.rewritten_query if result.rewritten_query != body.content else None,
	}
	if body.return_debug:
		meta["retrieval_debug"] = result.debug

	chat_sessions.add_message(session_id, "assistant", answer, metadata=meta)
	messages = [ChatMessageOut(**m) for m in chat_sessions.get_messages(session_id)]
	return ChatAnswerResponse(
		session_id=session_id,
		answer=answer,
		rewritten_query=result.rewritten_query,
		sources_markdown=sources_md,
		citation_report=report,
		retrieval_debug=result.debug if body.return_debug else None,
		messages=messages,
	)


@router.post("/sessions/{session_id}/messages:stream")
async def post_message_stream(
	session_id: str,
	body: ChatMessageRequest,
	index_service: IndexServiceDep,
	chat_sessions: ChatSessionsDep,
	settings: SettingsDep,
):
	"""SSE stream: token events, then a final JSON event with sources/citations."""
	_session_or_404(chat_sessions, session_id)
	chat_sessions.add_message(session_id, "user", body.content)

	retrieval = RetrievalService(index_service)
	result = retrieval.retrieve(
		body.content,
		rewrite_mode=body.rewrite_mode,
		enable_hybrid=body.enable_hybrid,
		use_reranker=body.use_reranker,
		top_k_dense=body.top_k_dense,
		top_k_sparse=body.top_k_sparse,
		top_k_fused=body.top_k_fused,
		rrf_k=body.rrf_k,
		document_ids=body.document_ids,
		return_debug=body.return_debug,
	)
	if result.error:
		chat_sessions.add_message(session_id, "assistant", result.error)

		async def error_events():
			yield {"event": "error", "data": json.dumps({"detail": result.error})}

		return EventSourceResponse(error_events())

	generator = body.generator or settings.default_generator
	if generator == "groq" and not (settings.groq_api_key or os.getenv("GROQ_API_KEY")):
		detail = "GROQ_API_KEY is not set. Configure it or use generator='local'."
		chat_sessions.add_message(session_id, "assistant", detail)

		async def missing_key_events():
			yield {"event": "error", "data": json.dumps({"detail": detail})}

		return EventSourceResponse(missing_key_events())

	gen = GenerationService()
	token_iter = gen.stream(
		body.content,
		result.chunks,
		generator=generator,  # type: ignore[arg-type]
		groq_model=body.groq_model or settings.default_groq_model,
	)

	async def event_generator():
		parts: list[str] = []
		for token in token_iter:
			parts.append(token)
			yield {"event": "token", "data": json.dumps({"token": token})}

		answer = "".join(parts).strip()
		citations = citations_from_chunks(result.chunks)
		sources_md = format_citations_markdown(citations) if citations else ""
		report = citations_report_payload(result.chunks, question=body.content, answer=answer)
		meta = {
			"sources_markdown": sources_md,
			"citation_report": report,
			"rewritten_query": result.rewritten_query if result.rewritten_query != body.content else None,
		}
		if body.return_debug:
			meta["retrieval_debug"] = result.debug
		chat_sessions.add_message(session_id, "assistant", answer, metadata=meta)

		final_payload = {
			"answer": answer,
			"rewritten_query": result.rewritten_query,
			"sources_markdown": sources_md,
			"citation_report": report,
			"retrieval_debug": result.debug if body.return_debug else None,
		}
		yield {"event": "final", "data": json.dumps(final_payload)}

	return EventSourceResponse(event_generator())
