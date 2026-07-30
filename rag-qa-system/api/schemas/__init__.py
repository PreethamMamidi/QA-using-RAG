"""Pydantic request/response schemas for the FastAPI backend."""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
	status: str = "ok"


class ReadyResponse(BaseModel):
	ready: bool
	detail: str = ""


class DocumentOut(BaseModel):
	document_id: str
	filename: str
	upload_time: str
	total_chunks: int = 0
	source_type: str = "file"
	source_url: Optional[str] = None


class KbStatsResponse(BaseModel):
	documents: int
	chunks: int
	index_loaded: bool
	hybrid_ready: bool
	faiss_ntotal: int = 0
	storage_dir: str = ""


class IngestResponse(BaseModel):
	"""V1: full knowledge-base rebuild after ingest."""

	stats_label: str
	sources_processed: int
	docs_loaded: int
	chunks_created: int
	documents: int
	note: str = (
		"V1 uses full KB rebuild (replace semantics). "
		"Incremental POST/DELETE /documents/{id} is reserved for V2."
	)


class UrlIngestRequest(BaseModel):
	url: str = Field(..., min_length=8)


class SessionCreateResponse(BaseModel):
	session_id: str


class ChatMessageOut(BaseModel):
	role: Literal["user", "assistant"]
	content: str
	sources_markdown: Optional[str] = None
	citation_report: Optional[Dict[str, Any]] = None
	retrieval_debug: Optional[Dict[str, Any]] = None
	rewritten_query: Optional[str] = None


class ChatMessageRequest(BaseModel):
	content: str = Field(..., min_length=1)
	generator: Literal["groq", "local"] = "groq"
	groq_model: str = "llama-3.3-70b-versatile"
	rewrite_mode: str = "general"
	enable_hybrid: bool = True
	use_reranker: bool = False
	top_k_dense: int = Field(8, ge=1, le=50)
	top_k_sparse: int = Field(8, ge=1, le=50)
	top_k_fused: int = Field(8, ge=1, le=25)
	rrf_k: int = Field(60, ge=1, le=200)
	document_ids: Optional[List[str]] = None
	return_debug: bool = False


class ChatAnswerResponse(BaseModel):
	session_id: str
	answer: str
	rewritten_query: str
	sources_markdown: str = ""
	citation_report: Dict[str, Any] = Field(default_factory=dict)
	retrieval_debug: Optional[Dict[str, Any]] = None
	messages: List[ChatMessageOut] = Field(default_factory=list)


class EvaluationRunRequest(BaseModel):
	dataset: str = "gold.json"
	use_reranker: bool = False


class DatasetInfo(BaseModel):
	name: str
	path: str


class ErrorResponse(BaseModel):
	detail: str
