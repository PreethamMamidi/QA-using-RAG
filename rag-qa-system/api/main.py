"""FastAPI application entrypoint."""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import chat, documents, evaluation, health, kb
from api.services.chat_session_service import ChatSessionService
from api.services.index_service import IndexService
from api.settings import get_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _sync_env_from_settings(settings) -> None:
	"""Ensure legacy os.getenv callers (Groq rewrite/generator) see settings."""
	if settings.groq_api_key and not os.getenv("GROQ_API_KEY"):
		os.environ["GROQ_API_KEY"] = settings.groq_api_key


@asynccontextmanager
async def lifespan(app: FastAPI):
	settings = get_settings()
	_sync_env_from_settings(settings)
	settings.storage_dir.mkdir(parents=True, exist_ok=True)

	index_service = IndexService(settings.storage_dir, settings.metadata_db_path)
	loaded = index_service.load_from_disk()
	logger.info("IndexService startup load_from_disk=%s ready=%s", loaded, index_service.ready)

	app.state.index_service = index_service
	app.state.chat_sessions = ChatSessionService()
	app.state.settings = settings
	yield
	# Process exit: nothing to flush; FAISS already on disk after ingest.


def create_app() -> FastAPI:
	settings = get_settings()
	app = FastAPI(
		title="RAG QA System API",
		version="1.0.0",
		description=(
			"HTTP API for multi-format RAG: ingest (V1 full rebuild), hybrid retrieval, "
			"streaming chat, citations, and evaluation. "
			"Incremental POST/DELETE /documents/{id} is reserved for V2."
		),
		lifespan=lifespan,
	)
	app.add_middleware(
		CORSMiddleware,
		allow_origins=settings.cors_origin_list(),
		allow_credentials=True,
		allow_methods=["*"],
		allow_headers=["*"],
	)

	app.include_router(health.router)
	app.include_router(kb.router)
	app.include_router(documents.router)
	app.include_router(chat.router)
	app.include_router(evaluation.router)
	return app


app = create_app()
