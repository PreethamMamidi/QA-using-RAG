"""Knowledge-base stats, reset, and reload."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from api.deps import IndexServiceDep, verify_api_key
from api.schemas import KbStatsResponse
from api.services.ingestion_service import IngestionService

router = APIRouter(prefix="/kb", tags=["knowledge-base"], dependencies=[Depends(verify_api_key)])


@router.get("/stats", response_model=KbStatsResponse)
def kb_stats(index_service: IndexServiceDep) -> KbStatsResponse:
	return KbStatsResponse(**index_service.stats())


@router.post("/reload", response_model=KbStatsResponse)
def kb_reload(index_service: IndexServiceDep) -> KbStatsResponse:
	loaded = index_service.load_from_disk()
	if not loaded and not index_service.ready:
		raise HTTPException(
			status_code=status.HTTP_404_NOT_FOUND,
			detail="No FAISS index / metadata.db found on disk.",
		)
	return KbStatsResponse(**index_service.stats())


@router.delete("", response_model=KbStatsResponse)
def kb_reset(index_service: IndexServiceDep) -> KbStatsResponse:
	"""Full knowledge-base reset (SQLite + FAISS + raw_docs)."""
	service = IngestionService(index_service)
	result = service.reset_knowledge_base()
	return KbStatsResponse(**result)
