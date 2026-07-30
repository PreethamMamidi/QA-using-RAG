"""Health and readiness routes."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from api.deps import IndexServiceDep, verify_api_key
from api.schemas import HealthResponse, ReadyResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
	return HealthResponse(status="ok")


@router.get("/ready", response_model=ReadyResponse, dependencies=[Depends(verify_api_key)])
def ready(index_service: IndexServiceDep) -> ReadyResponse:
	if index_service.ready:
		return ReadyResponse(ready=True, detail="Knowledge base loaded.")
	return ReadyResponse(
		ready=False,
		detail="FAISS index or chunks not loaded. Ingest documents or POST /kb/reload.",
	)
