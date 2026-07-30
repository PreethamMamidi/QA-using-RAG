"""Document listing and ingest endpoints (V1 full rebuild)."""
from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from api.deps import IndexServiceDep, verify_api_key
from api.schemas import DocumentOut, IngestResponse, UrlIngestRequest
from api.services.ingestion_service import IngestionService

router = APIRouter(prefix="/documents", tags=["documents"], dependencies=[Depends(verify_api_key)])


@router.get("", response_model=List[DocumentOut])
def list_documents(index_service: IndexServiceDep) -> List[DocumentOut]:
	docs = index_service.repository().list_documents()
	return [DocumentOut(**doc) for doc in docs]


@router.get("/{document_id}", response_model=DocumentOut)
def get_document(document_id: str, index_service: IndexServiceDep) -> DocumentOut:
	doc = index_service.repository().get_document(document_id)
	if doc is None:
		raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")
	return DocumentOut(**doc)


@router.post(
	"/ingest/files",
	response_model=IngestResponse,
	summary="Ingest uploaded files and rebuild the entire knowledge base (V1)",
	description=(
		"V1 semantics: clears existing SQLite metadata and rebuilds FAISS from the "
		"uploaded corpus (full replace when replace=true, the default). "
		"Also wipes storage/raw_docs before writing uploads. "
		"Incremental POST /documents is reserved for V2."
	),
)
async def ingest_files(
	index_service: IndexServiceDep,
	files: List[UploadFile] = File(...),
	replace: bool = Form(True),
) -> IngestResponse:
	if not files:
		raise HTTPException(status_code=400, detail="No files uploaded.")

	payload: list[tuple[str, bytes]] = []
	for upload in files:
		data = await upload.read()
		name = upload.filename or "upload.bin"
		payload.append((name, data))

	service = IngestionService(index_service)
	try:
		result = service.ingest_files(payload, replace=replace)
	except ValueError as exc:
		raise HTTPException(status_code=400, detail=str(exc)) from exc
	except Exception as exc:
		raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc

	return IngestResponse(**result)


@router.post(
	"/ingest/url",
	response_model=IngestResponse,
	summary="Ingest a web URL and rebuild the entire knowledge base (V1)",
)
def ingest_url(body: UrlIngestRequest, index_service: IndexServiceDep) -> IngestResponse:
	service = IngestionService(index_service)
	try:
		result = service.ingest_url(body.url)
	except ValueError as exc:
		raise HTTPException(status_code=400, detail=str(exc)) from exc
	except Exception as exc:
		raise HTTPException(status_code=500, detail=f"URL ingestion failed: {exc}") from exc
	return IngestResponse(**result)


@router.post(
	"",
	status_code=status.HTTP_501_NOT_IMPLEMENTED,
	summary="Reserved: incremental document append (V2)",
)
def incremental_add_placeholder() -> dict:
	return {
		"detail": "Incremental POST /documents is reserved for V2. Use POST /documents/ingest/files (full rebuild) in V1."
	}


@router.delete(
	"/{document_id}",
	status_code=status.HTTP_501_NOT_IMPLEMENTED,
	summary="Reserved: incremental document delete (V2)",
)
def incremental_delete_placeholder(document_id: str) -> dict:
	return {
		"detail": (
			f"Incremental DELETE /documents/{document_id} is reserved for V2. "
			"Use DELETE /kb for a full reset in V1."
		)
	}
