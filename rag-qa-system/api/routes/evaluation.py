"""Evaluation API routes."""
from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, status

from api.deps import IndexServiceDep, SettingsDep, verify_api_key
from api.schemas import DatasetInfo, EvaluationRunRequest
from api.services.evaluation_service import EvaluationService

router = APIRouter(prefix="/evaluation", tags=["evaluation"], dependencies=[Depends(verify_api_key)])


@router.get("/datasets", response_model=List[DatasetInfo])
def list_datasets(index_service: IndexServiceDep, settings: SettingsDep) -> List[DatasetInfo]:
	service = EvaluationService(index_service, settings.evaluation_dir)
	return [DatasetInfo(**item) for item in service.list_datasets()]


@router.post("/run")
def run_evaluation(
	body: EvaluationRunRequest,
	index_service: IndexServiceDep,
	settings: SettingsDep,
) -> Dict[str, Any]:
	service = EvaluationService(index_service, settings.evaluation_dir)
	try:
		return service.run(body.dataset, use_reranker=body.use_reranker)
	except FileNotFoundError as exc:
		raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
	except RuntimeError as exc:
		raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
	except Exception as exc:
		raise HTTPException(status_code=500, detail=f"Evaluation failed: {exc}") from exc
