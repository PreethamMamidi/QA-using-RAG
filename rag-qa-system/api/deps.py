"""FastAPI dependency injection helpers."""
from __future__ import annotations

from typing import Annotated, Optional

from fastapi import Depends, Header, HTTPException, Request, status

from api.services.chat_session_service import ChatSessionService
from api.services.index_service import IndexService
from api.settings import Settings, get_settings


def get_index_service(request: Request) -> IndexService:
	return request.app.state.index_service


def get_chat_sessions(request: Request) -> ChatSessionService:
	return request.app.state.chat_sessions


def verify_api_key(
	settings: Annotated[Settings, Depends(get_settings)],
	x_api_key: Annotated[Optional[str], Header()] = None,
) -> None:
	"""Optional API-key gate when Settings.api_key is configured."""
	expected = settings.api_key
	if not expected:
		return
	if x_api_key != expected:
		raise HTTPException(
			status_code=status.HTTP_401_UNAUTHORIZED,
			detail="Invalid or missing X-API-Key header.",
		)


IndexServiceDep = Annotated[IndexService, Depends(get_index_service)]
ChatSessionsDep = Annotated[ChatSessionService, Depends(get_chat_sessions)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
