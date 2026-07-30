"""In-process chat session store (V1; Redis/SQLite can replace later)."""
from __future__ import annotations

import uuid
from typing import Any, Dict, List, Literal, Optional
from threading import Lock

Role = Literal["user", "assistant"]


class ChatSessionService:
	"""Thread-safe in-memory chat histories keyed by session_id."""

	def __init__(self) -> None:
		self._sessions: Dict[str, List[Dict[str, Any]]] = {}
		self._lock = Lock()

	def create_session(self) -> str:
		session_id = str(uuid.uuid4())
		with self._lock:
			self._sessions[session_id] = []
		return session_id

	def exists(self, session_id: str) -> bool:
		with self._lock:
			return session_id in self._sessions

	def get_messages(self, session_id: str) -> List[Dict[str, Any]]:
		with self._lock:
			if session_id not in self._sessions:
				raise KeyError(session_id)
			return list(self._sessions[session_id])

	def clear_session(self, session_id: str) -> None:
		with self._lock:
			if session_id not in self._sessions:
				raise KeyError(session_id)
			self._sessions[session_id] = []

	def delete_session(self, session_id: str) -> None:
		with self._lock:
			if session_id not in self._sessions:
				raise KeyError(session_id)
			del self._sessions[session_id]

	def add_message(
		self,
		session_id: str,
		role: Role,
		content: str,
		*,
		metadata: Optional[Dict[str, Any]] = None,
	) -> Dict[str, Any]:
		with self._lock:
			if session_id not in self._sessions:
				raise KeyError(session_id)
			message: Dict[str, Any] = {"role": role, "content": content or ""}
			if metadata:
				message.update(metadata)
			self._sessions[session_id].append(message)
			return dict(message)

	def recent_turns(self, session_id: str, limit: int = 6) -> List[Dict[str, Any]]:
		"""Return last N turns for a future contextual rewrite module."""
		messages = self.get_messages(session_id)
		if limit <= 0:
			return []
		return messages[-limit:]
