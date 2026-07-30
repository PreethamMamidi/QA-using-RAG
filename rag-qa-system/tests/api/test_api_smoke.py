"""Smoke tests for the FastAPI backend (lightweight; no ingest/model downloads)."""
from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import create_app
from api.services.chat_session_service import ChatSessionService


def test_health_endpoint():
	app = create_app()
	with TestClient(app) as client:
		resp = client.get("/health")
		assert resp.status_code == 200
		assert resp.json()["status"] == "ok"


def test_ready_and_kb_stats_shape():
	app = create_app()
	with TestClient(app) as client:
		ready = client.get("/ready")
		assert ready.status_code == 200
		assert "ready" in ready.json()

		stats = client.get("/kb/stats")
		assert stats.status_code == 200
		payload = stats.json()
		for key in ("documents", "chunks", "index_loaded", "hybrid_ready"):
			assert key in payload


def test_chat_session_lifecycle():
	store = ChatSessionService()
	session_id = store.create_session()
	assert store.exists(session_id)
	store.add_message(session_id, "user", "hello")
	store.add_message(session_id, "assistant", "hi")
	assert len(store.get_messages(session_id)) == 2
	store.clear_session(session_id)
	assert store.get_messages(session_id) == []

	app = create_app()
	with TestClient(app) as client:
		created = client.post("/chat/sessions")
		assert created.status_code == 200
		sid = created.json()["session_id"]
		msgs = client.get(f"/chat/sessions/{sid}/messages")
		assert msgs.status_code == 200
		assert msgs.json() == []
		cleared = client.delete(f"/chat/sessions/{sid}")
		assert cleared.status_code == 200


def test_incremental_placeholders():
	app = create_app()
	with TestClient(app) as client:
		assert client.post("/documents").status_code == 501
		assert client.delete("/documents/demo").status_code == 501


def test_evaluation_datasets_list():
	app = create_app()
	with TestClient(app) as client:
		resp = client.get("/evaluation/datasets")
		assert resp.status_code == 200
		names = {item["name"] for item in resp.json()}
		assert "gold.json" in names or "sample_dataset.json" in names
