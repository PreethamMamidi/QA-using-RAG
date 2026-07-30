"""Application settings for the FastAPI backend."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
	"""Runtime configuration loaded from environment / .env."""

	model_config = SettingsConfigDict(
		env_file=str(ROOT_DIR / ".env"),
		env_file_encoding="utf-8",
		extra="ignore",
	)

	storage_dir: Path = ROOT_DIR / "storage"
	evaluation_dir: Path = ROOT_DIR / "evaluation"
	api_key: str | None = None
	cors_origins: str = "*"
	default_generator: str = "groq"
	default_groq_model: str = "llama-3.3-70b-versatile"
	groq_api_key: str | None = None

	@property
	def faiss_index_path(self) -> Path:
		return self.storage_dir / "faiss.index"

	@property
	def metadata_db_path(self) -> Path:
		return self.storage_dir / "metadata.db"

	@property
	def raw_docs_dir(self) -> Path:
		return self.storage_dir / "raw_docs"

	def cors_origin_list(self) -> List[str]:
		if self.cors_origins.strip() == "*":
			return ["*"]
		return [part.strip() for part in self.cors_origins.split(",") if part.strip()]


@lru_cache
def get_settings() -> Settings:
	"""Return cached settings instance."""
	return Settings()
