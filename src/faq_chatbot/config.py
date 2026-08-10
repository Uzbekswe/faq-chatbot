"""Typed, single-source runtime configuration."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded once from environment and an optional .env file."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: SecretStr | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    openai_chat_model: str = Field(default="gpt-5.6-luna", validation_alias="OPENAI_CHAT_MODEL")
    openai_embedding_model: str = Field(
        default="text-embedding-3-small", validation_alias="OPENAI_EMBEDDING_MODEL"
    )
    chroma_path: Path = Field(default=Path(".data/chroma"), validation_alias="CHROMA_PATH")
    database_url: str = Field(
        default="sqlite:///.data/faq_chatbot.sqlite3", validation_alias="DATABASE_URL"
    )
    top_k: int = Field(default=4, ge=1, le=10, validation_alias="TOP_K")
    similarity_threshold: float = Field(
        default=0.35, ge=0, le=1, validation_alias="SIMILARITY_THRESHOLD"
    )
    session_ttl_hours: int = Field(default=24, ge=1, le=720, validation_alias="SESSION_TTL_HOURS")
    cookie_secure: bool = Field(default=False, validation_alias="COOKIE_SECURE")
    max_message_chars: int = Field(default=2_000, ge=1, le=20_000)
    session_rate_limit: int = Field(default=20, ge=1, le=200)
    provider_timeout_seconds: float = Field(default=30, ge=1, le=120)

    @field_validator("database_url")
    @classmethod
    def only_sqlite_is_supported(cls, value: str) -> str:
        if not value.startswith("sqlite:///"):
            raise ValueError("DATABASE_URL must use sqlite:///path")
        return value

    @property
    def database_path(self) -> Path:
        return Path(self.database_url.removeprefix("sqlite:///"))

    @property
    def is_configured(self) -> bool:
        return self.openai_api_key is not None and bool(self.openai_api_key.get_secret_value())

    @property
    def missing_configuration(self) -> list[str]:
        return [] if self.is_configured else ["OPENAI_API_KEY"]


@lru_cache
def get_settings() -> Settings:
    return Settings()
