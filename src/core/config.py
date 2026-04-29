"""Application settings — single source of truth for environment configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    """All runtime configuration. Read once at startup; do not mutate.

    Reads from process env and `.env` (via python-dotenv loaded by pydantic-settings).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Telegram
    telegram_bot_token: SecretStr = Field(...)
    allowed_telegram_user_ids: Annotated[list[int], NoDecode] = Field(...)

    # LLM
    llm_provider: str = Field(...)
    llm_model: str = Field(...)
    llm_api_key: SecretStr = Field(...)
    llm_provider_base_url: str | None = None

    # Google Calendar
    google_oauth_client_secrets_path: Path = Field(...)
    google_oauth_token_path: Path = Field(default=Path("./data/token.json"))

    # Persistence
    checkpoint_db_path: Path = Field(default=Path("./data/checkpoints.sqlite"))

    # Time handling
    default_timezone: str | None = None

    # Logging
    log_level: str = "INFO"

    # LangSmith — opt-in
    langsmith_api_key: SecretStr | None = None
    langsmith_tracing: bool = False
    langsmith_project: str | None = None

    @field_validator("allowed_telegram_user_ids", mode="before")
    @classmethod
    def _parse_user_ids(cls, value: object) -> object:
        """Parse a CSV string into list[int]; pass through other inputs unchanged."""
        if isinstance(value, str):
            return [int(part.strip()) for part in value.split(",") if part.strip()]
        return value
