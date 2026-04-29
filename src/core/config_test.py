"""Tests for src.core.config.Settings."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.core.config import Settings


def test_settings_loads_from_env(settings_env) -> None:
    settings = Settings()

    assert settings.telegram_bot_token.get_secret_value() == "test-bot-token"
    assert settings.allowed_telegram_user_ids == [111, 222]
    assert settings.llm_provider == "openai"
    assert settings.llm_model == "gpt-4o-mini"
    assert settings.llm_api_key.get_secret_value() == "test-llm-key"
    assert settings.default_timezone == "Asia/Jakarta"
    assert settings.log_level == "INFO"
    assert settings.llm_provider_base_url is None


def test_allowed_user_ids_parses_csv_with_whitespace(
    monkeypatch: pytest.MonkeyPatch, settings_env
) -> None:
    monkeypatch.setenv("ALLOWED_TELEGRAM_USER_IDS", " 111 , 222 ,333 ")

    settings = Settings()

    assert settings.allowed_telegram_user_ids == [111, 222, 333]


def test_missing_required_var_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    # Do NOT request settings_env — start from empty env
    for var in (
        "TELEGRAM_BOT_TOKEN",
        "ALLOWED_TELEGRAM_USER_IDS",
        "LLM_PROVIDER",
        "LLM_MODEL",
        "LLM_API_KEY",
        "GOOGLE_OAUTH_CLIENT_SECRETS_PATH",
    ):
        monkeypatch.delenv(var, raising=False)

    with pytest.raises(ValidationError):
        Settings()


def test_optional_paths_default(settings_env, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GOOGLE_OAUTH_TOKEN_PATH", raising=False)
    monkeypatch.delenv("CHECKPOINT_DB_PATH", raising=False)

    settings = Settings()

    assert str(settings.google_oauth_token_path).endswith("data/token.json")
    assert str(settings.checkpoint_db_path).endswith("data/checkpoints.sqlite")


def test_langsmith_defaults_disabled(settings_env) -> None:
    settings = Settings()

    assert settings.langsmith_tracing is False
    assert settings.langsmith_api_key is None
