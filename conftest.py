"""Shared pytest fixtures."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# Set CI mode before any imports to prevent .env loading
os.environ["CI"] = "true"


@pytest.fixture
def settings_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Populate the minimum env vars required to construct Settings.

    Tests that need a fully-populated environment should request this fixture.
    Tests that explicitly verify missing-var behavior should NOT request it.
    """
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-bot-token")
    monkeypatch.setenv("ALLOWED_TELEGRAM_USER_IDS", "111,222")
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("LLM_API_KEY", "test-llm-key")
    monkeypatch.setenv("LLM_PROVIDER_EVAL", "openai")
    monkeypatch.setenv("LLM_MODEL_EVAL", "gpt-4o-mini")
    monkeypatch.setenv("LLM_API_KEY_EVAL", "test-llm-key-eval")
    monkeypatch.setenv(
        "GOOGLE_OAUTH_CLIENT_SECRETS_PATH",
        str(tmp_path / "credentials.json"),
    )
    monkeypatch.setenv("GOOGLE_OAUTH_TOKEN_PATH", str(tmp_path / "token.json"))
    monkeypatch.setenv("CHECKPOINT_DB_PATH", str(tmp_path / "checkpoints.sqlite"))
    monkeypatch.setenv("DEFAULT_TIMEZONE", "Asia/Tokyo")
    # Explicitly unset optional variables to prevent .env leakage
    monkeypatch.delenv("LLM_PROVIDER_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_PROVIDER_BASE_URL_EVAL", raising=False)
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)
