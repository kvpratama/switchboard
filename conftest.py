"""Shared pytest fixtures."""

from __future__ import annotations

import pytest


@pytest.fixture
def settings_env(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Populate the minimum env vars required to construct Settings.

    Tests that need a fully-populated environment should request this fixture.
    Tests that explicitly verify missing-var behavior should NOT request it.
    """
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-bot-token")
    monkeypatch.setenv("ALLOWED_TELEGRAM_USER_IDS", "111,222")
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("LLM_API_KEY", "test-llm-key")
    monkeypatch.setenv(
        "GOOGLE_OAUTH_CLIENT_SECRETS_PATH",
        str(tmp_path / "credentials.json"),
    )
    monkeypatch.setenv("GOOGLE_OAUTH_TOKEN_PATH", str(tmp_path / "token.json"))
    monkeypatch.setenv("CHECKPOINT_DB_PATH", str(tmp_path / "checkpoints.sqlite"))
    monkeypatch.setenv("DEFAULT_TIMEZONE", "Asia/Jakarta")
