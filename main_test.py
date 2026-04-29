"""Smoke tests for main.py composition root."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from src.core.config import Settings


def test_main_module_imports() -> None:
    import main  # noqa: F401


async def test_run_async_assembles_app_and_starts_polling(settings_env, mocker, tmp_path) -> None:
    # Pre-create checkpoint dir so AsyncSqliteSaver context manager succeeds.
    (tmp_path / "checkpoints.sqlite").parent.mkdir(parents=True, exist_ok=True)

    fake_checkpointer = MagicMock(name="checkpointer")
    mocker.patch(
        "main.AsyncSqliteSaver.from_conn_string",
        return_value=_AsyncContextManager(fake_checkpointer),
    )

    fake_agent = MagicMock(name="agent")
    build_agent = mocker.patch("main.build_agent", return_value=fake_agent)

    fake_app = MagicMock(name="application")
    fake_app.__aenter__ = AsyncMock(return_value=fake_app)
    fake_app.__aexit__ = AsyncMock(return_value=False)
    fake_app.initialize = AsyncMock()
    fake_app.start = AsyncMock()
    fake_app.stop = AsyncMock()
    fake_app.updater.start_polling = AsyncMock()
    fake_app.updater.stop = AsyncMock()
    build_application = mocker.patch("main.build_application", return_value=fake_app)

    # Make asyncio.Event().wait() return immediately so the test doesn't hang.
    fake_event = MagicMock()
    fake_event.wait = AsyncMock()
    mocker.patch("main.asyncio.Event", return_value=fake_event)

    import main

    settings = Settings()
    await main._run_async(settings)

    build_agent.assert_called_once()
    assert build_agent.call_args.kwargs["checkpointer"] is fake_checkpointer
    build_application.assert_called_once()
    assert build_application.call_args.kwargs["agent"] is fake_agent
    fake_app.initialize.assert_awaited_once()
    fake_app.start.assert_awaited_once()
    fake_app.updater.start_polling.assert_awaited_once()
    fake_app.updater.stop.assert_awaited_once()
    fake_app.stop.assert_awaited_once()


class _AsyncContextManager:
    def __init__(self, value) -> None:
        self.value = value

    async def __aenter__(self):
        return self.value

    async def __aexit__(self, *exc) -> bool:
        return False


def test_apply_langsmith_env_when_tracing_enabled(settings_env, monkeypatch) -> None:
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls-key")
    monkeypatch.setenv("LANGSMITH_PROJECT", "switchboard-test")

    from main import apply_langsmith_env

    settings = Settings()
    apply_langsmith_env(settings)

    import os

    assert os.environ["LANGSMITH_TRACING"] == "true"
    assert os.environ["LANGSMITH_API_KEY"] == "ls-key"
    assert os.environ["LANGSMITH_PROJECT"] == "switchboard-test"


def test_apply_langsmith_env_noop_when_disabled(settings_env, monkeypatch) -> None:
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)

    from main import apply_langsmith_env

    settings = Settings()
    apply_langsmith_env(settings)

    import os

    assert "LANGSMITH_API_KEY" not in os.environ
