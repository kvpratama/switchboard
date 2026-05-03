"""Switchboard entrypoint — loads settings, builds the agent, runs the bot."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from src.agent.builder import build_agent
from src.core.config import Settings
from src.core.logging import configure_logging
from src.telegram.bot import build_application

log = logging.getLogger(__name__)

EVENTS_SCOPE = "https://www.googleapis.com/auth/calendar.events"


def _warn_if_insufficient_scope(token_path: Path) -> None:
    """Log a warning if the saved token lacks the events read/write scope.

    No-op when the token file is absent or unreadable; ``load_credentials``
    will surface a clearer error in those cases.
    """
    if not token_path.exists():
        return
    try:
        data = json.loads(token_path.read_text())
    except Exception:
        return
    scopes = data.get("scopes") or []
    if EVENTS_SCOPE not in scopes:
        log.warning(
            "Saved Google token does not include the %s scope. "
            "Event creation will fail until you re-run "
            "`uv run python -m src.auth.bootstrap`.",
            EVENTS_SCOPE,
        )


async def _run_async(settings: Settings) -> None:
    """Manage the AsyncSqliteSaver lifetime around the bot's polling loop."""
    async with AsyncSqliteSaver.from_conn_string(str(settings.checkpoint_db_path)) as checkpointer:
        agent = await build_agent(settings=settings, checkpointer=checkpointer)
        application = build_application(settings=settings, agent=agent)

        log.info("Starting Switchboard bot — polling for updates")
        async with application:
            await application.start()
            updater = application.updater
            assert updater is not None  # noqa: S101 — Updater is built-in by default
            await updater.start_polling()
            try:
                await asyncio.Event().wait()
            finally:
                await updater.stop()
                await application.stop()


def run() -> None:
    """Compose the application and start long-polling."""
    load_dotenv()
    settings = Settings()
    configure_logging(settings.log_level)

    settings.checkpoint_db_path.parent.mkdir(parents=True, exist_ok=True)
    _warn_if_insufficient_scope(settings.google_oauth_token_path)

    asyncio.run(_run_async(settings))


if __name__ == "__main__":
    run()
