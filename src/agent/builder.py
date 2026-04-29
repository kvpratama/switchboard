"""Factory for the Switchboard LangChain agent."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, dynamic_prompt
from langchain.chat_models import init_chat_model

from src.calendar.client import GoogleCalendarClient
from src.calendar.tools import build_calendar_tools
from src.core.config import Settings

log = logging.getLogger(__name__)


def render_system_prompt(*, timezone: str) -> str:
    """Build the system prompt with the current time + timezone embedded.

    Re-rendered per agent invocation (via the dynamic_prompt middleware) so
    relative dates (today, tomorrow, next Monday) resolve correctly.

    Args:
        timezone: IANA timezone name, e.g. ``Asia/Tokyo``.

    Returns:
        The system prompt string.
    """
    try:
        tz = ZoneInfo(timezone)
    except Exception:
        log.warning(f"Invalid timezone '{timezone}'; falling back to UTC")
        tz = ZoneInfo("UTC")
        timezone = "UTC"

    now = datetime.now(tz)
    return (
        "You are Switchboard, a helpful assistant that answers questions "
        "about the user's Google Calendar.\n"
        f"The current local time is {now.isoformat()} ({timezone}).\n"
        "When the user mentions relative times (today, tomorrow, this week, "
        "next Monday, etc.), resolve them against the current local time and "
        "always pass ISO 8601 datetimes WITH the user's timezone offset to "
        "your tools.\n"
        "Be concise. If a tool returns an error, relay a short, friendly "
        "explanation to the user — do not retry blindly."
    )


async def build_agent(*, settings: Settings, checkpointer: Any) -> Any:
    """Construct the read-only calendar agent.

    Args:
        settings: Application settings.
        checkpointer: A LangGraph checkpointer (e.g. ``SqliteSaver``) used for
            multi-turn conversation memory keyed by Telegram ``chat_id``.

    Returns:
        Compiled agent ready to ``ainvoke``.
    """
    init_kwargs: dict[str, Any] = {
        "model": settings.llm_model,
        "model_provider": settings.llm_provider,
        "api_key": settings.llm_api_key.get_secret_value(),
        "temperature": 0,
    }
    if settings.llm_provider_base_url:
        init_kwargs["base_url"] = settings.llm_provider_base_url

    model = init_chat_model(**init_kwargs)

    client = GoogleCalendarClient(token_path=settings.google_oauth_token_path)
    tools = build_calendar_tools(client)

    timezone: str
    if settings.default_timezone:
        timezone = settings.default_timezone
    else:
        try:
            tz = await client.get_timezone()
            timezone = tz if tz else "UTC"
        except Exception:
            log.warning("Failed to fetch timezone from Calendar API; falling back to UTC")
            timezone = "UTC"

    @dynamic_prompt
    def _system_prompt(request: ModelRequest) -> str:
        return render_system_prompt(timezone=timezone)

    return create_agent(
        model=model,
        tools=tools,
        middleware=[_system_prompt],
        checkpointer=checkpointer,
    )
