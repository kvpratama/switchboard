"""Factory for the Switchboard LangChain agent."""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, dynamic_prompt
from langchain.chat_models import init_chat_model

from src.agent.prompt_loader import PromptLoader
from src.calendar.client import GoogleCalendarClient
from src.calendar.tools import build_calendar_tools
from src.core.config import Settings

log = logging.getLogger(__name__)


def render_system_prompt(
    *,
    template: str,
    timezone: str,
    now_provider: Callable[[], datetime] | None = None,
) -> str:
    """Build the system prompt with the current time + timezone embedded.

    Re-rendered per agent invocation (via the dynamic_prompt middleware) so
    relative dates (today, tomorrow, next Monday) resolve correctly.

    Args:
        template: Prompt template string with ``{current_time}``,
            ``{timezone}``, and ``{day_of_week}`` placeholders.
        timezone: IANA timezone name, e.g. ``Asia/Tokyo``.
        now_provider: Optional callable returning the current ``datetime``.
            Used by the eval harness to freeze time for reproducibility.
            If provided and the returned datetime has no tzinfo, it is
            interpreted in ``timezone``; otherwise it is converted to it.
            Defaults to ``datetime.now(tz)``.

    Returns:
        The system prompt string.
    """
    try:
        tz = ZoneInfo(timezone)
    except Exception:
        log.warning(f"Invalid timezone '{timezone}'; falling back to UTC")
        tz = ZoneInfo("UTC")
        timezone = "UTC"

    if now_provider is None:
        now = datetime.now(tz)
    else:
        provided = now_provider()
        now = provided.astimezone(tz) if provided.tzinfo else provided.replace(tzinfo=tz)
    return template.format(
        current_time=now.isoformat(),
        timezone=timezone,
        day_of_week=now.strftime("%A"),
    )


async def build_agent(
    *,
    settings: Settings,
    checkpointer: Any,
    calendar_client: GoogleCalendarClient | None = None,
    timezone: str | None = None,
    now_provider: Callable[[], datetime] | None = None,
) -> Any:
    """Construct the read-only calendar agent.

    Args:
        settings: Application settings.
        checkpointer: A LangGraph checkpointer (e.g. ``SqliteSaver``) used for
            multi-turn conversation memory keyed by Telegram ``chat_id``.
        calendar_client: Optional calendar client (for testing/mocking). If None,
            creates a real GoogleCalendarClient.
        timezone: Optional timezone override (for testing). If None, uses
            settings.default_timezone or fetches from Calendar API.
        now_provider: Optional callable returning the current ``datetime``.
            Used by the eval harness to freeze time for reproducibility.

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

    # Use provided client or create real one
    if calendar_client is None:
        calendar_client = GoogleCalendarClient(token_path=settings.google_oauth_token_path)

    tools = build_calendar_tools(calendar_client)

    # Use provided timezone or determine it
    if timezone is None:
        if settings.default_timezone:
            timezone = settings.default_timezone
        else:
            try:
                tz = await calendar_client.get_timezone()
                timezone = tz if tz else "UTC"
            except Exception:
                log.warning("Failed to fetch timezone from Calendar API; falling back to UTC")
                timezone = "UTC"

    loader = PromptLoader(prompt_name="switchboard-system")

    @dynamic_prompt
    async def _system_prompt(request: ModelRequest) -> str:
        template = await loader.get_template()
        return render_system_prompt(template=template, timezone=timezone, now_provider=now_provider)

    return create_agent(
        model=model,
        tools=tools,
        middleware=[_system_prompt],
        checkpointer=checkpointer,
    )
