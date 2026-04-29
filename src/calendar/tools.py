"""LangChain tools that wrap GoogleCalendarClient.

Each tool returns a JSON string so the LLM can read structured data.
On client errors, tools return an ``Error: ...`` string instead of raising,
so the LLM can relay a friendly message to the user.
"""

from __future__ import annotations

import json
from typing import Any

from langchain.tools import BaseTool, tool

from src.calendar.client import CalendarClientError, GoogleCalendarClient


def build_calendar_tools(client: GoogleCalendarClient) -> list[BaseTool]:
    """Return the three read-only Calendar tools, bound to ``client``.

    Args:
        client: Configured GoogleCalendarClient instance.

    Returns:
        List of LangChain ``BaseTool`` instances suitable for ``create_agent``.
    """

    @tool
    async def list_events(
        time_min: str,
        time_max: str,
        max_results: int = 50,
    ) -> str:
        """List calendar events within a time range on the primary calendar.

        Use this when the user asks about their schedule for a day, range, or
        period (e.g. "what's on today", "do I have anything Friday").

        Args:
            time_min: ISO 8601 datetime, inclusive lower bound (with timezone).
            time_max: ISO 8601 datetime, exclusive upper bound (with timezone).
            max_results: Maximum events to return (default 50, hard cap 250).
        """
        try:
            events = await client.list_events(
                time_min=time_min,
                time_max=time_max,
                max_results=max_results,
            )
        except CalendarClientError as exc:
            return f"Error: {exc}"
        return json.dumps(events)

    @tool
    async def search_events(
        query: str,
        time_min: str,
        time_max: str,
        max_results: int = 20,
    ) -> str:
        """Search events by free-text query on summary, description, location.

        Use this when the user asks about a specific person or topic
        (e.g. "when's my next meeting with Alex?", "find my dentist appointment").

        Args:
            query: Free-text search string.
            time_min: ISO 8601 lower bound (with timezone).
            time_max: ISO 8601 upper bound (with timezone).
            max_results: Maximum events to return (default 20).
        """
        try:
            events = await client.search_events(
                query=query,
                time_min=time_min,
                time_max=time_max,
                max_results=max_results,
            )
        except CalendarClientError as exc:
            return f"Error: {exc}"
        return json.dumps(events)

    @tool
    async def get_event(event_id: str) -> str:
        """Fetch a single event's full details by ID.

        Use this when the user asks for specifics about an event already
        referenced earlier in the conversation.

        Args:
            event_id: The event ID returned by list_events / search_events.
        """
        try:
            event: dict[str, Any] = await client.get_event(event_id)
        except CalendarClientError as exc:
            return f"Error: {exc}"
        return json.dumps(event)

    return [list_events, search_events, get_event]
