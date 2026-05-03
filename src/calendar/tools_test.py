"""Tests for src.calendar.tools."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

from src.calendar.client import CalendarClientError
from src.calendar.tools import build_calendar_tools


async def test_list_events_tool_returns_json_string() -> None:
    client = AsyncMock()
    client.list_events.return_value = [{"id": "1", "summary": "X"}]
    tools = build_calendar_tools(client)
    list_events = next(t for t in tools if t.name == "list_events")

    result = await list_events.ainvoke(
        {
            "time_min": "2026-04-29T00:00:00+07:00",
            "time_max": "2026-04-30T00:00:00+07:00",
        }
    )

    assert json.loads(result) == [{"id": "1", "summary": "X"}]
    client.list_events.assert_awaited_once_with(
        time_min="2026-04-29T00:00:00+07:00",
        time_max="2026-04-30T00:00:00+07:00",
        max_results=50,
    )


async def test_search_events_tool_passes_query() -> None:
    client = AsyncMock()
    client.search_events.return_value = []
    tools = build_calendar_tools(client)
    search_events = next(t for t in tools if t.name == "search_events")

    await search_events.ainvoke(
        {
            "query": "Alex",
            "time_min": "2026-04-29T00:00:00+07:00",
            "time_max": "2026-07-29T00:00:00+07:00",
        }
    )

    client.search_events.assert_awaited_once_with(
        query="Alex",
        time_min="2026-04-29T00:00:00+07:00",
        time_max="2026-07-29T00:00:00+07:00",
        max_results=20,
    )


async def test_get_event_tool_returns_json_string() -> None:
    client = AsyncMock()
    client.get_event.return_value = {"id": "evt-1", "summary": "X"}
    tools = build_calendar_tools(client)
    get_event = next(t for t in tools if t.name == "get_event")

    result = await get_event.ainvoke({"event_id": "evt-1"})

    assert json.loads(result) == {"id": "evt-1", "summary": "X"}


async def test_tool_returns_error_string_on_client_failure() -> None:
    client = AsyncMock()
    client.list_events.side_effect = CalendarClientError("boom")
    tools = build_calendar_tools(client)
    list_events = next(t for t in tools if t.name == "list_events")

    result = await list_events.ainvoke(
        {
            "time_min": "2026-04-29T00:00:00+07:00",
            "time_max": "2026-04-30T00:00:00+07:00",
        }
    )

    assert result == "Unable to retrieve events. Please check your calendar permissions."


def test_build_calendar_tools_returns_four_tools() -> None:
    client = AsyncMock()
    tools = build_calendar_tools(client)
    names = sorted(t.name for t in tools)
    assert names == ["create_event", "get_event", "list_events", "search_events"]


async def test_create_event_tool_returns_json_string() -> None:
    client = AsyncMock()
    client.create_event.return_value = {
        "id": "evt-new",
        "summary": "Lunch",
        "start": "2026-05-03T13:00:00+09:00",
        "end": "2026-05-03T14:00:00+09:00",
    }
    tools = build_calendar_tools(client)
    create_event = next(t for t in tools if t.name == "create_event")

    result = await create_event.ainvoke(
        {
            "summary": "Lunch",
            "start": "2026-05-03T13:00:00+09:00",
            "end": "2026-05-03T14:00:00+09:00",
        }
    )

    assert json.loads(result)["id"] == "evt-new"
    client.create_event.assert_awaited_once_with(
        summary="Lunch",
        start="2026-05-03T13:00:00+09:00",
        end="2026-05-03T14:00:00+09:00",
        description=None,
        location=None,
    )


async def test_create_event_tool_passes_optional_fields() -> None:
    client = AsyncMock()
    client.create_event.return_value = {"id": "evt-new"}
    tools = build_calendar_tools(client)
    create_event = next(t for t in tools if t.name == "create_event")

    await create_event.ainvoke(
        {
            "summary": "Lunch",
            "start": "2026-05-03T13:00:00+09:00",
            "end": "2026-05-03T14:00:00+09:00",
            "description": "with Sarah",
            "location": "Cafe",
        }
    )

    client.create_event.assert_awaited_once_with(
        summary="Lunch",
        start="2026-05-03T13:00:00+09:00",
        end="2026-05-03T14:00:00+09:00",
        description="with Sarah",
        location="Cafe",
    )


async def test_create_event_tool_returns_error_string_on_failure() -> None:
    client = AsyncMock()
    client.create_event.side_effect = CalendarClientError("boom")
    tools = build_calendar_tools(client)
    create_event = next(t for t in tools if t.name == "create_event")

    result = await create_event.ainvoke(
        {
            "summary": "Lunch",
            "start": "2026-05-03T13:00:00+09:00",
            "end": "2026-05-03T14:00:00+09:00",
        }
    )

    assert result == "Unable to create event. Please check your calendar permissions."
