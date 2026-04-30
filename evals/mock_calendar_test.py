"""Tests for eval.mock_calendar."""

from __future__ import annotations

from evals.mock_calendar import MockCalendarClient


async def test_list_events_returns_today_events_for_today_date() -> None:
    client = MockCalendarClient().get_mock()

    events = await client.list_events(time_min="2026-04-29T00:00:00+09:00")

    assert len(events) == 2
    assert {e["summary"] for e in events} == {"Team Meeting", "Lunch with Alex"}


async def test_list_events_returns_friday_events_for_friday_date() -> None:
    client = MockCalendarClient().get_mock()

    events = await client.list_events(time_min="2026-05-02T00:00:00+09:00")

    assert len(events) == 3


async def test_search_events_finds_match_in_summary() -> None:
    client = MockCalendarClient().get_mock()

    events = await client.search_events(query="Alex")

    assert len(events) == 1
    assert events[0]["summary"] == "Lunch with Alex"


async def test_get_event_returns_event_by_id() -> None:
    client = MockCalendarClient().get_mock()

    event = await client.get_event(event_id="event3")

    assert event is not None
    assert event["summary"] == "Project Review"


async def test_get_event_returns_none_for_unknown_id() -> None:
    client = MockCalendarClient().get_mock()

    event = await client.get_event(event_id="does-not-exist")

    assert event is None


async def test_get_timezone_returns_real_string_not_async_mock() -> None:
    # Regression: build_agent will await this if no timezone override is given.
    client = MockCalendarClient().get_mock()

    tz = await client.get_timezone()

    assert tz == "Asia/Tokyo"
