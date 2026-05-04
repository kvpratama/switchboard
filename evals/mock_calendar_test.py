"""Tests for eval.mock_calendar."""

from __future__ import annotations

from evals.mock_calendar import MockCalendarClient


async def test_list_events_returns_today_events_for_today_date() -> None:
    client = MockCalendarClient().get_mock()

    events = await client.list_events(
        time_min="2026-04-29T00:00:00+09:00",
        time_max="2026-04-29T23:59:59+09:00",
    )

    assert len(events) == 2
    assert {e["summary"] for e in events} == {"Team Meeting", "Lunch with Alex"}


async def test_list_events_returns_friday_events_for_friday_date() -> None:
    client = MockCalendarClient().get_mock()

    events = await client.list_events(
        time_min="2026-05-01T00:00:00+09:00",
        time_max="2026-05-01T23:59:59+09:00",
    )

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


async def test_create_event_records_call_and_returns_event() -> None:
    wrapper = MockCalendarClient()
    client = wrapper.get_mock()

    result = await client.create_event(
        summary="Lunch",
        start="2026-04-30T13:00:00+09:00",
        end="2026-04-30T14:00:00+09:00",
    )

    assert result["summary"] == "Lunch"
    assert result["start"]["dateTime"] == "2026-04-30T13:00:00+09:00"
    assert result["end"]["dateTime"] == "2026-04-30T14:00:00+09:00"
    assert result["id"].startswith("mock-event-")


async def test_create_event_records_optional_fields() -> None:
    wrapper = MockCalendarClient()
    client = wrapper.get_mock()

    result = await client.create_event(
        summary="Dentist",
        start="2026-05-04T15:00:00+09:00",
        end="2026-05-04T16:00:00+09:00",
        description="Bring insurance card",
        location="Downtown Dental",
    )

    assert result["description"] == "Bring insurance card"
    assert result["location"] == "Downtown Dental"


async def test_get_created_events_returns_recorded_calls() -> None:
    wrapper = MockCalendarClient()
    client = wrapper.get_mock()

    await client.create_event(
        summary="Lunch",
        start="2026-04-30T13:00:00+09:00",
        end="2026-04-30T14:00:00+09:00",
    )
    await client.create_event(
        summary="Dentist",
        start="2026-05-04T15:00:00+09:00",
        end="2026-05-04T16:00:00+09:00",
    )

    created = wrapper.get_created_events()
    assert len(created) == 2
    assert created[0]["summary"] == "Lunch"
    assert created[1]["summary"] == "Dentist"


async def test_reset_clears_created_events() -> None:
    wrapper = MockCalendarClient()
    client = wrapper.get_mock()

    await client.create_event(
        summary="Lunch",
        start="2026-04-30T13:00:00+09:00",
        end="2026-04-30T14:00:00+09:00",
    )

    wrapper.reset()
    assert wrapper.get_created_events() == []


async def test_get_created_events_empty_by_default() -> None:
    wrapper = MockCalendarClient()
    assert wrapper.get_created_events() == []
