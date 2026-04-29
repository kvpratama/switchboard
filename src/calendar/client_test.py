"""Tests for src.calendar.client."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from googleapiclient.errors import HttpError

from src.calendar.client import CalendarClientError, GoogleCalendarClient


@pytest.fixture
def fake_service(mocker) -> MagicMock:
    """Patch googleapiclient.discovery.build to return a controllable mock."""
    service = MagicMock(name="service")
    mocker.patch("src.calendar.client.build", return_value=service)
    return service


@pytest.fixture
def fake_credentials(mocker) -> MagicMock:
    creds = MagicMock(name="credentials")
    mocker.patch("src.calendar.client.load_credentials", return_value=creds)
    return creds


async def test_list_events_returns_compact_payload(
    tmp_path, fake_service: MagicMock, fake_credentials: MagicMock
) -> None:
    fake_service.events.return_value.list.return_value.execute.return_value = {
        "items": [
            {
                "id": "evt-1",
                "summary": "Standup",
                "start": {"dateTime": "2026-04-29T09:00:00+07:00"},
                "end": {"dateTime": "2026-04-29T09:15:00+07:00"},
                "location": "Zoom",
                "extra": "ignored",
            }
        ]
    }
    client = GoogleCalendarClient(token_path=tmp_path / "token.json")

    events = await client.list_events(
        time_min="2026-04-29T00:00:00+07:00",
        time_max="2026-04-30T00:00:00+07:00",
    )

    assert events == [
        {
            "id": "evt-1",
            "summary": "Standup",
            "start": "2026-04-29T09:00:00+07:00",
            "end": "2026-04-29T09:15:00+07:00",
            "location": "Zoom",
        }
    ]
    fake_service.events.return_value.list.assert_called_once()
    kwargs = fake_service.events.return_value.list.call_args.kwargs
    assert kwargs["calendarId"] == "primary"
    assert kwargs["singleEvents"] is True
    assert kwargs["orderBy"] == "startTime"


async def test_search_events_passes_query(
    tmp_path, fake_service: MagicMock, fake_credentials: MagicMock
) -> None:
    fake_service.events.return_value.list.return_value.execute.return_value = {"items": []}
    client = GoogleCalendarClient(token_path=tmp_path / "token.json")

    await client.search_events(
        query="Alex",
        time_min="2026-04-29T00:00:00+07:00",
        time_max="2026-07-29T00:00:00+07:00",
    )

    kwargs = fake_service.events.return_value.list.call_args.kwargs
    assert kwargs["q"] == "Alex"


async def test_get_event_returns_compact_payload(
    tmp_path, fake_service: MagicMock, fake_credentials: MagicMock
) -> None:
    fake_service.events.return_value.get.return_value.execute.return_value = {
        "id": "evt-1",
        "summary": "Standup",
        "description": "Daily sync",
        "start": {"dateTime": "2026-04-29T09:00:00+07:00"},
        "end": {"dateTime": "2026-04-29T09:15:00+07:00"},
    }
    client = GoogleCalendarClient(token_path=tmp_path / "token.json")

    event = await client.get_event("evt-1")

    assert event["id"] == "evt-1"
    assert event["summary"] == "Standup"
    assert event["description"] == "Daily sync"


async def test_http_error_is_wrapped(
    tmp_path, fake_service: MagicMock, fake_credentials: MagicMock
) -> None:
    response = MagicMock(status=403, reason="Forbidden")
    fake_service.events.return_value.list.return_value.execute.side_effect = HttpError(
        resp=response, content=b'{"error":"forbidden"}'
    )
    client = GoogleCalendarClient(token_path=tmp_path / "token.json")

    with pytest.raises(CalendarClientError):
        await client.list_events(
            time_min="2026-04-29T00:00:00+07:00",
            time_max="2026-04-30T00:00:00+07:00",
        )
