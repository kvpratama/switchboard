"""Dynamic mock calendar client for evaluation."""

import contextvars
import uuid
from typing import Any
from unittest.mock import AsyncMock

from evals.fixtures import ALL_EVENTS, search_events_by_query


class MockCalendarClient:
    """Mock calendar client that returns dynamic data based on query parameters."""

    def __init__(self, timezone: str = "Asia/Tokyo", **kwargs: Any) -> None:
        """Initialize mock client.

        Args:
            timezone: IANA timezone string returned by ``get_timezone``.
            **kwargs: Additional keyword arguments are accepted and ignored
                so this mock can stand in for the real calendar client.
        """
        self._timezone = timezone
        self._mock = AsyncMock()
        self._mock.list_events.side_effect = self._list_events
        self._mock.search_events.side_effect = self._search_events
        self._mock.get_event.side_effect = self._get_event
        self._mock.create_event.side_effect = self._create_event
        # Defensive default: build_agent may call get_timezone() if no
        # timezone override is passed. Returning a real string here avoids
        # AsyncMock surprise objects propagating into the prompt.
        self._mock.get_timezone.return_value = self._timezone
        # Per-task storage so concurrent eval runs (aevaluate max_concurrency>1)
        # do not clobber each other's recorded calls. Each asyncio task gets its
        # own list via ContextVar.
        self._created_events_var: contextvars.ContextVar[list[dict[str, Any]] | None] = (
            contextvars.ContextVar(f"created_events_{id(self)}", default=None)
        )

    def _get_created_events_list(self) -> list[dict[str, Any]]:
        """Return the recording list for the current task, creating it on first use."""
        lst = self._created_events_var.get()
        if lst is None:
            lst = []
            self._created_events_var.set(lst)
        return lst

    async def _list_events(
        self, time_min: str | None = None, time_max: str | None = None, **kwargs
    ) -> list[dict]:
        """Mock list_events - returns events based on date range."""
        if not time_min:
            return ALL_EVENTS

        # Filter events by time range
        filtered = []
        for event in ALL_EVENTS:
            event_start = event["start"].get("dateTime")
            if not event_start:
                continue

            # Check if event is within range
            if event_start >= time_min:
                if time_max is None or event_start < time_max:
                    filtered.append(event)

        return filtered

    async def _search_events(self, query: str, **kwargs) -> list[dict]:
        """Mock search_events - returns events matching query text."""
        return search_events_by_query(query)

    async def _get_event(self, event_id: str, **kwargs) -> dict | None:
        """Mock get_event - returns specific event by ID."""
        for event in ALL_EVENTS:
            if event["id"] == event_id:
                return event
        return None

    async def _create_event(
        self,
        *,
        summary: str,
        start: str,
        end: str,
        description: str | None = None,
        location: str | None = None,
    ) -> dict[str, Any]:
        """Mock create_event — records the call and returns a Google Calendar-style event dict."""
        call_record: dict[str, Any] = {
            "summary": summary,
            "start": start,
            "end": end,
        }
        if description is not None:
            call_record["description"] = description
        if location is not None:
            call_record["location"] = location
        self._get_created_events_list().append(call_record)

        result: dict[str, Any] = {
            "id": f"mock-event-{uuid.uuid4()}",
            "summary": summary,
            "start": {"dateTime": start},
            "end": {"dateTime": end},
        }
        if description is not None:
            result["description"] = description
        if location is not None:
            result["location"] = location
        return result

    def get_created_events(self) -> list[dict[str, Any]]:
        """Return all create_event calls recorded since the last reset.

        Recordings are scoped to the current asyncio task so concurrent
        eval runs do not see each other's events.
        """
        lst = self._created_events_var.get()
        return list(lst) if lst is not None else []

    def reset(self) -> None:
        """Clear recorded create_event calls for the current task."""
        self._created_events_var.set([])

    def get_mock(self) -> AsyncMock:
        """Get the underlying AsyncMock for tool creation."""
        return self._mock
