"""Dynamic mock calendar client for evaluation."""

from unittest.mock import AsyncMock

from evals.fixtures import ALL_EVENTS, search_events_by_query


class MockCalendarClient:
    """Mock calendar client that returns dynamic data based on query parameters."""

    def __init__(self):
        """Initialize mock client."""
        self._mock = AsyncMock()
        self._mock.list_events.side_effect = self._list_events
        self._mock.search_events.side_effect = self._search_events
        self._mock.get_event.side_effect = self._get_event
        # Defensive default: build_agent may call get_timezone() if no
        # timezone override is passed. Returning a real string here avoids
        # AsyncMock surprise objects propagating into the prompt.
        self._mock.get_timezone.return_value = "Asia/Tokyo"

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

    def get_mock(self) -> AsyncMock:
        """Get the underlying AsyncMock for tool creation."""
        return self._mock
