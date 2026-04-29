"""Dynamic mock calendar client for evaluation."""

from unittest.mock import AsyncMock

from eval.fixtures import ALL_EVENTS, get_events_by_date, search_events_by_query


class MockCalendarClient:
    """Mock calendar client that returns dynamic data based on query parameters."""

    def __init__(self):
        """Initialize mock client."""
        self._mock = AsyncMock()
        self._mock.list_events.side_effect = self._list_events
        self._mock.search_events.side_effect = self._search_events
        self._mock.get_event.side_effect = self._get_event

    async def _list_events(
        self, time_min: str | None = None, time_max: str | None = None, **kwargs
    ) -> list[dict]:
        """Mock list_events - returns events based on date range."""
        if not time_min:
            return ALL_EVENTS

        # Extract date from ISO timestamp (YYYY-MM-DD)
        date_str = time_min.split("T")[0]
        return get_events_by_date(date_str)

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
