"""Mock calendar event fixtures for evaluation."""

# Mock events for 2026-04-29 (today in the test scenario)
TODAY_EVENTS = [
    {
        "id": "event1",
        "summary": "Team Meeting",
        "start": {"dateTime": "2026-04-29T10:00:00+09:00"},
        "end": {"dateTime": "2026-04-29T11:00:00+09:00"},
        "location": "Conference Room A",
    },
    {
        "id": "event2",
        "summary": "Lunch with Alex",
        "start": {"dateTime": "2026-04-29T12:30:00+09:00"},
        "end": {"dateTime": "2026-04-29T13:30:00+09:00"},
        "location": "Cafe Downtown",
    },
]

# Mock events for 2026-04-30 (tomorrow)
TOMORROW_EVENTS = [
    {
        "id": "event3",
        "summary": "Project Review",
        "start": {"dateTime": "2026-04-30T14:00:00+09:00"},
        "end": {"dateTime": "2026-04-30T15:00:00+09:00"},
        "location": "Zoom",
    },
    {
        "id": "event4",
        "summary": "1-on-1 with Sarah",
        "start": {"dateTime": "2026-04-30T16:00:00+09:00"},
        "end": {"dateTime": "2026-04-30T16:30:00+09:00"},
    },
]

# Mock events for 2026-05-02 (Friday)
FRIDAY_EVENTS = [
    {
        "id": "event5",
        "summary": "Client Call",
        "start": {"dateTime": "2026-05-02T09:00:00+09:00"},
        "end": {"dateTime": "2026-05-02T10:00:00+09:00"},
    },
    {
        "id": "event6",
        "summary": "Team Lunch",
        "start": {"dateTime": "2026-05-02T12:00:00+09:00"},
        "end": {"dateTime": "2026-05-02T13:00:00+09:00"},
        "location": "Italian Restaurant",
    },
    {
        "id": "event7",
        "summary": "Code Review",
        "start": {"dateTime": "2026-05-02T15:00:00+09:00"},
        "end": {"dateTime": "2026-05-02T16:00:00+09:00"},
    },
]

# All events combined for search
ALL_EVENTS = TODAY_EVENTS + TOMORROW_EVENTS + FRIDAY_EVENTS


def get_events_by_date(date_str: str) -> list[dict]:
    """Get events for a specific date (YYYY-MM-DD format)."""
    if date_str == "2026-04-29":
        return TODAY_EVENTS
    elif date_str == "2026-04-30":
        return TOMORROW_EVENTS
    elif date_str == "2026-05-02":
        return FRIDAY_EVENTS
    return []


def search_events_by_query(query: str) -> list[dict]:
    """Search events by text query."""
    query_lower = query.lower()
    return [
        event
        for event in ALL_EVENTS
        if query_lower in str(event.get("summary", "")).lower()
        or query_lower in str(event.get("location", "")).lower()
    ]
