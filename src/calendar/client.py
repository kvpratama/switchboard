"""Async wrapper over the Google Calendar v3 API."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from src.auth.google_oauth import load_credentials


class CalendarClientError(RuntimeError):
    """Raised when a Calendar API call fails."""


def _compact(event: dict[str, Any]) -> dict[str, Any]:
    """Return a small, LLM-friendly view of an event."""
    start = event.get("start", {})
    end = event.get("end", {})
    return {
        "id": event.get("id"),
        "summary": event.get("summary"),
        "description": event.get("description"),
        "start": start.get("dateTime") or start.get("date"),
        "end": end.get("dateTime") or end.get("date"),
        "location": event.get("location"),
    }


def _strip_nones(d: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if v is not None}


class GoogleCalendarClient:
    """Thin async wrapper around googleapiclient for the primary calendar."""

    def __init__(self, token_path: Path, calendar_id: str = "primary") -> None:
        """Build the underlying Calendar service.

        Args:
            token_path: Path to the OAuth token JSON written by bootstrap.
            calendar_id: Calendar to operate on. v1 hardcodes ``primary``.
        """
        creds = load_credentials(token_path)
        # cache_discovery=False avoids file-cache warnings on read-only filesystems.
        self._service = build("calendar", "v3", credentials=creds, cache_discovery=False)
        self._calendar_id = calendar_id

    async def list_events(
        self,
        *,
        time_min: str,
        time_max: str,
        max_results: int = 50,
    ) -> list[dict[str, Any]]:
        """List events in [time_min, time_max) on the configured calendar."""

        def _call() -> dict[str, Any]:
            return (
                self._service.events()
                .list(
                    calendarId=self._calendar_id,
                    timeMin=time_min,
                    timeMax=time_max,
                    maxResults=min(max_results, 250),
                    singleEvents=True,
                    orderBy="startTime",
                )
                .execute()
            )

        try:
            response = await asyncio.to_thread(_call)
        except HttpError as exc:
            raise CalendarClientError(f"list_events failed: {exc}") from exc

        return [_strip_nones(_compact(e)) for e in response.get("items", [])]

    async def search_events(
        self,
        *,
        query: str,
        time_min: str,
        time_max: str,
        max_results: int = 20,
    ) -> list[dict[str, Any]]:
        """Free-text search over event summary/description/location."""

        def _call() -> dict[str, Any]:
            return (
                self._service.events()
                .list(
                    calendarId=self._calendar_id,
                    q=query,
                    timeMin=time_min,
                    timeMax=time_max,
                    maxResults=min(max_results, 100),
                    singleEvents=True,
                    orderBy="startTime",
                )
                .execute()
            )

        try:
            response = await asyncio.to_thread(_call)
        except HttpError as exc:
            raise CalendarClientError(f"search_events failed: {exc}") from exc

        return [_strip_nones(_compact(e)) for e in response.get("items", [])]

    async def get_event(self, event_id: str) -> dict[str, Any]:
        """Fetch a single event by ID."""

        def _call() -> dict[str, Any]:
            return (
                self._service.events().get(calendarId=self._calendar_id, eventId=event_id).execute()
            )

        try:
            response = await asyncio.to_thread(_call)
        except HttpError as exc:
            raise CalendarClientError(f"get_event failed: {exc}") from exc

        return _strip_nones(_compact(response))
