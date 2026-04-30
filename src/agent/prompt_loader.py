"""Fetches the system prompt template from LangSmith with TTL caching."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from datetime import datetime

from langsmith import Client

log = logging.getLogger(__name__)

FALLBACK_SYSTEM_TEMPLATE = (
    "You are Switchboard, a helpful assistant that answers questions "
    "about the user's Google Calendar.\n"
    "The current local time is {current_time} ({timezone}), {day_of_week}.\n"
    "When the user mentions relative times (today, tomorrow, this week, "
    "next Monday, etc.), resolve them against the current local time and "
    "always pass ISO 8601 datetimes WITH the user's timezone offset to "
    "your tools.\n"
    "Be concise. If a tool returns an error, relay a short, friendly "
    "explanation to the user — do not retry blindly."
)

FALLBACK_ACCURACY_TEMPLATE = (
    "Compare the actual response to the expected response. "
    "The actual response should convey the same information as "
    "expected, but does not need to match word-for-word.\n\n"
    "Expected: {expected_response}\n"
    "Actual: {actual_response}\n\n"
    "Is the actual response accurate?"
)

_FALLBACK_REGISTRY: dict[str, str] = {
    "switchboard-system": FALLBACK_SYSTEM_TEMPLATE,
    "switchboard-accuracy-evaluator": FALLBACK_ACCURACY_TEMPLATE,
}


class PromptLoader:
    """Fetches the system prompt template from LangSmith with TTL caching.

    - If cached value is fresh (within TTL), returns it.
    - Otherwise, refetches from LangSmith via ``asyncio.to_thread``.
    - On a refresh failure, logs a warning and keeps serving the cached value.
    - On a first-fetch failure, logs an error and returns ``fallback_template``.

    Args:
        prompt_name: Name of the prompt in LangSmith Prompt Hub. Used to
            look up a fallback template from the built-in registry.
        fallback_template: Template string used when LangSmith is unreachable
            on the very first fetch. If ``None``, the fallback is looked up
            from the registry by ``prompt_name``; raises ``KeyError`` if the
            name is not registered.
        ttl_seconds: Time-to-live for the cached template in seconds.
        client: Optional LangSmith ``Client`` instance. If ``None``, a default
            client is created (reads ``LANGSMITH_API_KEY`` from the environment).
        now_provider: Optional callable returning the current ``datetime``.
            Injected so tests can freeze time deterministically.
    """

    def __init__(
        self,
        *,
        prompt_name: str,
        fallback_template: str | None = None,
        ttl_seconds: int = 300,
        client: Client | None = None,
        now_provider: Callable[[], datetime] | None = None,
    ) -> None:
        self._prompt_name = prompt_name
        if fallback_template is not None:
            self._fallback_template = fallback_template
        elif prompt_name in _FALLBACK_REGISTRY:
            self._fallback_template = _FALLBACK_REGISTRY[prompt_name]
        else:
            raise KeyError(
                f"No fallback template registered for prompt '{prompt_name}'. "
                "Provide fallback_template explicitly or register one in "
                "_FALLBACK_REGISTRY."
            )
        self._ttl_seconds = ttl_seconds
        self._client = client or Client()
        self._now_provider = now_provider or datetime.now
        self._cache: tuple[str, datetime] | None = None

    def _is_cache_fresh(self) -> bool:
        """Return ``True`` if the cached template is within TTL."""
        if self._cache is None:
            return False
        _, fetched_at = self._cache
        elapsed = (self._now_provider() - fetched_at).total_seconds()
        return elapsed < self._ttl_seconds

    async def get_template(self) -> str:
        """Return the current template string.

        - If cached value is fresh (within TTL), returns it.
        - Otherwise, refetches from LangSmith via ``asyncio.to_thread``.
        - On a refresh failure, logs a warning and keeps serving the cached value.
        - On a first-fetch failure, logs an error and returns ``fallback_template``.
        """
        if self._is_cache_fresh():
            assert self._cache is not None
            return self._cache[0]

        try:
            prompt = await asyncio.to_thread(self._client.pull_prompt, self._prompt_name)
            template_str: str = prompt.template if hasattr(prompt, "template") else str(prompt)
            self._cache = (template_str, self._now_provider())
            return template_str
        except Exception:
            if self._cache is not None:
                log.warning(
                    "Failed to refresh prompt '%s' from LangSmith; serving cached version",
                    self._prompt_name,
                    exc_info=True,
                )
                assert self._cache is not None
                return self._cache[0]
            log.error(
                "Failed to fetch prompt '%s' from LangSmith on first attempt; "
                "using fallback template",
                self._prompt_name,
                exc_info=True,
            )
            self._cache = (self._fallback_template, self._now_provider())
            return self._fallback_template
