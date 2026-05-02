"""Fetches prompt templates from LangSmith with the SDK's built-in cache.

This module configures the LangSmith global prompt cache once on import (TTL =
300 s, stale-while-revalidate). ``PromptLoader`` is a thin async wrapper around
``Client.pull_prompt`` that adds:

1. A fallback-template registry used when LangSmith is unreachable on the very
   first fetch.
2. Memoization of the last successful template, so a transient refresh failure
   keeps serving the last good version instead of falling back.
"""

from __future__ import annotations

import asyncio
import logging

from langsmith import Client
from langsmith.prompt_cache import configure_global_prompt_cache

log = logging.getLogger(__name__)

# Stale-while-revalidate: pulls served from cache for 300 s, then refreshed in
# the background on the next call. Configured once for the whole process.
configure_global_prompt_cache(ttl_seconds=300)

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
    """Async wrapper around ``Client.pull_prompt`` with fallback handling.

    Caching is delegated entirely to the LangSmith SDK's global prompt cache
    (configured at module import time). This class only adds:

    - First-fetch fallback: if LangSmith is unreachable on the very first call
      and we have nothing to serve, return ``fallback_template``.
    - Last-success memoization: if a later refresh fails, return the most
      recently fetched template instead of falling back.

    Args:
        prompt_name: Name of the prompt in LangSmith Prompt Hub. Used to look
            up a fallback template from the built-in registry when
            ``fallback_template`` is not provided.
        fallback_template: Template string used when LangSmith is unreachable
            on the very first fetch. If ``None``, the fallback is looked up
            from the registry by ``prompt_name``; raises ``KeyError`` if the
            name is not registered.
        client: Optional LangSmith ``Client``. If ``None``, a default client is
            created (reads ``LANGSMITH_API_KEY`` from the environment).
    """

    def __init__(
        self,
        *,
        prompt_name: str,
        fallback_template: str | None = None,
        client: Client | None = None,
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
        self._client = client or Client()
        self._last_template: str | None = None

    async def get_template(self) -> str:
        """Return the current template string.

        - Calls ``client.pull_prompt`` (which is served from the LangSmith
          SDK's global in-memory cache when fresh).
        - On failure, returns the last successfully fetched template if any,
          otherwise the configured fallback template.
        """
        try:
            prompt = await asyncio.to_thread(self._client.pull_prompt, self._prompt_name)
            template_str: str = prompt.template
            self._last_template = template_str
            return template_str
        except Exception:
            if self._last_template is not None:
                log.warning(
                    "Failed to refresh prompt '%s' from LangSmith; serving last successful version",
                    self._prompt_name,
                    # exc_info=True,
                )
                return self._last_template
            log.error(
                "Failed to fetch prompt '%s' from LangSmith on first attempt; "
                "using fallback template",
                self._prompt_name,
                # exc_info=True,
            )
            return self._fallback_template
