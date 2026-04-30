"""Tests for src.agent.prompt_loader."""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

from src.agent.prompt_loader import (
    FALLBACK_ACCURACY_TEMPLATE,
    FALLBACK_SYSTEM_TEMPLATE,
    PromptLoader,
)


def _make_loader(
    *,
    ttl_seconds: int = 300,
    now: datetime | None = None,
    pull_prompt_result: str = "Hello {current_time} {timezone} {day_of_week}",
    pull_prompt_side_effect: Exception | None = None,
) -> PromptLoader:
    """Build a PromptLoader with a mock Client and optional frozen time."""
    mock_client = MagicMock()
    if pull_prompt_side_effect is not None:
        mock_client.pull_prompt.side_effect = pull_prompt_side_effect
    else:
        mock_prompt = MagicMock()
        mock_prompt.template = pull_prompt_result
        mock_client.pull_prompt.return_value = mock_prompt

    now_provider = (lambda: now) if now is not None else None
    return PromptLoader(
        prompt_name="switchboard-system",
        ttl_seconds=ttl_seconds,
        client=mock_client,
        now_provider=now_provider,
    )


async def test_returns_template_from_langsmith() -> None:
    loader = _make_loader(pull_prompt_result="LangSmith template")

    template = await loader.get_template()

    assert template == "LangSmith template"


async def test_caches_within_ttl() -> None:
    now = datetime(2026, 4, 30, 12, 0, 0)
    loader = _make_loader(ttl_seconds=300, now=now, pull_prompt_result="cached")

    await loader.get_template()
    # Advance time but stay within TTL
    loader._now_provider = lambda: now + timedelta(seconds=299)
    template = await loader.get_template()

    assert template == "cached"
    # pull_prompt should have been called only once
    loader._client.pull_prompt.assert_called_once()  # ty: ignore[unresolved-attribute]


async def test_refetches_after_ttl_expires() -> None:
    now = datetime(2026, 4, 30, 12, 0, 0)
    loader = _make_loader(ttl_seconds=300, now=now, pull_prompt_result="first")

    await loader.get_template()

    # Advance past TTL and change the mock return
    loader._now_provider = lambda: now + timedelta(seconds=301)
    mock_prompt2 = MagicMock()
    mock_prompt2.template = "second"
    loader._client.pull_prompt.return_value = mock_prompt2  # ty: ignore[unresolved-attribute]

    template = await loader.get_template()

    assert template == "second"
    assert loader._client.pull_prompt.call_count == 2  # ty: ignore[unresolved-attribute]


async def test_first_fetch_failure_returns_fallback() -> None:
    loader = _make_loader(pull_prompt_side_effect=ConnectionError("network down"))

    template = await loader.get_template()

    assert template == FALLBACK_SYSTEM_TEMPLATE


def test_registry_lookup_for_system_prompt() -> None:
    loader = PromptLoader(prompt_name="switchboard-system", client=MagicMock())

    assert loader._fallback_template == FALLBACK_SYSTEM_TEMPLATE


def test_registry_lookup_for_accuracy_evaluator() -> None:
    loader = PromptLoader(prompt_name="switchboard-accuracy-evaluator", client=MagicMock())

    assert loader._fallback_template == FALLBACK_ACCURACY_TEMPLATE


def test_unregistered_prompt_name_raises_keyerror() -> None:
    with pytest.raises(KeyError, match="unknown-prompt"):
        PromptLoader(prompt_name="unknown-prompt", client=MagicMock())


async def test_refresh_failure_keeps_cache() -> None:
    now = datetime(2026, 4, 30, 12, 0, 0)
    loader = _make_loader(ttl_seconds=300, now=now, pull_prompt_result="original")

    await loader.get_template()

    # Advance past TTL, then make pull_prompt fail
    loader._now_provider = lambda: now + timedelta(seconds=301)
    loader._client.pull_prompt.side_effect = ConnectionError("network down")  # ty: ignore[unresolved-attribute]

    template = await loader.get_template()

    assert template == "original"
