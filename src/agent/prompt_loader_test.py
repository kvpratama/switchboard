"""Tests for src.agent.prompt_loader."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.agent.prompt_loader import (
    FALLBACK_ACCURACY_TEMPLATE,
    FALLBACK_SYSTEM_TEMPLATE,
    PromptLoader,
)


def _make_loader(
    *,
    pull_prompt_result: str = "Hello {current_time} {timezone} {day_of_week}",
    pull_prompt_side_effect: Exception | None = None,
) -> PromptLoader:
    """Build a PromptLoader with a mock LangSmith Client."""
    mock_client = MagicMock()
    if pull_prompt_side_effect is not None:
        mock_client.pull_prompt.side_effect = pull_prompt_side_effect
    else:
        mock_prompt = MagicMock()
        mock_prompt.template = pull_prompt_result
        mock_client.pull_prompt.return_value = mock_prompt

    return PromptLoader(
        prompt_name="switchboard-system",
        client=mock_client,
    )


async def test_returns_template_from_langsmith() -> None:
    loader = _make_loader(pull_prompt_result="LangSmith template")

    template = await loader.get_template()

    assert template == "LangSmith template"


async def test_first_fetch_failure_returns_fallback() -> None:
    loader = _make_loader(pull_prompt_side_effect=ConnectionError("network down"))

    template = await loader.get_template()

    assert template == FALLBACK_SYSTEM_TEMPLATE


async def test_refresh_failure_after_success_serves_last_template() -> None:
    loader = _make_loader(pull_prompt_result="original")

    first = await loader.get_template()
    assert first == "original"

    # Now simulate LangSmith going down on the next refresh.
    loader._client.pull_prompt.side_effect = ConnectionError("network down")  # ty: ignore[unresolved-attribute]
    loader._client.pull_prompt.return_value = None  # ty: ignore[unresolved-attribute]

    second = await loader.get_template()

    assert second == "original"


def test_registry_lookup_for_system_prompt() -> None:
    loader = PromptLoader(prompt_name="switchboard-system", client=MagicMock())

    assert loader._fallback_template == FALLBACK_SYSTEM_TEMPLATE


def test_registry_lookup_for_accuracy_evaluator() -> None:
    loader = PromptLoader(prompt_name="switchboard-accuracy-evaluator", client=MagicMock())

    assert loader._fallback_template == FALLBACK_ACCURACY_TEMPLATE


def test_unregistered_prompt_name_raises_keyerror() -> None:
    with pytest.raises(KeyError, match="unknown-prompt"):
        PromptLoader(prompt_name="unknown-prompt", client=MagicMock())
