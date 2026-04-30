"""Tests for src.agent.builder."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src.agent.builder import build_agent, render_system_prompt
from src.agent.prompt_loader import FALLBACK_SYSTEM_TEMPLATE
from src.core.config import Settings


def test_render_system_prompt_includes_timezone_and_current_time() -> None:
    prompt = render_system_prompt(template=FALLBACK_SYSTEM_TEMPLATE, timezone="Asia/Tokyo")

    assert "Asia/Tokyo" in prompt
    assert "calendar" in prompt.lower()
    # Includes some date-like structure
    assert "20" in prompt  # current year prefix


def test_render_system_prompt_uses_now_provider_for_reproducibility() -> None:
    fixed = datetime(2026, 4, 29, 17, 0, 0, tzinfo=ZoneInfo("Asia/Tokyo"))

    prompt = render_system_prompt(
        template=FALLBACK_SYSTEM_TEMPLATE, timezone="Asia/Tokyo", now_provider=lambda: fixed
    )

    assert "2026-04-29T17:00:00+09:00" in prompt
    assert "Wednesday" in prompt


def test_render_system_prompt_now_provider_naive_datetime_uses_timezone() -> None:
    naive = datetime(2026, 4, 29, 17, 0, 0)

    prompt = render_system_prompt(
        template=FALLBACK_SYSTEM_TEMPLATE, timezone="Asia/Tokyo", now_provider=lambda: naive
    )

    assert "2026-04-29T17:00:00+09:00" in prompt
    assert "Wednesday" in prompt


def test_render_system_prompt_uses_custom_template() -> None:
    fixed = datetime(2026, 4, 29, 17, 0, 0, tzinfo=ZoneInfo("Asia/Tokyo"))
    custom = "Time: {current_time}, TZ: {timezone}, Day: {day_of_week}"

    prompt = render_system_prompt(
        template=custom, timezone="Asia/Tokyo", now_provider=lambda: fixed
    )

    assert prompt == "Time: 2026-04-29T17:00:00+09:00, TZ: Asia/Tokyo, Day: Wednesday"


def test_fallback_system_template_has_required_placeholders() -> None:
    assert "{current_time}" in FALLBACK_SYSTEM_TEMPLATE
    assert "{timezone}" in FALLBACK_SYSTEM_TEMPLATE
    assert "{day_of_week}" in FALLBACK_SYSTEM_TEMPLATE


async def test_build_agent_wires_model_and_tools(settings_env, mocker) -> None:
    fake_model = FakeListChatModel(responses=["ok"])
    init = mocker.patch("src.agent.builder.init_chat_model", return_value=fake_model)
    fake_client = MagicMock()
    mocker.patch("src.agent.builder.GoogleCalendarClient", return_value=fake_client)
    fake_tools = [MagicMock(name="t1"), MagicMock(name="t2"), MagicMock(name="t3")]
    mocker.patch("src.agent.builder.build_calendar_tools", return_value=fake_tools)

    captured: dict[str, object] = {}

    def fake_create_agent(**kwargs: object) -> object:
        captured.update(kwargs)
        return MagicMock(name="agent")

    mocker.patch("src.agent.builder.create_agent", side_effect=fake_create_agent)
    mock_loader = MagicMock()
    mock_loader.get_template = MagicMock(return_value="test template")
    mocker.patch("src.agent.builder.PromptLoader", return_value=mock_loader)

    settings = Settings()
    checkpointer = MagicMock(name="checkpointer")
    agent = await build_agent(settings=settings, checkpointer=checkpointer)

    assert agent is not None
    assert captured["model"] is fake_model
    tools = captured["tools"]
    assert isinstance(tools, list)
    assert tools == fake_tools
    assert captured["checkpointer"] is checkpointer
    init.assert_called_once()
    init_kwargs = init.call_args.kwargs
    assert init_kwargs["model"] == settings.llm_model
    assert init_kwargs["model_provider"] == settings.llm_provider
    assert init_kwargs["temperature"] == 0
    # Verify PromptLoader was constructed with the right args
    from src.agent.builder import PromptLoader

    PromptLoader.assert_called_once_with(  # ty: ignore[unresolved-attribute]
        prompt_name="switchboard-system",
        ttl_seconds=300,
        now_provider=None,
    )
