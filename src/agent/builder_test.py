"""Tests for src.agent.builder."""

from __future__ import annotations

from unittest.mock import MagicMock

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src.agent.builder import build_agent, render_system_prompt
from src.core.config import Settings


def test_render_system_prompt_includes_timezone_and_current_time() -> None:
    prompt = render_system_prompt(timezone="Asia/Jakarta")

    assert "Asia/Jakarta" in prompt
    assert "calendar" in prompt.lower()
    # Includes some date-like structure
    assert "20" in prompt  # current year prefix


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
