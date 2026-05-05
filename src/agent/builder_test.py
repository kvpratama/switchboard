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


async def test_build_agent_skips_hitl_when_disabled(settings_env, mocker) -> None:
    """When enable_hitl=False, no HumanInTheLoopMiddleware is passed."""
    fake_model = FakeListChatModel(responses=["ok"])
    mocker.patch("src.agent.builder.init_chat_model", return_value=fake_model)
    fake_client = MagicMock()
    mocker.patch("src.agent.builder.GoogleCalendarClient", return_value=fake_client)
    fake_tools = [MagicMock(name="t1")]
    mocker.patch("src.agent.builder.build_calendar_tools", return_value=fake_tools)

    captured: dict[str, object] = {}

    def fake_create_agent(**kwargs: object) -> object:
        captured.update(kwargs)
        return MagicMock(name="agent")

    mocker.patch("src.agent.builder.create_agent", side_effect=fake_create_agent)
    mock_loader = MagicMock()
    mock_loader.get_template = MagicMock(return_value="test template")
    mocker.patch("src.agent.builder.PromptLoader", return_value=mock_loader)

    await build_agent(settings=Settings(), checkpointer=None, enable_hitl=False)

    # Ensure create_agent actually ran before asserting on its kwargs;
    # otherwise an empty `captured` would silently pass the assertion below.
    assert "middleware" in captured, "create_agent was not invoked by build_agent"
    middleware = captured["middleware"]
    middleware_types = [type(m).__name__ for m in middleware]  # ty: ignore[not-iterable]
    assert "HumanInTheLoopMiddleware" not in middleware_types


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
    )


async def test_agent_interrupts_before_create_event(settings_env, mocker) -> None:
    """The agent must pause via HITL middleware before create_event runs."""
    from langchain_core.messages import AIMessage
    from langgraph.checkpoint.memory import MemorySaver

    from src.agent.builder import build_agent

    fake_client = MagicMock()
    fake_client.create_event = mocker.AsyncMock(
        side_effect=AssertionError("create_event must not be called before approval")
    )
    fake_client.get_timezone = mocker.AsyncMock(return_value="Asia/Tokyo")

    tool_call = {
        "name": "create_event",
        "args": {
            "summary": "Lunch",
            "start": "2026-05-03T13:00:00+09:00",
            "end": "2026-05-03T14:00:00+09:00",
        },
        "id": "call-1",
        "type": "tool_call",
    }
    ai_msg = AIMessage(content="", tool_calls=[tool_call])

    fake_model = MagicMock()
    fake_model.ainvoke = mocker.AsyncMock(return_value=ai_msg)
    fake_model.bind_tools = MagicMock(return_value=fake_model)
    mocker.patch("src.agent.builder.init_chat_model", return_value=fake_model)

    checkpointer = MemorySaver()
    agent = await build_agent(
        settings=Settings(),
        checkpointer=checkpointer,
        calendar_client=fake_client,
        timezone="Asia/Tokyo",
    )

    config = {"configurable": {"thread_id": "test-1"}}
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "schedule lunch tomorrow 1pm"}]},
        config=config,
    )

    assert "__interrupt__" in result
    fake_client.create_event.assert_not_awaited()


async def test_agent_resumes_and_calls_create_event_on_approve(settings_env, mocker) -> None:
    from langchain_core.messages import AIMessage
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.types import Command

    from src.agent.builder import build_agent

    fake_client = MagicMock()
    fake_client.create_event = mocker.AsyncMock(return_value={"id": "evt-new"})
    fake_client.get_timezone = mocker.AsyncMock(return_value="Asia/Tokyo")

    first_call = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "create_event",
                "args": {
                    "summary": "Lunch",
                    "start": "2026-05-03T13:00:00+09:00",
                    "end": "2026-05-03T14:00:00+09:00",
                },
                "id": "call-1",
                "type": "tool_call",
            }
        ],
    )
    final_text = AIMessage(content="Created.")

    fake_model = MagicMock()
    fake_model.ainvoke = mocker.AsyncMock(side_effect=[first_call, final_text])
    fake_model.bind_tools = MagicMock(return_value=fake_model)
    mocker.patch("src.agent.builder.init_chat_model", return_value=fake_model)

    checkpointer = MemorySaver()
    agent = await build_agent(
        settings=Settings(),
        checkpointer=checkpointer,
        calendar_client=fake_client,
        timezone="Asia/Tokyo",
    )

    config = {"configurable": {"thread_id": "test-2"}}
    await agent.ainvoke(
        {"messages": [{"role": "user", "content": "schedule lunch"}]},
        config=config,
    )

    result = await agent.ainvoke(
        Command(resume={"decisions": [{"type": "approve"}]}),
        config=config,
    )

    fake_client.create_event.assert_awaited_once()
    assert "__interrupt__" not in result


async def test_agent_resumes_and_does_not_call_create_event_on_reject(settings_env, mocker) -> None:
    from langchain_core.messages import AIMessage
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.types import Command

    from src.agent.builder import build_agent

    fake_client = MagicMock()
    fake_client.create_event = mocker.AsyncMock(return_value={"id": "evt-new"})
    fake_client.get_timezone = mocker.AsyncMock(return_value="Asia/Tokyo")

    first_call = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "create_event",
                "args": {
                    "summary": "Lunch",
                    "start": "2026-05-03T13:00:00+09:00",
                    "end": "2026-05-03T14:00:00+09:00",
                },
                "id": "call-1",
                "type": "tool_call",
            }
        ],
    )
    final_text = AIMessage(content="Cancelled.")

    fake_model = MagicMock()
    fake_model.ainvoke = mocker.AsyncMock(side_effect=[first_call, final_text])
    fake_model.bind_tools = MagicMock(return_value=fake_model)
    mocker.patch("src.agent.builder.init_chat_model", return_value=fake_model)

    checkpointer = MemorySaver()
    agent = await build_agent(
        settings=Settings(),
        checkpointer=checkpointer,
        calendar_client=fake_client,
        timezone="Asia/Tokyo",
    )

    config = {"configurable": {"thread_id": "test-3"}}
    await agent.ainvoke(
        {"messages": [{"role": "user", "content": "schedule lunch"}]},
        config=config,
    )

    result = await agent.ainvoke(
        Command(resume={"decisions": [{"type": "reject"}]}),
        config=config,
    )

    fake_client.create_event.assert_not_awaited()
    assert "__interrupt__" not in result
