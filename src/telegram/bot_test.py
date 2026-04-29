"""Tests for src.telegram.bot."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from langchain.messages import AIMessage, HumanMessage, SystemMessage

from src.telegram.bot import handle_message


async def test_handle_message_trims_conversation_when_exceeds_max() -> None:
    """Test that conversations are trimmed when they exceed MAX_MESSAGES."""
    agent = MagicMock()

    # Simulate a long conversation history (60 messages)
    from langchain_core.messages import BaseMessage

    old_messages: list[BaseMessage] = [SystemMessage(content="System prompt")]
    old_messages.extend([HumanMessage(content=f"msg {i}") for i in range(59)])

    state_mock = MagicMock()
    state_mock.values = {"messages": old_messages}
    agent.aget_state = AsyncMock(return_value=state_mock)
    agent.aupdate_state = AsyncMock()
    agent.ainvoke = AsyncMock(return_value={"messages": [AIMessage(content="Response")]})

    update = MagicMock()
    update.effective_chat.id = 12345
    update.message.text = "new message"
    update.message.reply_text = AsyncMock()

    settings_mock = MagicMock()
    settings_mock.max_conversation_messages = 50

    context = MagicMock()
    context.bot_data = {"agent": agent, "settings": settings_mock}

    await handle_message(update, context)

    # Should have called update_state to trim
    agent.aupdate_state.assert_awaited_once()
    update_call = agent.aupdate_state.call_args

    # Check that messages were trimmed
    from langgraph.types import Overwrite

    trimmed_messages = update_call.args[1]["messages"]
    assert isinstance(trimmed_messages, Overwrite)
    assert len(trimmed_messages.value) == 50  # MAX_MESSAGES
    assert trimmed_messages.value[0].content == "System prompt"  # System message kept


async def test_handle_message_skips_trim_when_under_max() -> None:
    """Test that trimming is skipped when message count is under MAX_MESSAGES."""
    agent = MagicMock()

    # Simulate a short conversation (10 messages)
    short_messages = [HumanMessage(content=f"msg {i}") for i in range(10)]

    state_mock = MagicMock()
    state_mock.values = {"messages": short_messages}
    agent.aget_state = AsyncMock(return_value=state_mock)
    agent.aupdate_state = AsyncMock()
    agent.ainvoke = AsyncMock(return_value={"messages": [AIMessage(content="Response")]})

    update = MagicMock()
    update.effective_chat.id = 12345
    update.message.text = "new message"
    update.message.reply_text = AsyncMock()

    settings_mock = MagicMock()
    settings_mock.max_conversation_messages = 50

    context = MagicMock()
    context.bot_data = {"agent": agent, "settings": settings_mock}

    await handle_message(update, context)

    # Should NOT have called update_state
    agent.aupdate_state.assert_not_awaited()


async def test_handle_message_sends_agent_reply_to_chat() -> None:
    agent = MagicMock()

    # Mock state for trimming check
    state_mock = MagicMock()
    state_mock.values = {"messages": []}  # Empty, no trimming needed
    agent.aget_state = AsyncMock(return_value=state_mock)

    agent.ainvoke = AsyncMock(
        return_value={"messages": [AIMessage(content="You have 3 events today.")]}
    )

    update = MagicMock()
    update.effective_chat.id = 12345
    update.message.text = "what's on today?"
    update.message.reply_text = AsyncMock()

    settings_mock = MagicMock()
    settings_mock.max_conversation_messages = 50

    context = MagicMock()
    context.bot_data = {"agent": agent, "settings": settings_mock}

    await handle_message(update, context)

    agent.ainvoke.assert_awaited_once()
    call_kwargs = agent.ainvoke.call_args.kwargs
    inputs = agent.ainvoke.call_args.args[0]
    assert inputs["messages"][0]["role"] == "user"
    assert inputs["messages"][0]["content"] == "what's on today?"
    assert call_kwargs["config"]["configurable"]["thread_id"] == "12345"
    assert call_kwargs["config"]["recursion_limit"] == 10
    update.message.reply_text.assert_awaited_once_with("You have 3 events today.")


async def test_handle_message_apologizes_on_agent_exception() -> None:
    agent = MagicMock()

    # Mock state for trimming check
    state_mock = MagicMock()
    state_mock.values = {"messages": []}
    agent.aget_state = AsyncMock(return_value=state_mock)

    agent.ainvoke = AsyncMock(side_effect=RuntimeError("LLM down"))

    update = MagicMock()
    update.effective_chat.id = 12345
    update.message.text = "hi"
    update.message.reply_text = AsyncMock()

    settings_mock = MagicMock()
    settings_mock.max_conversation_messages = 50

    context = MagicMock()
    context.bot_data = {"agent": agent, "settings": settings_mock}

    await handle_message(update, context)

    update.message.reply_text.assert_awaited_once()
    sent = update.message.reply_text.call_args.args[0]
    assert "sorry" in sent.lower() or "couldn" in sent.lower()


async def test_handle_message_skips_when_no_text() -> None:
    agent = MagicMock()
    agent.ainvoke = AsyncMock()

    update = MagicMock()
    update.message = None  # e.g. edited message without text

    context = MagicMock()
    context.bot_data = {"agent": agent}

    await handle_message(update, context)

    agent.ainvoke.assert_not_awaited()


def test_build_application_registers_filtered_handler(settings_env, mocker) -> None:
    app_mock = MagicMock(name="application")
    app_mock.bot_data = {}
    builder_mock = MagicMock()
    builder_mock.token.return_value = builder_mock
    builder_mock.build.return_value = app_mock
    mocker.patch("src.telegram.bot.Application.builder", return_value=builder_mock)

    from typing import Any

    from src.core.config import Settings
    from src.telegram.bot import build_application

    app = build_application(settings=Settings(), agent=MagicMock())

    builder_mock.token.assert_called_once()
    assert app.bot_data["agent"] is not None
    # Cast to Any to allow mock assertion
    app_any: Any = app
    app_any.add_handler.assert_called_once()
