"""Tests for src.telegram.bot."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import AIMessage

from src.telegram.bot import handle_message


async def test_handle_message_sends_agent_reply_to_chat() -> None:
    agent = MagicMock()
    agent.ainvoke = AsyncMock(
        return_value={"messages": [AIMessage(content="You have 3 events today.")]}
    )

    update = MagicMock()
    update.effective_chat.id = 12345
    update.message.text = "what's on today?"
    update.message.reply_text = AsyncMock()

    context = MagicMock()
    context.bot_data = {"agent": agent}

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
    agent.ainvoke = AsyncMock(side_effect=RuntimeError("LLM down"))

    update = MagicMock()
    update.effective_chat.id = 12345
    update.message.text = "hi"
    update.message.reply_text = AsyncMock()

    context = MagicMock()
    context.bot_data = {"agent": agent}

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
