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
    state_mock.tasks = ()  # No pending interrupt
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
    state_mock.tasks = ()
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
    state_mock.tasks = ()
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
    state_mock.tasks = ()
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


async def test_handle_message_posts_approval_prompt_when_agent_interrupts() -> None:
    agent = MagicMock()

    state_mock = MagicMock()
    state_mock.values = {"messages": []}
    state_mock.tasks = ()  # No pending interrupt before this turn
    agent.aget_state = AsyncMock(return_value=state_mock)

    interrupt_payload = [
        {
            "value": [
                {
                    "action_request": {
                        "action": "create_event",
                        "args": {
                            "summary": "Lunch",
                            "start": "2026-05-03T13:00:00+09:00",
                            "end": "2026-05-03T14:00:00+09:00",
                        },
                    }
                }
            ]
        }
    ]
    agent.ainvoke = AsyncMock(return_value={"messages": [], "__interrupt__": interrupt_payload})

    update = MagicMock()
    update.effective_chat.id = 12345
    update.message.text = "schedule lunch tomorrow 1pm"
    update.message.reply_text = AsyncMock()

    settings_mock = MagicMock()
    settings_mock.max_conversation_messages = 50

    context = MagicMock()
    context.bot_data = {"agent": agent, "settings": settings_mock}

    await handle_message(update, context)

    update.message.reply_text.assert_awaited_once()
    call_args = update.message.reply_text.call_args
    text = call_args.args[0]
    assert "Create this event?" in text
    assert "Lunch" in text
    assert call_args.kwargs.get("reply_markup") is not None


async def test_handle_message_resumes_with_reject_feedback_when_pending_interrupt() -> None:
    from langgraph.types import Command

    agent = MagicMock()

    state_mock = MagicMock()
    state_mock.values = {"messages": [], "__interrupt__": [{"value": "..."}]}
    state_mock.tasks = (MagicMock(),)
    agent.aget_state = AsyncMock(return_value=state_mock)

    agent.ainvoke = AsyncMock(
        return_value={
            "messages": [],
            "__interrupt__": [
                {
                    "value": [
                        {
                            "action_request": {
                                "action": "create_event",
                                "args": {
                                    "summary": "Lunch",
                                    "start": "2026-05-03T14:00:00+09:00",
                                    "end": "2026-05-03T15:00:00+09:00",
                                },
                            }
                        }
                    ]
                }
            ],
        }
    )

    update = MagicMock()
    update.effective_chat.id = 12345
    update.message.text = "make it 2pm"
    update.message.reply_text = AsyncMock()

    settings_mock = MagicMock()
    settings_mock.max_conversation_messages = 50

    context = MagicMock()
    context.bot_data = {"agent": agent, "settings": settings_mock}

    await handle_message(update, context)

    agent.ainvoke.assert_awaited_once()
    sent_input = agent.ainvoke.call_args.args[0]
    assert isinstance(sent_input, Command)
    decisions = sent_input.resume["decisions"]
    assert decisions[0]["type"] == "reject"
    assert decisions[0]["feedback"] == "make it 2pm"

    update.message.reply_text.assert_awaited_once()
    text = update.message.reply_text.call_args.args[0]
    assert "Create this event?" in text


async def test_handle_callback_approve_resumes_agent_with_approve_decision() -> None:
    from langgraph.types import Command

    from src.telegram.approval import APPROVE_CALLBACK
    from src.telegram.bot import handle_callback

    agent = MagicMock()
    agent.ainvoke = AsyncMock(return_value={"messages": [AIMessage(content="Created.")]})

    update = MagicMock()
    update.effective_chat.id = 12345
    update.effective_user.id = 11111
    query = MagicMock()
    query.data = APPROVE_CALLBACK
    query.answer = AsyncMock()
    query.message.reply_text = AsyncMock()
    query.message.edit_reply_markup = AsyncMock()
    update.callback_query = query

    settings_mock = MagicMock()
    settings_mock.allowed_telegram_user_ids = [11111]
    context = MagicMock()
    context.bot_data = {"agent": agent, "settings": settings_mock}

    await handle_callback(update, context)

    query.answer.assert_awaited_once()
    agent.ainvoke.assert_awaited_once()
    sent_input = agent.ainvoke.call_args.args[0]
    assert isinstance(sent_input, Command)
    assert sent_input.resume == {"decisions": [{"type": "approve"}]}
    query.message.reply_text.assert_awaited_once_with("Created.")


async def test_handle_callback_reject_resumes_with_reject_and_cancellation() -> None:
    from langgraph.types import Command

    from src.telegram.approval import REJECT_CALLBACK
    from src.telegram.bot import handle_callback

    agent = MagicMock()
    agent.ainvoke = AsyncMock(return_value={"messages": [AIMessage(content="OK, cancelled.")]})

    update = MagicMock()
    update.effective_chat.id = 12345
    update.effective_user.id = 11111
    query = MagicMock()
    query.data = REJECT_CALLBACK
    query.answer = AsyncMock()
    query.message.reply_text = AsyncMock()
    query.message.edit_reply_markup = AsyncMock()
    update.callback_query = query

    settings_mock = MagicMock()
    settings_mock.allowed_telegram_user_ids = [11111]
    context = MagicMock()
    context.bot_data = {"agent": agent, "settings": settings_mock}

    await handle_callback(update, context)

    sent_input = agent.ainvoke.call_args.args[0]
    assert isinstance(sent_input, Command)
    decisions = sent_input.resume["decisions"]
    assert decisions[0]["type"] == "reject"
    assert "feedback" not in decisions[0]
    query.message.reply_text.assert_awaited_once_with("OK, cancelled.")


async def test_handle_callback_edit_posts_hint_without_resuming() -> None:
    from src.telegram.approval import EDIT_CALLBACK, EDIT_HINT
    from src.telegram.bot import handle_callback

    agent = MagicMock()
    agent.ainvoke = AsyncMock()

    update = MagicMock()
    update.effective_chat.id = 12345
    update.effective_user.id = 11111
    query = MagicMock()
    query.data = EDIT_CALLBACK
    query.answer = AsyncMock()
    query.message.reply_text = AsyncMock()
    query.message.edit_reply_markup = AsyncMock()
    update.callback_query = query

    settings_mock = MagicMock()
    settings_mock.allowed_telegram_user_ids = [11111]
    context = MagicMock()
    context.bot_data = {"agent": agent, "settings": settings_mock}

    await handle_callback(update, context)

    query.answer.assert_awaited_once()
    query.message.reply_text.assert_awaited_once_with(EDIT_HINT)
    agent.ainvoke.assert_not_awaited()


def test_build_application_registers_message_and_callback_handlers(settings_env, mocker) -> None:
    from telegram.ext import CallbackQueryHandler, MessageHandler

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
    app_any: Any = app
    handlers = [call.args[0] for call in app_any.add_handler.call_args_list]
    assert any(isinstance(h, MessageHandler) for h in handlers)
    assert any(isinstance(h, CallbackQueryHandler) for h in handlers)


async def test_handle_callback_rejects_unauthorized_user() -> None:
    from src.telegram.approval import APPROVE_CALLBACK
    from src.telegram.bot import handle_callback

    agent = MagicMock()
    agent.ainvoke = AsyncMock()

    update = MagicMock()
    update.effective_chat.id = 12345
    update.effective_user.id = 99999  # Unauthorized user
    query = MagicMock()
    query.data = APPROVE_CALLBACK
    query.answer = AsyncMock()
    query.message.reply_text = AsyncMock()
    update.callback_query = query

    settings_mock = MagicMock()
    settings_mock.allowed_telegram_user_ids = [11111, 22222]  # Authorized users

    context = MagicMock()
    context.bot_data = {"agent": agent, "settings": settings_mock}

    await handle_callback(update, context)

    query.answer.assert_awaited_once()
    agent.ainvoke.assert_not_awaited()
    query.message.reply_text.assert_not_awaited()
