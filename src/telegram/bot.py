"""Telegram bot wiring — handlers + Application builder."""

from __future__ import annotations

import logging
from typing import Any

from langchain.messages import trim_messages
from langgraph.types import Command, Overwrite
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from src.core.config import Settings
from src.telegram.approval import (
    APPROVE_CALLBACK,
    EDIT_CALLBACK,
    EDIT_HINT,
    REJECT_CALLBACK,
    extract_create_event_args,
    render_approval_keyboard,
    render_approval_message,
)
from src.telegram.auth_filter import AllowedUserFilter
from telegram import Update

log = logging.getLogger(__name__)

_GENERIC_ERROR = "Sorry — I couldn't process that just now. Please try again in a moment."


async def _trim_conversation_if_needed(
    agent: Any, config: dict[str, Any], max_messages: int
) -> None:
    """Trim conversation history if it exceeds the maximum message count.

    Args:
        agent: The LangChain agent instance.
        config: The agent config containing thread_id.
        max_messages: Maximum number of messages to keep.
    """
    state = await agent.aget_state(config)
    messages = state.values.get("messages", [])

    if len(messages) > max_messages:
        trimmed = trim_messages(
            messages,
            max_tokens=max_messages,
            token_counter=len,  # Count by number of messages
            strategy="last",
            include_system=True,
        )
        await agent.aupdate_state(config, {"messages": Overwrite(trimmed)})


async def _has_pending_interrupt(agent: Any, config: dict[str, Any]) -> bool:
    """Return True if the thread has a paused interrupt awaiting resume."""
    state = await agent.aget_state(config)
    if state is None:
        return False
    if getattr(state, "tasks", None):
        return True
    values = getattr(state, "values", None) or {}
    return "__interrupt__" in values


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Forward an incoming text message to the agent and reply with its output.

    The agent is fetched from ``context.bot_data['agent']`` so it can be
    injected at startup and mocked in tests. ``thread_id`` is the Telegram
    ``chat_id`` so each chat has its own conversation memory.
    """
    if update.message is None or not update.message.text:
        return
    if update.effective_chat is None:
        return

    agent = context.bot_data["agent"]
    settings: Settings = context.bot_data["settings"]
    chat_id = update.effective_chat.id
    text = update.message.text

    config = {
        "configurable": {"thread_id": str(chat_id)},
        "recursion_limit": 10,
    }

    try:
        if await _has_pending_interrupt(agent, config):
            agent_input: Any = Command(resume={"decisions": [{"type": "reject", "feedback": text}]})
        else:
            await _trim_conversation_if_needed(agent, config, settings.max_conversation_messages)
            agent_input = {"messages": [{"role": "user", "content": text}]}

        result = await agent.ainvoke(agent_input, config=config)
    except Exception:
        log.exception("agent invocation failed for chat_id=%s", chat_id)
        await update.message.reply_text(_GENERIC_ERROR)
        return

    await _send_result(update.message, result)


async def _send_result(message: Any, result: dict[str, Any]) -> None:
    """Post either the approval prompt or the agent's text reply."""
    interrupt_value = result.get("__interrupt__")
    if interrupt_value:
        args = extract_create_event_args(interrupt_value)
        if args is not None:
            await message.reply_text(
                render_approval_message(args),
                reply_markup=render_approval_keyboard(),
            )
            return

    await message.reply_text(_extract_reply(result))


async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle Approve / Edit / Reject button taps from the approval prompt."""
    query = update.callback_query
    if query is None or query.message is None or update.effective_chat is None:
        return
    await query.answer()
    message = query.message

    data = query.data
    if data == EDIT_CALLBACK:
        await message.reply_text(EDIT_HINT)  # ty: ignore[unresolved-attribute]
        return

    if data == APPROVE_CALLBACK:
        resume_value: dict[str, Any] = {"decisions": [{"type": "approve"}]}
    elif data == REJECT_CALLBACK:
        resume_value = {"decisions": [{"type": "reject", "feedback": "User cancelled the draft."}]}
    else:
        log.warning("unknown callback_data: %s", data)
        return

    agent = context.bot_data["agent"]
    chat_id = update.effective_chat.id
    config = {
        "configurable": {"thread_id": str(chat_id)},
        "recursion_limit": 10,
    }

    try:
        result = await agent.ainvoke(Command(resume=resume_value), config=config)
    except Exception:
        log.exception("agent resume failed for chat_id=%s", chat_id)
        await message.reply_text(_GENERIC_ERROR)  # ty: ignore[unresolved-attribute]
        return

    await _send_result(message, result)


def _extract_reply(result: dict[str, Any]) -> str:
    messages = result.get("messages") or []
    if not messages:
        return _GENERIC_ERROR
    last = messages[-1]
    content = getattr(last, "content", None)
    if isinstance(content, str) and content:
        return content
    return _GENERIC_ERROR


def build_application(*, settings: Settings, agent: Any) -> Application:
    """Build the python-telegram-bot Application and register handlers.

    Args:
        settings: Application settings (provides bot token, allowed IDs).
        agent: Compiled LangChain agent instance.

    Returns:
        Configured Application ready to ``run_polling()``.
    """
    application = (
        Application.builder().token(settings.telegram_bot_token.get_secret_value()).build()
    )

    application.bot_data["agent"] = agent
    application.bot_data["settings"] = settings

    allowed = AllowedUserFilter(settings.allowed_telegram_user_ids)
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND & allowed, handle_message)
    )
    application.add_handler(CallbackQueryHandler(handle_callback))

    return application
