"""Telegram bot wiring — handlers + Application builder."""

from __future__ import annotations

import logging
from typing import Any

from telegram.ext import (
    Application,
    ContextTypes,
    MessageHandler,
    filters,
)

from src.core.config import Settings
from src.telegram.auth_filter import AllowedUserFilter
from telegram import Update

log = logging.getLogger(__name__)

_GENERIC_ERROR = "Sorry — I couldn't process that just now. Please try again in a moment."


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
    chat_id = update.effective_chat.id
    text = update.message.text

    try:
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": text}]},
            config={
                "configurable": {"thread_id": str(chat_id)},
                "recursion_limit": 10,
            },
        )
    except Exception:
        log.exception("agent invocation failed for chat_id=%s", chat_id)
        await update.message.reply_text(_GENERIC_ERROR)
        return

    reply = _extract_reply(result)
    await update.message.reply_text(reply)


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

    allowed = AllowedUserFilter(settings.allowed_telegram_user_ids)
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND & allowed, handle_message)
    )

    return application
