"""Approval-message rendering for the create_event HITL interrupt."""

from __future__ import annotations

from typing import Any

from telegram import InlineKeyboardButton, InlineKeyboardMarkup

APPROVE_CALLBACK = "approve"
EDIT_CALLBACK = "edit"
REJECT_CALLBACK = "reject"

EDIT_HINT = "What would you like to change? Reply with the correction (e.g. 'make it 2pm')."


def render_approval_message(args: dict[str, Any]) -> str:
    """Render the human-readable approval prompt for a draft event.

    Args:
        args: The ``create_event`` tool call arguments captured by the
            HITL interrupt. Keys: ``summary``, ``start``, ``end``,
            optional ``description`` and ``location``.

    Returns:
        Plain-text message body suitable for ``reply_text``.
    """
    lines = [
        "📅 Create this event?",
        "",
        f"Title:    {args.get('summary', '—')}",
        f"Start:    {args.get('start', '—')}",
        f"End:      {args.get('end', '—')}",
    ]
    location = args.get("location")
    if location:
        lines.append(f"Location: {location}")
    description = args.get("description")
    if description:
        lines.append(f"Notes:    {description}")
    return "\n".join(lines)


def render_approval_keyboard() -> InlineKeyboardMarkup:
    """Return the inline keyboard with Approve / Edit / Reject buttons."""
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("✅ Approve", callback_data=APPROVE_CALLBACK),
                InlineKeyboardButton("✏️ Edit", callback_data=EDIT_CALLBACK),
                InlineKeyboardButton("❌ Reject", callback_data=REJECT_CALLBACK),
            ]
        ]
    )


def extract_create_event_args(interrupt_value: Any) -> dict[str, Any] | None:
    """Walk an interrupt payload and return args for the first create_event call.

    HITL middleware nests the action under either ``action_request`` or
    directly as ``action`` in different versions. This helper is tolerant
    of both shapes.

    Args:
        interrupt_value: The value from ``result["__interrupt__"]``.

    Returns:
        The ``args`` dict for the first ``create_event`` action found, or
        ``None`` if no such action is present.
    """
    if not interrupt_value:
        return None

    def _walk(node: Any) -> dict[str, Any] | None:
        if isinstance(node, dict):
            action = node.get("action")
            if action == "create_event" and isinstance(node.get("args"), dict):
                return node["args"]
            request = node.get("action_request")
            if isinstance(request, dict):
                found = _walk(request)
                if found is not None:
                    return found
            for value in node.values():
                found = _walk(value)
                if found is not None:
                    return found
        elif isinstance(node, list):
            for item in node:
                found = _walk(item)
                if found is not None:
                    return found
            return None
        elif hasattr(node, "value"):
            return _walk(node.value)
        return None

    return _walk(interrupt_value)
