"""Tests for src.telegram.approval."""

from __future__ import annotations

from src.telegram.approval import (
    APPROVE_CALLBACK,
    EDIT_CALLBACK,
    REJECT_CALLBACK,
    extract_create_event_args,
    render_approval_keyboard,
    render_approval_message,
)


def test_render_approval_message_includes_required_fields() -> None:
    args = {
        "summary": "Lunch with Sarah",
        "start": "2026-05-03T13:00:00+09:00",
        "end": "2026-05-03T14:00:00+09:00",
    }

    text = render_approval_message(args)

    assert "Lunch with Sarah" in text
    assert "2026-05-03T13:00:00+09:00" in text
    assert "2026-05-03T14:00:00+09:00" in text
    assert "Create this event?" in text


def test_render_approval_message_includes_optional_fields_when_present() -> None:
    args = {
        "summary": "Lunch",
        "start": "2026-05-03T13:00:00+09:00",
        "end": "2026-05-03T14:00:00+09:00",
        "description": "with Sarah",
        "location": "Cafe",
    }

    text = render_approval_message(args)

    assert "with Sarah" in text
    assert "Cafe" in text


def test_render_approval_keyboard_has_three_buttons_with_callback_data() -> None:
    keyboard = render_approval_keyboard()

    rows = keyboard.inline_keyboard
    assert len(rows) == 1
    buttons = rows[0]
    assert len(buttons) == 3
    callback_data = [b.callback_data for b in buttons]
    assert callback_data == [APPROVE_CALLBACK, EDIT_CALLBACK, REJECT_CALLBACK]


def test_extract_create_event_args_finds_first_create_event_call() -> None:
    interrupt_value = [
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

    args = extract_create_event_args(interrupt_value)

    assert args == {
        "summary": "Lunch",
        "start": "2026-05-03T13:00:00+09:00",
        "end": "2026-05-03T14:00:00+09:00",
    }


def test_extract_create_event_args_returns_none_when_absent() -> None:
    assert extract_create_event_args([]) is None
    assert extract_create_event_args(None) is None
