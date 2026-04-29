"""Tests for src.telegram.auth_filter."""

from __future__ import annotations

from unittest.mock import MagicMock

from src.telegram.auth_filter import AllowedUserFilter


def test_allowed_user_passes() -> None:
    filt = AllowedUserFilter(allowed_ids=[111, 222])
    update = MagicMock()
    update.effective_user.id = 111

    assert filt.filter(update) is True


def test_disallowed_user_rejected() -> None:
    filt = AllowedUserFilter(allowed_ids=[111, 222])
    update = MagicMock()
    update.effective_user.id = 999

    assert filt.filter(update) is False


def test_no_user_rejected() -> None:
    filt = AllowedUserFilter(allowed_ids=[111])
    update = MagicMock()
    update.effective_user = None

    assert filt.filter(update) is False
