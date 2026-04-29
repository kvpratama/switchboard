"""python-telegram-bot filter that whitelists numeric user IDs."""

from __future__ import annotations

from collections.abc import Iterable

from telegram.ext.filters import UpdateFilter

from telegram import Update


class AllowedUserFilter(UpdateFilter):
    """Pass updates whose ``effective_user.id`` is in the allowed set."""

    def __init__(self, allowed_ids: Iterable[int]) -> None:
        """Store the allowed-id set.

        Args:
            allowed_ids: Iterable of numeric Telegram user IDs.
        """
        super().__init__()
        self._allowed = set(allowed_ids)

    def filter(self, update: Update) -> bool:
        """Return True iff the update's effective user is allowed."""
        user = update.effective_user
        if user is None:
            return False
        return user.id in self._allowed
