"""Tests for src.auth.google_oauth."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.auth.google_oauth import (
    EVENTS_SCOPES,
    GoogleAuthError,
    load_credentials,
)


def test_events_scopes_grant_read_and_write() -> None:
    assert EVENTS_SCOPES == [
        "https://www.googleapis.com/auth/calendar.events",
        "https://www.googleapis.com/auth/calendar.settings.readonly",
    ]


def test_load_credentials_reads_token_file(tmp_path: Path, mocker) -> None:
    token_path = tmp_path / "token.json"
    token_path.write_text("{}")  # Existence is what matters; mock parses contents.

    fake_creds = MagicMock(valid=True, expired=False)
    from_file = mocker.patch(
        "src.auth.google_oauth.Credentials.from_authorized_user_file",
        return_value=fake_creds,
    )

    creds = load_credentials(token_path)

    assert creds is fake_creds
    from_file.assert_called_once_with(str(token_path), EVENTS_SCOPES)


def test_load_credentials_refreshes_when_expired(tmp_path: Path, mocker) -> None:
    token_path = tmp_path / "token.json"
    token_path.write_text("{}")

    fake_creds = MagicMock(valid=False, expired=True, refresh_token="refresh-token")
    fake_creds.to_json.return_value = '{"refreshed": true}'
    mocker.patch(
        "src.auth.google_oauth.Credentials.from_authorized_user_file",
        return_value=fake_creds,
    )

    creds = load_credentials(token_path)

    fake_creds.refresh.assert_called_once()
    assert creds is fake_creds
    assert token_path.read_text() == '{"refreshed": true}'


def test_load_credentials_missing_file_raises(tmp_path: Path) -> None:
    missing = tmp_path / "nope.json"

    with pytest.raises(GoogleAuthError, match="not found"):
        load_credentials(missing)


def test_load_credentials_no_refresh_token_raises(tmp_path: Path, mocker) -> None:
    token_path = tmp_path / "token.json"
    token_path.write_text("{}")

    fake_creds = MagicMock(valid=False, expired=True, refresh_token=None)
    mocker.patch(
        "src.auth.google_oauth.Credentials.from_authorized_user_file",
        return_value=fake_creds,
    )

    with pytest.raises(GoogleAuthError, match="re-run bootstrap"):
        load_credentials(token_path)


def test_bootstrap_module_imports() -> None:
    import src.auth.bootstrap  # noqa: F401
