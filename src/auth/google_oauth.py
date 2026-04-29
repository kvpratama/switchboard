"""Google OAuth credential loading and refresh.

This module never starts a new OAuth flow — that is the bootstrap script's job.
At runtime we only load existing credentials and refresh them when expired.
"""

from __future__ import annotations

from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

READONLY_SCOPES: list[str] = ["https://www.googleapis.com/auth/calendar.readonly"]


class GoogleAuthError(RuntimeError):
    """Raised when credentials cannot be loaded or refreshed."""


def load_credentials(token_path: Path) -> Credentials:
    """Load Google OAuth credentials from ``token_path``, refreshing if expired.

    Args:
        token_path: Path to the JSON file written by the bootstrap script.

    Returns:
        A valid ``Credentials`` instance.

    Raises:
        GoogleAuthError: If the token file is missing or cannot be refreshed.
    """
    if not token_path.exists():
        raise GoogleAuthError(
            f"Google token file not found at {token_path}. "
            "Run `uv run python -m src.auth.bootstrap` first."
        )

    try:
        creds = Credentials.from_authorized_user_file(str(token_path), READONLY_SCOPES)
    except Exception as e:
        raise GoogleAuthError(
            f"Failed to load Google credentials from {token_path}: {e}. "
            "Run `uv run python -m src.auth.bootstrap` to re-authorize."
        ) from e

    if creds.valid:
        return creds

    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        token_path.write_text(creds.to_json())
        return creds

    raise GoogleAuthError(
        "Google credentials are invalid and cannot be refreshed; re-run bootstrap to re-authorize."
    )
