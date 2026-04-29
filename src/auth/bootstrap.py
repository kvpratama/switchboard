"""Interactive Google OAuth bootstrap.

Usage:
    uv run python -m src.auth.bootstrap

Opens a browser, prompts the operator to grant calendar.readonly access,
and writes the resulting credentials to ``GOOGLE_OAUTH_TOKEN_PATH``.

For headless servers, SSH-tunnel the local OAuth port to your laptop:
    ssh -L 8080:localhost:8080 user@server
"""

from __future__ import annotations

import logging
import sys

from google_auth_oauthlib.flow import InstalledAppFlow

from src.auth.google_oauth import READONLY_SCOPES
from src.core.config import Settings
from src.core.logging import configure_logging

log = logging.getLogger(__name__)


def main() -> int:
    """Run the OAuth installed-app flow and persist the token.

    Returns:
        Process exit code (0 on success).
    """
    settings = Settings()
    configure_logging(settings.log_level)

    secrets = settings.google_oauth_client_secrets_path
    if not secrets.exists():
        log.error("Client secrets not found at %s", secrets)
        return 1

    flow = InstalledAppFlow.from_client_secrets_file(str(secrets), READONLY_SCOPES)
    creds = flow.run_local_server(port=0)

    token_path = settings.google_oauth_token_path
    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_text(creds.to_json())

    log.info("Wrote credentials to %s", token_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
