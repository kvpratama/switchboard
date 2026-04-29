"""Logging configuration."""

from __future__ import annotations

import logging
import sys

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"
_HANDLER_NAME = "switchboard_console_handler"


def configure_logging(level: str = "INFO") -> None:
    """Configure root logger.

    Args:
        level: Log level name (e.g. "DEBUG", "INFO"). Falls back to INFO if invalid.
    """
    numeric = logging.getLevelName(level.upper())
    if not isinstance(numeric, int):
        numeric = logging.INFO

    root = logging.getLogger()
    # Remove only the handler owned by this module from prior configure_logging calls.
    for handler in list(root.handlers):
        if getattr(handler, "name", None) == _HANDLER_NAME:
            root.removeHandler(handler)

    handler = logging.StreamHandler(stream=sys.stderr)
    handler.set_name(_HANDLER_NAME)
    handler.setFormatter(logging.Formatter(_FORMAT))
    root.addHandler(handler)
    root.setLevel(numeric)
