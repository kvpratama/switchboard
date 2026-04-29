"""Logging configuration."""

from __future__ import annotations

import logging
import sys

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"


def configure_logging(level: str = "INFO") -> None:
    """Configure root logger.

    Args:
        level: Log level name (e.g. "DEBUG", "INFO"). Falls back to INFO if invalid.
    """
    numeric = logging.getLevelName(level.upper())
    if not isinstance(numeric, int):
        numeric = logging.INFO

    root = logging.getLogger()
    # Clear handlers from any prior configure_logging call (test re-runs).
    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(logging.Formatter(_FORMAT))
    root.addHandler(handler)
    root.setLevel(numeric)
