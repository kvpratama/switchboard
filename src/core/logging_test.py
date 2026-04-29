"""Tests for src.core.logging."""

from __future__ import annotations

import logging

from src.core.logging import configure_logging


def test_configure_logging_sets_root_level() -> None:
    configure_logging("DEBUG")
    assert logging.getLogger().level == logging.DEBUG

    configure_logging("INFO")
    assert logging.getLogger().level == logging.INFO


def test_configure_logging_invalid_level_falls_back_to_info(
    caplog: object,
) -> None:
    configure_logging("NOT_A_LEVEL")
    assert logging.getLogger().level == logging.INFO
