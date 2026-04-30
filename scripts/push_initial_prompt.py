#!/usr/bin/env python3
"""One-time script to push the initial system prompt to LangSmith Prompt Hub.

Run once after creating the LangSmith account:

    uv run python scripts/push_initial_prompt.py

Requires LANGSMITH_API_KEY in the environment.
"""

from __future__ import annotations

import sys

from langsmith import Client

from src.agent.prompt_loader import _FALLBACK_REGISTRY


def main() -> None:
    """Push all registered fallback templates to LangSmith Prompt Hub."""
    client = Client()
    failed = False
    for prompt_name, template in _FALLBACK_REGISTRY.items():
        try:
            client.push_prompt(prompt_name, object=template)
        except Exception as exc:
            print(f"Failed to push prompt '{prompt_name}': {exc}", file=sys.stderr)
            failed = True
        else:
            print(f"Successfully pushed prompt '{prompt_name}' to LangSmith")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
