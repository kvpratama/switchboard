"""Run function for evaluation with mocked calendar client."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from evals.mock_calendar import MockCalendarClient
from src.agent.builder import build_agent
from src.core.config import Settings

# Fixed time for evaluation: 2026-04-29T10:00:00+09:00.
# All relative dates ("today", "tomorrow", "Friday") resolve against this so
# eval results are reproducible regardless of the wall clock.
EVAL_TIME = datetime(2026, 4, 29, 10, 0, 0, tzinfo=ZoneInfo("Asia/Tokyo"))

# Shared mock client so evaluators can inspect recorded calls.
_mock_client: MockCalendarClient | None = None

# Cache the built agent across invocations so we don't pay agent-construction
# cost for every example in the dataset.
_agent: Any | None = None


def _get_mock_client() -> MockCalendarClient:
    """Return the shared mock client, creating it on first use."""
    global _mock_client
    if _mock_client is None:
        _mock_client = MockCalendarClient()
    return _mock_client


async def _get_agent() -> Any:
    """Build the eval agent on first use and reuse it for subsequent calls."""
    global _agent
    if _agent is None:
        settings = Settings()
        mock_client = _get_mock_client()
        _agent = await build_agent(
            settings=settings,
            checkpointer=None,  # Stateless for eval
            calendar_client=mock_client.get_mock(),
            timezone="Asia/Tokyo",
            now_provider=lambda: EVAL_TIME,
            enable_hitl=False,  # Auto-approve create_event in eval
        )
    return _agent


def _extract_tool_calls(messages: list[Any]) -> list[dict[str, Any]]:
    """Extract tool call name+args from AIMessage objects in the message list.

    Iterates all messages and collects ``tool_calls`` from each AIMessage.
    No deduplication needed because ``ainvoke`` returns the final state
    with each message appearing exactly once.

    Args:
        messages: List of message objects from the agent result.

    Returns:
        List of dicts with ``tool`` and ``args`` keys.
    """
    tool_calls: list[dict[str, Any]] = []
    for msg in messages:
        tc = getattr(msg, "tool_calls", None)
        if tc:
            for call in tc:
                tool_calls.append(
                    {"tool": call.get("name", call.get("type", "")), "args": call.get("args", {})}
                )
    return tool_calls


async def run_agent(inputs: dict) -> dict:
    """Run agent on evaluation inputs with mocked calendar.

    Uses ``ainvoke`` and extracts tool calls from the final message list.
    Resets the mock client before each run to avoid state leakage.

    Args:
        inputs: Dictionary with ``query`` key containing user question.

    Returns:
        Dictionary with ``response`` (final answer text) and ``tool_calls``
        (list of ``{tool, args}`` dicts captured from execution).
    """
    # Reset mock state before each run
    _get_mock_client().reset()

    agent = await _get_agent()

    config = {"configurable": {"thread_id": f"eval-{uuid.uuid4()}"}}
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": inputs["query"]}]},
        config=config,
    )

    messages = result.get("messages", [])
    tool_calls = _extract_tool_calls(messages)

    # Extract final response from last message with content
    final_response = "No response"
    if messages:
        final_message = messages[-1]
        final_response = (
            final_message.content if hasattr(final_message, "content") else str(final_message)
        )

    return {"response": final_response, "tool_calls": tool_calls}
