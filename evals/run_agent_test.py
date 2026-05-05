"""Tests for evals.run_agent."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from evals.run_agent import run_agent


async def test_run_agent_returns_response_and_tool_calls() -> None:
    """run_agent should return both response text and captured tool_calls."""
    tool_call = {
        "name": "create_event",
        "args": {
            "summary": "Lunch",
            "start": "2026-04-30T13:00:00+09:00",
            "end": "2026-04-30T14:00:00+09:00",
        },
        "id": "call-1",
        "type": "tool_call",
    }
    ai_msg_with_tool = AIMessage(content="", tool_calls=[tool_call])
    tool_result = ToolMessage(content='{"id": "mock-event-1"}', tool_call_id="call-1")
    final_ai = AIMessage(content="I've scheduled lunch for tomorrow at 1:00 PM.")

    mock_agent = MagicMock()
    mock_agent.ainvoke = AsyncMock(
        return_value={
            "messages": [
                HumanMessage(content="schedule lunch tomorrow at 1pm"),
                ai_msg_with_tool,
                tool_result,
                final_ai,
            ]
        }
    )

    with (
        patch("evals.run_agent._get_agent", return_value=mock_agent),
        patch("evals.run_agent._get_mock_client") as mock_get_client,
    ):
        mock_get_client.return_value.reset = MagicMock()
        result = await run_agent({"query": "schedule lunch tomorrow at 1pm"})

    assert "response" in result
    assert "tool_calls" in result
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["tool"] == "create_event"
    assert result["tool_calls"][0]["args"]["summary"] == "Lunch"


async def test_run_agent_returns_empty_tool_calls_when_no_tools_used() -> None:
    """run_agent should return empty tool_calls list when agent just responds."""
    final_ai = AIMessage(content="You have 2 events today.")

    mock_agent = MagicMock()
    mock_agent.ainvoke = AsyncMock(
        return_value={
            "messages": [
                HumanMessage(content="What's on my calendar today?"),
                final_ai,
            ]
        }
    )

    with (
        patch("evals.run_agent._get_agent", return_value=mock_agent),
        patch("evals.run_agent._get_mock_client") as mock_get_client,
    ):
        mock_get_client.return_value.reset = MagicMock()
        result = await run_agent({"query": "What's on my calendar today?"})

    assert result["response"] == "You have 2 events today."
    assert result["tool_calls"] == []
