"""Run function for evaluation with mocked calendar client."""

from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

from evals.mock_calendar import MockCalendarClient
from src.agent.builder import build_agent
from src.core.config import Settings

# Load environment variables
load_dotenv()

# Fixed time for evaluation: 2026-04-29T10:00:00+09:00.
# All relative dates ("today", "tomorrow", "Friday") resolve against this so
# eval results are reproducible regardless of the wall clock.
EVAL_TIME = datetime(2026, 4, 29, 10, 0, 0, tzinfo=ZoneInfo("Asia/Tokyo"))

# Cache the built agent across invocations so we don't pay agent-construction
# cost for every example in the dataset.
_agent: Any | None = None


async def _get_agent() -> Any:
    """Build the eval agent on first use and reuse it for subsequent calls."""
    global _agent
    if _agent is None:
        settings = Settings()
        mock_client = MockCalendarClient()
        _agent = await build_agent(
            settings=settings,
            checkpointer=None,  # Stateless for eval
            calendar_client=mock_client.get_mock(),
            timezone="Asia/Tokyo",
            now_provider=lambda: EVAL_TIME,
        )
    return _agent


async def run_agent(inputs: dict) -> dict:
    """Run agent on evaluation inputs with mocked calendar.

    Args:
        inputs: Dictionary with "query" key containing user question.

    Returns:
        Dictionary with "response" key containing agent's answer.
    """
    agent = await _get_agent()

    result = await agent.ainvoke({"messages": [{"role": "user", "content": inputs["query"]}]})

    # Extract final response
    messages = result.get("messages", [])
    if messages:
        final_message = messages[-1]
        response = (
            final_message.content if hasattr(final_message, "content") else str(final_message)
        )
    else:
        response = "No response"

    return {"response": response}
