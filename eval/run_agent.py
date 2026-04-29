"""Run function for evaluation with mocked calendar client."""

from datetime import datetime
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

from eval.mock_calendar import MockCalendarClient
from src.agent.builder import build_agent
from src.core.config import Settings

# Load environment variables
load_dotenv()

# Fixed time for evaluation: 2026-04-29T17:00:00+09:00
EVAL_TIME = datetime(2026, 4, 29, 17, 0, 0, tzinfo=ZoneInfo("Asia/Tokyo"))


async def run_agent(inputs: dict) -> dict:
    """Run agent on evaluation inputs with mocked calendar.

    Args:
        inputs: Dictionary with "query" key containing user question.

    Returns:
        Dictionary with "response" key containing agent's answer.
    """
    # Load settings
    settings = Settings()

    # Create mock calendar client
    mock_client = MockCalendarClient()

    # Build agent with mocked client and fixed timezone
    agent = await build_agent(
        settings=settings,
        checkpointer=None,  # Stateless for eval
        calendar_client=mock_client.get_mock(),
        timezone="Asia/Tokyo",
    )

    # Invoke agent with query
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
