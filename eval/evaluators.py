"""Evaluators for Switchboard agent evaluation."""
# pyright: reportIndexIssue=false, reportAssignmentType=false

import os
from typing import Annotated, TypedDict

from langchain.chat_models import init_chat_model


class AccuracyGrade(TypedDict):
    """Structured output for accuracy evaluation."""

    reasoning: Annotated[str, ..., "Explain your reasoning"]
    is_accurate: Annotated[bool, ..., "True if response is accurate"]


# Initialize LLM judge using same config as agent
judge = init_chat_model(
    model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
    model_provider=os.getenv("LLM_PROVIDER", "openai"),
    api_key=os.getenv("LLM_API_KEY"),
    temperature=0,
).with_structured_output(AccuracyGrade, method="json_schema", strict=True)


async def accuracy_evaluator(run, example):
    """Evaluate if agent response is accurate compared to expected output.

    Args:
        run: Run object with outputs from agent execution.
        example: Example object with expected outputs from dataset.

    Returns:
        Dictionary with score (0 or 1) and comment explaining the grade.
    """
    # Handle both RunTree (local) and dict (uploaded) formats
    run_outputs = run.outputs if hasattr(run, "outputs") else run.get("outputs", {}) or {}
    example_outputs = (
        example.outputs if hasattr(example, "outputs") else example.get("outputs", {}) or {}
    )

    actual_response = run_outputs.get("response", "")
    expected_response = example_outputs.get("response", "")

    # Ask LLM judge to evaluate accuracy
    grade = await judge.ainvoke(  # type: ignore[assignment]
        [
            {
                "role": "user",
                "content": f"""
                Compare the actual response to the expected response.
The actual response should convey the same information as expected, 
but doesn't need to match word-for-word.

Expected: {expected_response}
Actual: {actual_response}

Is the actual response accurate?
""",
            }
        ]
    )

    return {
        "score": 1 if grade["is_accurate"] else 0,  # ty:ignore[not-subscriptable]
        "comment": grade["reasoning"],  # ty:ignore[not-subscriptable]
    }


def response_length_evaluator(run, example):
    """Evaluate if response is concise (not too verbose).

    Args:
        run: Run object with outputs from agent execution.
        example: Example object (not used in this evaluator).

    Returns:
        Dictionary with score (0 or 1) and comment.
    """
    run_outputs = run.outputs if hasattr(run, "outputs") else run.get("outputs", {}) or {}
    response = run_outputs.get("response", "")

    # Penalize responses over 200 characters (should be concise)
    is_concise = len(response) <= 200

    return {
        "score": 1 if is_concise else 0,
        "comment": f"""
        Response length: {len(response)} chars. 
        {"Concise" if is_concise else "Too verbose"}
        """,
    }
