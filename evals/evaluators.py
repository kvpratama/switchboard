"""LLM-as-judge evaluators for Switchboard agent evaluation.

These evaluators import third-party packages (langchain, pydantic, settings)
at module level and therefore CANNOT be uploaded to LangSmith's evaluator
sandbox. Run them locally via ``evaluate(evaluators=[...])`` instead.

For sandbox-safe code evaluators that can be uploaded, see
``evals/evaluators_code.py``.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated, Any, TypedDict

from langchain.chat_models import init_chat_model

from src.core.config import Settings


class AccuracyGrade(TypedDict):
    """Structured output for accuracy evaluation."""

    reasoning: Annotated[str, ..., "Explain your reasoning"]
    is_accurate: Annotated[bool, ..., "True if response is accurate"]


@lru_cache(maxsize=1)
def _get_judge() -> Any:
    """Lazily build the LLM judge so importing this module has no side effects."""
    settings = Settings()
    return init_chat_model(
        model=settings.llm_model_eval,
        model_provider=settings.llm_provider_eval,
        api_key=settings.llm_api_key_eval.get_secret_value(),
        base_url=settings.llm_provider_base_url_eval,
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
    run_outputs = (run.outputs if hasattr(run, "outputs") else run.get("outputs", {})) or {}
    example_outputs = (
        example.outputs if hasattr(example, "outputs") else example.get("outputs", {})
    ) or {}

    actual_response = run_outputs.get("response", "")
    expected_response = example_outputs.get("response", "")

    judge = _get_judge()
    grade: AccuracyGrade = await judge.ainvoke(
        [
            {
                "role": "user",
                "content": (
                    "Compare the actual response to the expected response. "
                    "The actual response should convey the same information as "
                    "expected, but does not need to match word-for-word.\n\n"
                    f"Expected: {expected_response}\n"
                    f"Actual: {actual_response}\n\n"
                    "Is the actual response accurate?"
                ),
            }
        ]
    )

    return {
        "score": 1 if grade["is_accurate"] else 0,
        "comment": grade["reasoning"],
    }
