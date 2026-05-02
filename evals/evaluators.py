"""LLM-as-judge evaluators for Switchboard agent evaluation.

These evaluators import third-party packages (langchain, pydantic, settings)
at module level and therefore CANNOT be uploaded to LangSmith's evaluator
sandbox. Run them locally via ``evaluate(evaluators=[...])`` instead.

For sandbox-safe code evaluators that can be uploaded, see
``evals/evaluators_code.py``.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Annotated, Any, TypedDict, cast

from langchain.chat_models import init_chat_model

from src.agent.prompt_loader import PromptLoader
from src.core.config import Settings

if TYPE_CHECKING:
    from langchain_core.runnables import Runnable
    from langsmith.schemas import Example, Run


class AccuracyGrade(TypedDict):
    """Structured output for accuracy evaluation."""

    reasoning: Annotated[str, ..., "Explain your reasoning"]
    is_accurate: Annotated[bool, ..., "True if response is accurate"]


class EvaluationResult(TypedDict):
    """Result returned by an evaluator function."""

    score: int
    comment: str


def _extract_outputs(obj: Run | Example | dict[str, Any]) -> dict[str, Any]:
    """Extract the ``outputs`` mapping from a Run/Example or dict, defaulting to ``{}``."""
    raw = obj.outputs if hasattr(obj, "outputs") else obj.get("outputs", {})
    return cast("dict[str, Any]", raw or {})


@lru_cache(maxsize=1)
def _get_judge() -> Runnable[Any, AccuracyGrade]:
    """Lazily build the LLM judge so importing this module has no side effects."""
    settings = Settings()
    if (
        settings.llm_provider_eval is None
        or settings.llm_model_eval is None
        or settings.llm_api_key_eval is None
    ):
        raise RuntimeError(
            "Evaluation LLM is not configured. Set LLM_PROVIDER_EVAL, "
            "LLM_MODEL_EVAL and LLM_API_KEY_EVAL in your environment to run "
            "the LLM-as-judge evaluator."
        )
    return cast(
        "Runnable[Any, AccuracyGrade]",
        init_chat_model(
            model=settings.llm_model_eval,
            model_provider=settings.llm_provider_eval,
            api_key=settings.llm_api_key_eval.get_secret_value(),
            base_url=settings.llm_provider_base_url_eval,
            temperature=0,
        ).with_structured_output(AccuracyGrade, method="json_schema", strict=True),
    )


async def accuracy_evaluator(
    run: Run | dict[str, Any],
    example: Example | dict[str, Any],
) -> EvaluationResult:
    """Evaluate if agent response is accurate compared to expected output.

    Args:
        run: LangSmith ``Run`` (or dict for uploaded evaluators) with
            ``outputs.response`` from agent execution.
        example: LangSmith ``Example`` (or dict) with ``outputs.response``
            from the dataset.

    Returns:
        Dictionary with ``score`` (0 or 1) and ``comment`` explaining the grade.
    """
    # Handle both RunTree (local) and dict (uploaded) formats
    run_outputs = _extract_outputs(run)
    example_outputs = _extract_outputs(example)

    actual_response = run_outputs.get("response", "")
    expected_response = example_outputs.get("response", "")

    judge = _get_judge()
    _accuracy_loader = PromptLoader(prompt_name="switchboard-accuracy-evaluator")
    template = await _accuracy_loader.get_template()
    content = template.format(
        expected_response=expected_response,
        actual_response=actual_response,
    )
    grade: AccuracyGrade = await judge.ainvoke(
        [
            {
                "role": "user",
                "content": content,
            }
        ]
    )

    return {
        "score": 1 if grade["is_accurate"] else 0,
        "comment": grade["reasoning"],
    }
