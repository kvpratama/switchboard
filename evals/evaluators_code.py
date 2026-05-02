"""Sandbox-safe code evaluators (uploadable to LangSmith).

This module is intentionally restricted to the standard library and uses the
``(run, example)`` signature documented for offline evaluators. Imports stay
inside the function body (or guarded by ``TYPE_CHECKING``) so the module is
safe to upload via:

    langsmith evaluator upload evals/evaluators_code.py \\
      --name "Response Length" --function response_length_evaluator \\
      --dataset "Switchboard Eval" --replace --api-key $LANGSMITH_API_KEY
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict, cast

if TYPE_CHECKING:
    from langsmith import Run


class _LengthResult(TypedDict):
    score: int
    comment: str


def response_length_evaluator(
    run: Run | dict[str, Any],
    example: Any,
) -> _LengthResult:
    """Score 1 if the agent response is concise (<= 300 chars), else 0.

    Args:
        run: LangSmith ``Run`` (or dict for uploaded evaluators) with
            ``outputs.response``.
        example: Dataset example (unused by this evaluator).

    Returns:
        Dict with ``score`` (0 or 1) and a one-line ``comment``.
    """
    raw_outputs = run.outputs if hasattr(run, "outputs") else run.get("outputs", {})
    run_outputs = cast("dict[str, Any]", raw_outputs or {})
    response = run_outputs.get("response", "")

    length = len(response)
    is_concise = length <= 300
    verdict = "concise" if is_concise else "too verbose"

    return {
        "score": 1 if is_concise else 0,
        "comment": f"Response length: {length} chars ({verdict})",
    }
