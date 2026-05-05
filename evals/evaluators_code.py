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


def _extract_tool_calls(
    run: Run | dict[str, Any],
    example: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return ``(actual_tool_calls, expected_tool_calls)`` from a run/example pair.

    Handles both LangSmith ``Run`` objects (``.outputs`` attribute) and the
    dict form used by uploaded evaluators.
    """
    raw_run = run.outputs if hasattr(run, "outputs") else run.get("outputs", {})
    run_outputs = cast("dict[str, Any]", raw_run or {})
    raw_ex = example.outputs if hasattr(example, "outputs") else example.get("outputs", {})
    example_outputs = cast("dict[str, Any]", raw_ex or {})
    return run_outputs.get("tool_calls", []), example_outputs.get("tool_calls", [])


class _InvocationResult(TypedDict):
    """Result shape returned by ``tool_invocation_evaluator``.

    Attributes:
        score: ``1`` if all expected tools were invoked, ``0`` otherwise.
        comment: Human-readable summary of expected vs. actual tool calls.
    """

    score: int
    comment: str


def tool_invocation_evaluator(
    run: Run | dict[str, Any],
    example: Any,
) -> _InvocationResult:
    """Score 1 if expected tool(s) were called, 0 otherwise.

    Compares tool names only — argument validation is handled by
    ``parameter_accuracy_evaluator``.

    Args:
        run: LangSmith ``Run`` (or dict) with ``outputs.tool_calls``.
        example: Dataset example with ``outputs.tool_calls``.

    Returns:
        Dict with ``score`` (0 or 1) and a ``comment``.
    """
    actual_calls, expected_calls = _extract_tool_calls(run, example)

    actual_tools = {c.get("tool") for c in actual_calls if c.get("tool")}
    expected_tools = {c.get("tool") for c in expected_calls if c.get("tool")}

    # When no tools are expected, any tool call is a failure.
    if not expected_tools:
        if not actual_tools:
            return {"score": 1, "comment": "No tools expected and none called"}
        return {
            "score": 0,
            "comment": (f"No tools expected, but got unexpected calls: {sorted(actual_tools)}"),
        }

    if expected_tools <= actual_tools:
        return {
            "score": 1,
            "comment": f"Expected {sorted(expected_tools)}, got {sorted(actual_tools)}",
        }

    missing = expected_tools - actual_tools
    return {
        "score": 0,
        "comment": (
            f"Expected {sorted(expected_tools)}, got {sorted(actual_tools)}; "
            f"missing {sorted(missing)}"
        ),
    }


class _ParameterResult(TypedDict):
    """Result shape returned by ``parameter_accuracy_evaluator``.

    Attributes:
        score: Fraction of matching ``create_event`` parameters in the
            range ``0.0``–``1.0``.
        comment: Human-readable summary listing matched and mismatched
            parameter names.
    """

    score: float
    comment: str


def parameter_accuracy_evaluator(
    run: Run | dict[str, Any],
    example: Any,
) -> _ParameterResult:
    """Score percentage of correct create_event parameters (0.0 to 1.0).

    Compares each parameter (summary, start, end, location, description)
    between actual and expected ``create_event`` args. ``None`` vs missing
    key is treated as equivalent. Extra parameters in actual are ignored.

    Args:
        run: LangSmith ``Run`` (or dict) with ``outputs.tool_calls``.
        example: Dataset example with ``outputs.tool_calls``.

    Returns:
        Dict with ``score`` (0.0–1.0) and a ``comment`` listing
        matches/mismatches.
    """
    actual_calls, expected_calls = _extract_tool_calls(run, example)

    # Find the first create_event call in each
    actual_create = next((c for c in actual_calls if c.get("tool") == "create_event"), None)
    expected_create = next((c for c in expected_calls if c.get("tool") == "create_event"), None)

    if expected_create is None:
        return {"score": 1.0, "comment": "No create_event expected"}

    if actual_create is None:
        return {"score": 0.0, "comment": "Expected create_event but none was called"}

    expected_args = expected_create.get("args", {})
    actual_args = actual_create.get("args", {})

    # Compare each expected parameter
    params_to_check = ["summary", "start", "end", "location", "description"]
    matched: list[str] = []
    mismatched: list[str] = []

    for param in params_to_check:
        if param not in expected_args:
            continue
        expected_val = expected_args[param]
        actual_val = actual_args.get(param)

        # None vs missing key treated as equivalent
        if expected_val is None and actual_val is None:
            matched.append(param)
        elif expected_val == actual_val:
            matched.append(param)
        else:
            mismatched.append(param)

    total = len(matched) + len(mismatched)
    if total == 0:
        return {"score": 1.0, "comment": "No parameters to compare"}

    score = len(matched) / total
    comment_parts = []
    if matched:
        comment_parts.append(f"matched: {', '.join(matched)}")
    if mismatched:
        comment_parts.append(f"mismatched: {', '.join(mismatched)}")
    return {"score": score, "comment": "; ".join(comment_parts)}
