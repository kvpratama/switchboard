"""Sandbox-safe code evaluators (uploadable to LangSmith).

This module is intentionally restricted to the standard library and uses the
``(run, example)`` signature documented for offline evaluators. Imports stay
inside the function body so the module is safe to upload via:

    langsmith evaluator upload evals/evaluators_code.py \\
      --name "Response Length" --function response_length_evaluator \\
      --dataset "Switchboard Eval" --replace --api-key $LANGSMITH_API_KEY
"""


def response_length_evaluator(run, example):
    """Score 1 if the agent response is concise (<= 200 chars), else 0.

    Args:
        run: LangSmith ``Run`` (or dict for uploaded evaluators) with
            ``outputs.response``.
        example: Dataset example (unused by this evaluator).

    Returns:
        Dict with ``score`` (0 or 1) and a one-line ``comment``.
    """
    run_outputs = (run.outputs if hasattr(run, "outputs") else run.get("outputs", {})) or {}
    response = run_outputs.get("response", "")

    length = len(response)
    is_concise = length <= 300
    verdict = "concise" if is_concise else "too verbose"

    return {
        "score": 1 if is_concise else 0,
        "comment": f"Response length: {length} chars ({verdict})",
    }
