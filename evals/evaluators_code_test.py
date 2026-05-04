"""Tests for eval.evaluators_code (sandbox-safe code evaluators)."""

from __future__ import annotations

from types import SimpleNamespace

from evals.evaluators_code import (
    parameter_accuracy_evaluator,
    response_length_evaluator,
    tool_invocation_evaluator,
)


def test_response_length_evaluator_concise_scores_one() -> None:
    run = SimpleNamespace(outputs={"response": "Short answer."})
    example = SimpleNamespace(outputs={})

    result = response_length_evaluator(run, example)

    assert result["score"] == 1
    assert "13 chars" in result["comment"]
    assert "concise" in result["comment"]


def test_response_length_evaluator_verbose_scores_zero() -> None:
    long_response = "x" * 301
    run = SimpleNamespace(outputs={"response": long_response})
    example = SimpleNamespace(outputs={})

    result = response_length_evaluator(run, example)

    assert result["score"] == 0
    assert "too verbose" in result["comment"]


def test_response_length_evaluator_handles_uploaded_dict_format() -> None:
    # Uploaded evaluators receive plain dicts, not RunTree objects.
    run = {"outputs": {"response": "ok"}}
    example = {"outputs": {}}

    result = response_length_evaluator(run, example)

    assert result["score"] == 1


def test_response_length_evaluator_handles_missing_outputs() -> None:
    run = SimpleNamespace(outputs=None)
    example = SimpleNamespace(outputs={})

    result = response_length_evaluator(run, example)

    assert result["score"] == 1  # empty string is "concise"
    assert "0 chars" in result["comment"]


# --- tool_invocation_evaluator ---


def test_tool_invocation_score_1_when_expected_tool_called() -> None:
    run = {"outputs": {"tool_calls": [{"tool": "create_event", "args": {"summary": "Lunch"}}]}}
    example = {"outputs": {"tool_calls": [{"tool": "create_event", "args": {"summary": "Lunch"}}]}}

    result = tool_invocation_evaluator(run, example)

    assert result["score"] == 1


def test_tool_invocation_score_0_when_wrong_tool_called() -> None:
    run = {"outputs": {"tool_calls": [{"tool": "list_events", "args": {}}]}}
    example = {"outputs": {"tool_calls": [{"tool": "create_event", "args": {}}]}}

    result = tool_invocation_evaluator(run, example)

    assert result["score"] == 0


def test_tool_invocation_score_0_when_no_tool_called_but_expected() -> None:
    run = {"outputs": {"tool_calls": []}}
    example = {"outputs": {"tool_calls": [{"tool": "create_event", "args": {}}]}}

    result = tool_invocation_evaluator(run, example)

    assert result["score"] == 0
    assert "create_event" in result["comment"]


def test_tool_invocation_score_1_when_expected_tool_among_multiple() -> None:
    run = {
        "outputs": {
            "tool_calls": [
                {"tool": "list_events", "args": {}},
                {"tool": "create_event", "args": {"summary": "Lunch"}},
            ]
        }
    }
    example = {"outputs": {"tool_calls": [{"tool": "create_event", "args": {}}]}}

    result = tool_invocation_evaluator(run, example)

    assert result["score"] == 1


def test_tool_invocation_score_1_when_no_tool_expected_and_none_called() -> None:
    run = {"outputs": {"tool_calls": []}}
    example = {"outputs": {"tool_calls": []}}

    result = tool_invocation_evaluator(run, example)

    assert result["score"] == 1


def test_tool_invocation_handles_runtree_format() -> None:
    """Handle RunTree objects that use .outputs attribute instead of dict access."""

    class FakeRun:
        outputs = {"tool_calls": [{"tool": "create_event", "args": {}}]}

    class FakeExample:
        outputs = {"tool_calls": [{"tool": "create_event", "args": {}}]}

    result = tool_invocation_evaluator(FakeRun(), FakeExample())

    assert result["score"] == 1


# --- parameter_accuracy_evaluator ---


def test_parameter_accuracy_score_1_when_all_params_match() -> None:
    run = {
        "outputs": {
            "tool_calls": [
                {
                    "tool": "create_event",
                    "args": {
                        "summary": "Lunch",
                        "start": "2026-04-30T13:00:00+09:00",
                        "end": "2026-04-30T14:00:00+09:00",
                    },
                }
            ]
        }
    }
    example = {
        "outputs": {
            "tool_calls": [
                {
                    "tool": "create_event",
                    "args": {
                        "summary": "Lunch",
                        "start": "2026-04-30T13:00:00+09:00",
                        "end": "2026-04-30T14:00:00+09:00",
                    },
                }
            ]
        }
    }

    result = parameter_accuracy_evaluator(run, example)

    assert result["score"] == 1.0


def test_parameter_accuracy_score_partial_when_some_params_wrong() -> None:
    run = {
        "outputs": {
            "tool_calls": [
                {
                    "tool": "create_event",
                    "args": {
                        "summary": "Lunch",
                        "start": "2026-04-30T13:00:00+09:00",
                        "end": "2026-04-30T15:00:00+09:00",
                    },
                }
            ]
        }
    }
    example = {
        "outputs": {
            "tool_calls": [
                {
                    "tool": "create_event",
                    "args": {
                        "summary": "Lunch",
                        "start": "2026-04-30T13:00:00+09:00",
                        "end": "2026-04-30T14:00:00+09:00",
                    },
                }
            ]
        }
    }

    result = parameter_accuracy_evaluator(run, example)

    assert result["score"] == 2 / 3


def test_parameter_accuracy_score_0_when_no_tool_called() -> None:
    run = {"outputs": {"tool_calls": []}}
    example = {
        "outputs": {
            "tool_calls": [
                {
                    "tool": "create_event",
                    "args": {
                        "summary": "Lunch",
                        "start": "2026-04-30T13:00:00+09:00",
                        "end": "2026-04-30T14:00:00+09:00",
                    },
                }
            ]
        }
    }

    result = parameter_accuracy_evaluator(run, example)

    assert result["score"] == 0.0


def test_parameter_accuracy_none_vs_missing_key_equivalent() -> None:
    run = {
        "outputs": {
            "tool_calls": [
                {
                    "tool": "create_event",
                    "args": {
                        "summary": "Lunch",
                        "start": "2026-04-30T13:00:00+09:00",
                        "end": "2026-04-30T14:00:00+09:00",
                    },
                }
            ]
        }
    }
    example = {
        "outputs": {
            "tool_calls": [
                {
                    "tool": "create_event",
                    "args": {
                        "summary": "Lunch",
                        "start": "2026-04-30T13:00:00+09:00",
                        "end": "2026-04-30T14:00:00+09:00",
                        "location": None,
                        "description": None,
                    },
                }
            ]
        }
    }

    result = parameter_accuracy_evaluator(run, example)

    assert result["score"] == 1.0


def test_parameter_accuracy_ignores_extra_params_in_actual() -> None:
    run = {
        "outputs": {
            "tool_calls": [
                {
                    "tool": "create_event",
                    "args": {
                        "summary": "Lunch",
                        "start": "2026-04-30T13:00:00+09:00",
                        "end": "2026-04-30T14:00:00+09:00",
                        "location": "Cafe",
                    },
                }
            ]
        }
    }
    example = {
        "outputs": {
            "tool_calls": [
                {
                    "tool": "create_event",
                    "args": {
                        "summary": "Lunch",
                        "start": "2026-04-30T13:00:00+09:00",
                        "end": "2026-04-30T14:00:00+09:00",
                    },
                }
            ]
        }
    }

    result = parameter_accuracy_evaluator(run, example)

    assert result["score"] == 1.0


def test_parameter_accuracy_handles_runtree_format() -> None:

    class FakeRun:
        outputs = {
            "tool_calls": [
                {
                    "tool": "create_event",
                    "args": {
                        "summary": "Lunch",
                        "start": "2026-04-30T13:00:00+09:00",
                        "end": "2026-04-30T14:00:00+09:00",
                    },
                }
            ]
        }

    class FakeExample:
        outputs = {
            "tool_calls": [
                {
                    "tool": "create_event",
                    "args": {
                        "summary": "Lunch",
                        "start": "2026-04-30T13:00:00+09:00",
                        "end": "2026-04-30T14:00:00+09:00",
                    },
                }
            ]
        }

    result = parameter_accuracy_evaluator(FakeRun(), FakeExample())

    assert result["score"] == 1.0
