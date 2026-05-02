"""Tests for eval.evaluators_code (sandbox-safe code evaluators)."""

from __future__ import annotations

from types import SimpleNamespace

from evals.evaluators_code import response_length_evaluator


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
