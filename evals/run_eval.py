"""Run evaluation on Switchboard agent with mocked calendar data."""

import asyncio
import json
from pathlib import Path

from langsmith import Client, aevaluate

from evals.evaluators import accuracy_evaluator
from evals.evaluators_code import response_length_evaluator
from evals.run_agent import run_agent

DATASET_PATH = Path(__file__).parent / "dataset.json"
DATASET_NAME = "Switchboard Eval"


def _ensure_dataset() -> str:
    """Create the LangSmith dataset on first run; return its name.

    Idempotent: if the dataset already exists, leave it alone. We deliberately
    do NOT re-upload examples on every run to avoid duplicates — manage
    examples via the LangSmith UI or ``langsmith dataset`` CLI.
    """
    client = Client()
    examples = json.loads(DATASET_PATH.read_text())

    if client.has_dataset(dataset_name=DATASET_NAME):
        return DATASET_NAME

    print(f"Creating LangSmith dataset '{DATASET_NAME}' with {len(examples)} examples...")
    client.create_dataset(dataset_name=DATASET_NAME, description="Switchboard offline eval set")
    client.create_examples(
        inputs=[ex["inputs"] for ex in examples],
        outputs=[ex["outputs"] for ex in examples],
        dataset_name=DATASET_NAME,
    )
    return DATASET_NAME


async def _run() -> None:
    print("Starting Switchboard evaluation...")
    dataset_name = _ensure_dataset()
    print(f"Dataset: {dataset_name}")
    print("Evaluators: accuracy_evaluator, response_length_evaluator")
    print("-" * 60)

    results = await aevaluate(
        run_agent,
        data=dataset_name,
        evaluators=[accuracy_evaluator, response_length_evaluator],
        experiment_prefix="switchboard-eval",
        max_concurrency=3,
    )

    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print(f"Results: {results}")
    print("=" * 60)


def main() -> None:
    """Entry point for the ``switchboard-eval`` console script."""
    asyncio.run(_run())


if __name__ == "__main__":
    main()
