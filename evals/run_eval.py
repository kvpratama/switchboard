"""Run evaluation on Switchboard agent with mocked calendar data."""

import asyncio

from dotenv import load_dotenv
from langsmith import aevaluate

from evals.dataset import ensure_dataset
from evals.evaluators import accuracy_evaluator
from evals.evaluators_code import response_length_evaluator
from evals.run_agent import run_agent

# Load environment variables from .env
load_dotenv()


async def _run() -> None:
    print("Starting Switchboard evaluation...")
    dataset_name = ensure_dataset()
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
