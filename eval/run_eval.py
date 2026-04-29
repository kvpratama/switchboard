"""Run evaluation on Switchboard agent with mocked calendar data."""
# pyright: reportCallIssue=false

from langsmith import evaluate  # type: ignore[import-untyped]

from eval.evaluators import accuracy_evaluator, response_length_evaluator
from eval.run_agent import run_agent


def main():
    """Run evaluation on the dataset."""
    print("Starting Switchboard evaluation...")
    print("Dataset: eval/dataset.json")
    print("Evaluators: accuracy_evaluator, response_length_evaluator")
    print("-" * 60)

    # Run evaluation
    results = evaluate(
        run_agent,
        data="eval/dataset.json",
        evaluators=[accuracy_evaluator, response_length_evaluator],
        experiment_prefix="switchboard-eval",
        max_concurrency=3,
    )  # ty:ignore[no-matching-overload]

    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print(f"Results: {results}")
    print("=" * 60)


if __name__ == "__main__":
    main()
