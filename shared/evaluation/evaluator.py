"""Run evaluation pipeline: generate code for held-out examples and compute metrics."""

import json
import sys
from pathlib import Path

from shared.config import settings
from shared.data_models import TrainingExample
from shared.data_validator import validate_jsonl
from shared.evaluation.metrics import (
    bleu_score,
    compute_all_metrics,
    exact_match,
    operation_coverage,
    structural_similarity,
)


def load_evaluation_data(path: Path | None = None) -> list[TrainingExample]:
    """Load and validate the evaluation dataset."""
    path = path or settings.evaluation_data_path
    return validate_jsonl(path)


def evaluate_predictions(
    examples: list[TrainingExample],
    predictions: list[str],
) -> dict:
    """Evaluate a list of predictions against reference examples.

    Returns per-example and aggregate metrics.
    """
    references = [ex.pyspark_code for ex in examples]
    aggregate = compute_all_metrics(references, predictions)

    per_example = []
    for ex, pred in zip(examples, predictions):
        per_example.append(
            {
                "description": ex.description,
                "bleu": bleu_score(ex.pyspark_code, pred),
                "exact_match": exact_match(ex.pyspark_code, pred),
                "operation_coverage": operation_coverage(ex.pyspark_code, pred),
                "structural_similarity": structural_similarity(ex.pyspark_code, pred),
            }
        )

    return {"aggregate": aggregate, "per_example": per_example}


def print_report(results: dict) -> None:
    """Print a formatted evaluation report."""
    print("=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)

    agg = results["aggregate"]
    print(f"\n{'Metric':<25} {'Score':>10}")
    print("-" * 36)
    for metric, score in agg.items():
        print(f"{metric:<25} {score:>10.4f}")

    print(f"\n{'Per-Example Results':}")
    print("-" * 60)
    for i, ex in enumerate(results["per_example"], 1):
        print(f"\n  Example {i}: {ex['description'][:60]}...")
        print(f"    BLEU: {ex['bleu']:.4f}  |  OpCov: {ex['operation_coverage']:.4f}  |  Struct: {ex['structural_similarity']:.4f}  |  Exact: {ex['exact_match']}")


def main() -> None:
    """Run evaluation. Expects a predictions file or generates via an approach."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate PySpark code generation")
    parser.add_argument(
        "--predictions",
        help="Path to JSONL file with predictions (one {'pyspark_code': ...} per line)",
    )
    parser.add_argument(
        "--eval-data",
        help="Path to evaluation JSONL (defaults to config)",
    )
    args = parser.parse_args()

    eval_path = Path(args.eval_data) if args.eval_data else None
    examples = load_evaluation_data(eval_path)
    print(f"Loaded {len(examples)} evaluation examples")

    if args.predictions:
        pred_path = Path(args.predictions)
        predictions = []
        with open(pred_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    predictions.append(json.loads(line)["pyspark_code"])
    else:
        print("No predictions file provided. Using reference code as baseline (perfect score).")
        predictions = [ex.pyspark_code for ex in examples]

    results = evaluate_predictions(examples, predictions)
    print_report(results)


if __name__ == "__main__":
    main()
