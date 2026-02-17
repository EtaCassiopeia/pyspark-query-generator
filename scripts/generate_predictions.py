"""Generate predictions from a running API for the evaluation dataset.

Usage:
    python scripts/generate_predictions.py --api http://localhost:8080 --output outputs/predictions.jsonl
    python scripts/generate_predictions.py --api http://localhost:8081 --output outputs/ft_predictions.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

import httpx

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.config import settings
from shared.data_validator import validate_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate predictions via API")
    parser.add_argument("--api", required=True, help="API base URL (e.g. http://localhost:8080)")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--eval-data", help="Evaluation data path (defaults to config)")
    parser.add_argument("--timeout", type=float, default=120.0, help="Request timeout in seconds")
    args = parser.parse_args()

    eval_path = Path(args.eval_data) if args.eval_data else settings.evaluation_data_path
    examples = validate_jsonl(eval_path)
    print(f"Loaded {len(examples)} evaluation examples from {eval_path}")

    api_url = args.api.rstrip("/")

    # Health check
    try:
        resp = httpx.get(f"{api_url}/health", timeout=10.0)
        health = resp.json()
        print(f"API health: {health}")
    except httpx.HTTPError as e:
        print(f"ERROR: Cannot reach API at {api_url}/health — {e}", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for i, ex in enumerate(examples, 1):
            print(f"  [{i}/{len(examples)}] {ex.description[:60]}...", end=" ", flush=True)
            try:
                resp = httpx.post(
                    f"{api_url}/generate",
                    json={"description": ex.description},
                    timeout=args.timeout,
                )
                resp.raise_for_status()
                result = resp.json()
                f.write(json.dumps({"pyspark_code": result["pyspark_code"]}) + "\n")
                print("OK")
            except httpx.HTTPError as e:
                print(f"FAILED: {e}")
                f.write(json.dumps({"pyspark_code": ""}) + "\n")

    print(f"\nPredictions saved to {output_path}")
    print(f"Evaluate with: python -m shared.evaluation.evaluator --predictions {output_path}")


if __name__ == "__main__":
    main()
