"""Validate JSONL training/evaluation data files."""

import json
import sys
from pathlib import Path

from pydantic import ValidationError

from shared.data_models import TrainingExample


def validate_jsonl(path: Path) -> list[TrainingExample]:
    """Validate a JSONL file and return parsed examples.

    Raises ValueError with details on invalid lines.
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    examples: list[TrainingExample] = []
    errors: list[str] = []

    with open(path) as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                example = TrainingExample(**data)
                examples.append(example)
            except json.JSONDecodeError as e:
                errors.append(f"Line {line_num}: Invalid JSON — {e}")
            except ValidationError as e:
                errors.append(f"Line {line_num}: Validation error — {e}")

    if errors:
        raise ValueError(f"Validation failed for {path}:\n" + "\n".join(errors))

    return examples


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Validate JSONL training data")
    parser.add_argument("--input", required=True, help="Path to JSONL file")
    args = parser.parse_args()

    path = Path(args.input)
    try:
        examples = validate_jsonl(path)
        print(f"Validated {len(examples)} examples from {path}")
        for ex in examples:
            print(f"  [{ex.complexity}] {ex.description[:80]}...")
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
