"""Convert JSONL training data to ChatML instruction format for fine-tuning."""

import json
from pathlib import Path

from shared.config import settings
from shared.data_validator import validate_jsonl
from shared.prompt_templates import SYSTEM_PROMPT


def prepare_chatml(
    input_path: Path | None = None,
    output_path: Path | None = None,
    val_split: float = 0.1,
) -> None:
    """Convert training examples to ChatML conversation format.

    Produces train.jsonl and val.jsonl with the format expected by TRL SFTTrainer.
    """
    input_path = input_path or settings.training_data_path
    examples = validate_jsonl(input_path)

    # Deterministic split
    split_idx = max(1, int(len(examples) * (1 - val_split)))
    train_examples = examples[:split_idx]
    val_examples = examples[split_idx:]

    output_dir = Path(output_path) if output_path else input_path.parent / "prepared"
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_data in [("train", train_examples), ("val", val_examples)]:
        out_file = output_dir / f"{split_name}.jsonl"
        with open(out_file, "w") as f:
            for ex in split_data:
                conversation = {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": ex.description},
                        {"role": "assistant", "content": ex.pyspark_code},
                    ]
                }
                f.write(json.dumps(conversation) + "\n")
        print(f"Wrote {len(split_data)} examples to {out_file}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Prepare data for fine-tuning")
    parser.add_argument("--input", help="Input JSONL path")
    parser.add_argument("--output", help="Output directory")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split ratio")
    args = parser.parse_args()

    prepare_chatml(
        input_path=Path(args.input) if args.input else None,
        output_path=Path(args.output) if args.output else None,
        val_split=args.val_split,
    )


if __name__ == "__main__":
    main()
