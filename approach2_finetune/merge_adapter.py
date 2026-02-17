"""Merge LoRA adapter weights into the base model for standalone deployment."""

from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from shared.config import settings


def merge(
    base_model: str | None = None,
    adapter_path: Path = Path("outputs/lora_adapter"),
    output_path: Path = Path("outputs/merged_model"),
) -> None:
    """Load base model + LoRA adapter and merge into a single model."""
    base_model = base_model or settings.base_model

    print(f"Loading base model: {base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    print(f"Loading adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(model, str(adapter_path))

    print("Merging weights...")
    model = model.merge_and_unload()

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))
    print(f"Merged model saved to {output_path}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument("--base-model", help="Base model name/path")
    parser.add_argument("--adapter-path", default="outputs/lora_adapter")
    parser.add_argument("--output-path", default="outputs/merged_model")
    args = parser.parse_args()

    merge(
        base_model=args.base_model,
        adapter_path=Path(args.adapter_path),
        output_path=Path(args.output_path),
    )


if __name__ == "__main__":
    main()
