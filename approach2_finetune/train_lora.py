"""QLoRA fine-tuning with HuggingFace Transformers + PEFT + TRL."""

from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

from shared.config import settings


@dataclass
class GPUProfile:
    """Training parameters tuned for a specific VRAM budget."""

    name: str
    batch_size: int
    gradient_accumulation_steps: int
    max_seq_length: int


# Profiles keyed by VRAM thresholds (GB). The highest threshold that fits is selected.
_GPU_PROFILES: list[tuple[int, GPUProfile]] = [
    (80, GPUProfile("A100 80GB",    batch_size=4, gradient_accumulation_steps=4,  max_seq_length=2048)),
    (40, GPUProfile("A100 40GB",    batch_size=4, gradient_accumulation_steps=4,  max_seq_length=2048)),
    (24, GPUProfile("A10G/RTX 4090",batch_size=2, gradient_accumulation_steps=8,  max_seq_length=2048)),
    (16, GPUProfile("T4/RTX 4080",  batch_size=1, gradient_accumulation_steps=16, max_seq_length=1536)),
    (0,  GPUProfile("Low VRAM",     batch_size=1, gradient_accumulation_steps=16, max_seq_length=1024)),
]


def detect_gpu_profile() -> GPUProfile:
    """Auto-detect GPU and return an appropriate training profile."""
    if not torch.cuda.is_available():
        print("WARNING: No CUDA GPU detected — training will be extremely slow on CPU")
        return _GPU_PROFILES[-1][1]

    vram_gb = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
    gpu_name = torch.cuda.get_device_name(0)
    print(f"Detected GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)")

    for threshold, profile in _GPU_PROFILES:
        if vram_gb >= threshold:
            print(f"Selected profile: {profile.name} "
                  f"(batch_size={profile.batch_size}, "
                  f"grad_accum={profile.gradient_accumulation_steps}, "
                  f"max_seq_len={profile.max_seq_length})")
            return profile

    return _GPU_PROFILES[-1][1]


def train(
    data_dir: Path | None = None,
    output_dir: Path = Path("outputs/lora_adapter"),
    base_model: str | None = None,
    auto_detect_gpu: bool = True,
) -> None:
    """Run QLoRA fine-tuning on prepared ChatML data.

    When auto_detect_gpu is True, batch size, gradient accumulation, and max
    sequence length are tuned for the detected GPU's VRAM.  Values from .env
    are used as overrides when auto_detect_gpu is False.
    """
    base_model = base_model or settings.base_model
    data_dir = data_dir or settings.training_data_path.parent / "prepared"

    # Determine training parameters
    if auto_detect_gpu:
        profile = detect_gpu_profile()
        batch_size = profile.batch_size
        grad_accum = profile.gradient_accumulation_steps
        max_seq_length = profile.max_seq_length
    else:
        batch_size = settings.batch_size
        grad_accum = settings.gradient_accumulation_steps
        max_seq_length = 2048

    effective_batch = batch_size * grad_accum
    print(f"Effective batch size: {batch_size} x {grad_accum} = {effective_batch}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # QLoRA: 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    # LoRA config
    lora_config = LoraConfig(
        r=settings.lora_r,
        lora_alpha=settings.lora_alpha,
        lora_dropout=settings.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load datasets
    dataset = load_dataset(
        "json",
        data_files={
            "train": str(data_dir / "train.jsonl"),
            "validation": str(data_dir / "val.jsonl"),
        },
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=settings.training_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=settings.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=1,
        save_strategy="epoch",
        eval_strategy="epoch",
        bf16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        report_to="none",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        max_seq_length=max_seq_length,
        packing=False,
    )

    trainer.train()

    # Save adapter
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"LoRA adapter saved to {output_dir}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune with QLoRA")
    parser.add_argument("--data-dir", help="Prepared data directory")
    parser.add_argument("--output-dir", default="outputs/lora_adapter")
    parser.add_argument("--base-model", help="Base model name/path")
    parser.add_argument(
        "--no-auto-gpu",
        action="store_true",
        help="Disable GPU auto-detection; use .env values for batch size etc.",
    )
    args = parser.parse_args()

    train(
        data_dir=Path(args.data_dir) if args.data_dir else None,
        output_dir=Path(args.output_dir),
        base_model=args.base_model,
        auto_detect_gpu=not args.no_auto_gpu,
    )


if __name__ == "__main__":
    main()
