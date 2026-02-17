"""vLLM serving script with optional LoRA adapter and quantization support."""

import subprocess
import sys

from shared.config import settings


def serve(
    model_path: str | None = None,
    lora_adapter: str | None = None,
    quantization: str | None = None,
) -> None:
    """Start a vLLM OpenAI-compatible server.

    Can serve either a merged model or base model + LoRA adapter.
    Supports quantization (awq, gptq, squeezellm) for fitting larger
    models into limited VRAM (e.g. 16 GB on RTX 4080 Super).
    """
    model_path = model_path or settings.base_model

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--host", settings.vllm_host,
        "--port", str(settings.vllm_port),
        "--gpu-memory-utilization", str(settings.vllm_gpu_memory_utilization),
        "--max-model-len", str(settings.vllm_max_model_len),
        "--trust-remote-code",
    ]

    # Quantization — needed for 16 GB GPUs with non-quantized models
    quant = quantization or settings.vllm_quantization
    if quant:
        cmd.extend(["--quantization", quant])

    if lora_adapter:
        cmd.extend(["--enable-lora", "--lora-modules", f"pyspark-lora={lora_adapter}"])

    print(f"Starting vLLM server: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Serve model with vLLM")
    parser.add_argument("--model", help="Model path (merged model or base model)")
    parser.add_argument("--lora-adapter", help="Path to LoRA adapter (optional)")
    parser.add_argument(
        "--quantization",
        choices=["awq", "gptq", "squeezellm", ""],
        default=None,
        help="Quantization method (for 16 GB GPUs use awq or gptq)",
    )
    args = parser.parse_args()

    serve(
        model_path=args.model,
        lora_adapter=args.lora_adapter,
        quantization=args.quantization,
    )


if __name__ == "__main__":
    main()
