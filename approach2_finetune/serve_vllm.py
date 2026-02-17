"""vLLM serving script with optional LoRA adapter support."""

import subprocess
import sys

from shared.config import settings


def serve(
    model_path: str | None = None,
    lora_adapter: str | None = None,
) -> None:
    """Start a vLLM OpenAI-compatible server.

    Can serve either a merged model or base model + LoRA adapter.
    """
    model_path = model_path or settings.base_model

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--host", settings.vllm_host,
        "--port", str(settings.vllm_port),
        "--gpu-memory-utilization", str(settings.vllm_gpu_memory_utilization),
        "--max-model-len", "4096",
        "--trust-remote-code",
    ]

    if lora_adapter:
        cmd.extend(["--enable-lora", "--lora-modules", f"pyspark-lora={lora_adapter}"])

    print(f"Starting vLLM server: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Serve model with vLLM")
    parser.add_argument("--model", help="Model path (merged model or base model)")
    parser.add_argument("--lora-adapter", help="Path to LoRA adapter (optional)")
    args = parser.parse_args()

    serve(model_path=args.model, lora_adapter=args.lora_adapter)


if __name__ == "__main__":
    main()
