"""Central configuration for the PySpark Query Generator project."""

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    # Paths
    project_root: Path = Path(".")
    data_dir: Path = Path("data/samples")
    model_cache_dir: Path = Path("models")

    # Embedding / RAG
    embedding_model: str = "all-MiniLM-L6-v2"
    faiss_index_path: Path = Path("data/faiss_index")
    rag_top_k: int = 3

    # LLM (Ollama / vLLM compatible)
    llm_base_url: str = "http://localhost:11434/v1"
    llm_model: str = "deepseek-coder:6.7b-instruct"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 2048

    # Fine-tuning
    base_model: str = "deepseek-ai/deepseek-coder-6.7b-instruct"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    training_epochs: int = 3
    learning_rate: float = 2e-4
    batch_size: int = 4
    gradient_accumulation_steps: int = 4

    # vLLM serving
    vllm_host: str = "0.0.0.0"
    vllm_port: int = 8000
    vllm_gpu_memory_utilization: float = 0.9

    # Azure ML
    azure_subscription_id: str = ""
    azure_resource_group: str = ""
    azure_ml_workspace_name: str = ""
    azure_region: str = "eastus2"
    azure_compute_name: str = "gpu-cluster"
    azure_compute_size: str = "Standard_NC24ads_A100_v4"
    azure_compute_min_nodes: int = 0
    azure_compute_max_nodes: int = 1

    @property
    def training_data_path(self) -> Path:
        return self.data_dir / "training_data.jsonl"

    @property
    def evaluation_data_path(self) -> Path:
        return self.data_dir / "evaluation_data.jsonl"


settings = Settings()
