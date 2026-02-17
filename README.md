# PySpark Query Generator

Generate PySpark code from natural language descriptions using two approaches:

1. **RAG + Local LLM** — Retrieve similar examples, build few-shot prompt, generate with Ollama/vLLM
2. **Fine-tuned Model** — QLoRA fine-tuning of DeepSeek Coder 6.7B on Azure ML

Both approaches expose the same FastAPI interface (`/generate`, `/health`) and can be deployed within a private Azure VNet.

## Prerequisites

- Python 3.10+
- Docker & Docker Compose (for RAG approach)
- GPU with 16GB+ VRAM (for fine-tuning / local inference)
- Azure subscription (for Approach 2 cloud training)

## Quickstart

### Install

```bash
# Core + RAG dependencies
pip install -e ".[rag]"

# Core + fine-tuning dependencies
pip install -e ".[finetune]"

# Everything
pip install -e ".[rag,finetune,azure,eval]"
```

### Configure

```bash
cp .env.example .env
# Edit .env with your settings
```

### Validate training data

```bash
python -m shared.data_validator --input data/samples/training_data.jsonl
```

### Approach 1: RAG

```bash
# Build FAISS index
python -m approach1_rag.ingestion

# Start with Docker Compose (includes Ollama)
cd approach1_rag && docker-compose up

# Or run API directly (requires Ollama running separately)
python -m approach1_rag.api
```

### Approach 2: Fine-tuning

```bash
# Prepare data
python -m approach2_finetune.prepare_data

# Train locally (requires GPU)
python -m approach2_finetune.train_lora

# Or submit to Azure ML
python -m approach2_finetune.azure_ml.submit_training_job

# Serve with vLLM
python -m approach2_finetune.serve_vllm --model outputs/merged_model

# Start API
python -m approach2_finetune.api
```

### Evaluate

```bash
python -m shared.evaluation.evaluator --eval-data data/samples/evaluation_data.jsonl
```

### API Usage

Both approaches expose the same interface:

```bash
# Health check
curl http://localhost:8080/health

# Generate PySpark code
curl -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{"description": "Get total sales per category for the last 30 days"}'
```

## Project Structure

```
├── shared/              # Shared config, models, prompts, evaluation
├── approach1_rag/       # RAG pipeline (FAISS + Ollama/vLLM)
├── approach2_finetune/  # QLoRA fine-tuning + Azure ML integration
├── data/samples/        # Training and evaluation JSONL
├── docs/                # Detailed documentation
└── notebooks/           # Interactive exploration notebooks
```

## Documentation

- [Business Proposal](docs/proposal.md) — Cost estimates, timeline, approach comparison
- [Azure Infrastructure](docs/azure-infrastructure.md) — Resource setup guide
- [RAG Guide](docs/approach1-rag-guide.md) — Step-by-step RAG deployment
- [Fine-tuning Guide](docs/approach2-finetune-guide.md) — Training and deployment walkthrough
