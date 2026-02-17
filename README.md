# PySpark Query Generator

Generate PySpark code from natural language descriptions using two approaches:

1. **RAG + Local LLM** — Retrieve similar examples, build few-shot prompt, generate with Ollama/vLLM
2. **Fine-tuned Model** — QLoRA fine-tuning of DeepSeek Coder 6.7B on Azure ML or locally

Both approaches expose the same FastAPI interface (`/generate`, `/health`) and can run on a local GPU or within a private Azure VNet.

## Prerequisites

- Python 3.10+
- NVIDIA GPU with 16 GB+ VRAM (RTX 4080 Super, RTX 4090, T4, A100, etc.)
- Docker & Docker Compose (optional, for containerized deployment)
- Ollama (for Approach 1)
- Azure subscription (optional, for cloud training)

## Local Quickstart (GPU)

```bash
# One-time setup: copies .env.local (tuned for 16 GB VRAM) and installs deps
make setup-local

# Verify GPU is detected
make gpu-check

# Validate training data
make validate
```

### Approach 1: RAG

```bash
make index          # Build FAISS index from training examples
make ollama-pull    # Download the model (one-time, ~4 GB)
make ollama-start   # Start Ollama (Terminal 1)
make rag-api        # Start RAG API on :8080 (Terminal 2)
make rag-test       # Smoke test (Terminal 3)
```

### Approach 2: Fine-tuning

```bash
make prepare-data   # Convert JSONL → ChatML format
make train          # QLoRA training (~30 min on RTX 4080 Super)
make merge          # Merge LoRA adapter into standalone model
make vllm-start     # Start vLLM server on :8000 (Terminal 1)
make finetune-api   # Start fine-tune API on :8081 (Terminal 2)
make finetune-test  # Smoke test (Terminal 3)
```

### Docker (local GPU)

```bash
make docker-up-rag       # Approach 1: Ollama + RAG API
make docker-up-finetune  # Approach 2: vLLM + fine-tune API
make docker-train        # Run training in Docker (prepare → train → merge)
make docker-down         # Stop all
```

### Evaluate

```bash
# Generate predictions from a running API
python scripts/generate_predictions.py --api http://localhost:8080 --output outputs/rag_preds.jsonl
python scripts/generate_predictions.py --api http://localhost:8081 --output outputs/ft_preds.jsonl

# Compare
make evaluate-predictions PRED=outputs/rag_preds.jsonl
make evaluate-predictions PRED=outputs/ft_preds.jsonl
```

### API Usage

Both approaches expose the same interface:

```bash
curl http://localhost:8080/health

curl -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{"description": "Get total sales per category for the last 30 days"}'
```

Run `make help` to see all available commands.

## Project Structure

```
├── shared/              # Shared config, models, prompts, evaluation
├── approach1_rag/       # RAG pipeline (FAISS + Ollama/vLLM)
├── approach2_finetune/  # QLoRA fine-tuning + Azure ML integration
├── data/samples/        # Training and evaluation JSONL
├── scripts/             # Helper scripts (generate predictions, etc.)
├── docs/                # Detailed documentation
└── notebooks/           # Interactive exploration notebooks
```

## Documentation

- [Local GPU Guide](docs/local-gpu-guide.md) — Run both approaches on RTX 4080 Super
- [Technology Glossary](docs/glossary.md) — Plain-language explanation of every tool and concept
- [Business Proposal](docs/proposal.md) — Cost estimates, timeline, approach comparison
- [Approach 2 Proposal](docs/approach2-proposal.md) — Fine-tuning proposal for management
- [Azure Infrastructure](docs/azure-infrastructure.md) — Resource setup guide
- [RAG Guide](docs/approach1-rag-guide.md) — Step-by-step RAG deployment
- [Fine-tuning Guide](docs/approach2-finetune-guide.md) — Training and deployment walkthrough
