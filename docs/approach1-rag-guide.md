# Approach 1: RAG + Local LLM — Step-by-Step Guide

## Architecture

```
User Request
    │
    ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│ Embed query  │────▶│ FAISS search │────▶│ Build prompt│
│ (MiniLM)     │     │ (top-k)      │     │ (few-shot)  │
└─────────────┘     └──────────────┘     └──────┬──────┘
                                                 │
                                                 ▼
                                         ┌──────────────┐
                                         │  LLM Generate │
                                         │  (Ollama/vLLM)│
                                         └──────┬───────┘
                                                 │
                                                 ▼
                                          PySpark Code
```

## Step 1: Prepare Training Data

Ensure your training examples are in `data/samples/training_data.jsonl`. Each line should have:

```json
{
  "description": "Natural language description",
  "pyspark_code": "from pyspark.sql import ...",
  "tables": ["schema.table"],
  "operations": ["filter", "groupBy"],
  "complexity": "medium"
}
```

Validate:

```bash
python -m shared.data_validator --input data/samples/training_data.jsonl
```

## Step 2: Build FAISS Index

```bash
python -m approach1_rag.ingestion
```

This will:
1. Load and validate the JSONL file
2. Embed all descriptions using `all-MiniLM-L6-v2` (384-dim)
3. Build a FAISS `IndexFlatIP` (cosine similarity on normalized vectors)
4. Save `index.faiss` and `metadata.pkl` to `data/faiss_index/`

## Step 3: Start the LLM Backend

### Option A: Ollama (recommended for development)

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull the model
ollama pull deepseek-coder:6.7b-instruct

# Ollama runs on port 11434 by default
```

### Option B: vLLM (recommended for production)

```bash
python -m approach2_finetune.serve_vllm \
  --model deepseek-ai/deepseek-coder-6.7b-instruct
```

## Step 4: Start the RAG API

```bash
python -m approach1_rag.api
```

The API runs on `http://localhost:8080` with endpoints:
- `GET /health` — Check API and LLM status
- `POST /generate` — Generate PySpark code
- `GET /docs` — Interactive Swagger UI

## Step 5: Docker Compose (Production)

```bash
cd approach1_rag
docker-compose up -d
```

This starts both Ollama and the RAG API in containers.

## Step 6: Test

```bash
# Health check
curl http://localhost:8080/health

# Generate
curl -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{
    "description": "Calculate average salary per department for active employees"
  }'
```

## Adding New Examples

1. Add examples to `data/samples/training_data.jsonl`
2. Re-run ingestion: `python -m approach1_rag.ingestion`
3. Restart the API (or implement hot-reload)

## Tuning Parameters

| Parameter | Default | Description |
|---|---|---|
| `RAG_TOP_K` | 3 | Number of examples to retrieve |
| `LLM_TEMPERATURE` | 0.1 | Lower = more deterministic |
| `LLM_MAX_TOKENS` | 2048 | Maximum output length |
| `EMBEDDING_MODEL` | all-MiniLM-L6-v2 | Sentence transformer model |
