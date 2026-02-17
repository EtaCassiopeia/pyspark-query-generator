# Local GPU Deployment Guide — RTX 4080 Super

Run both approaches (RAG and fine-tuning) entirely on your local machine with an NVIDIA RTX 4080 Super (16 GB VRAM).

## Prerequisites

| Requirement | Version | Check command |
|---|---|---|
| NVIDIA Driver | 535+ | `nvidia-smi` |
| CUDA Toolkit | 12.1+ | `nvcc --version` |
| Python | 3.10+ | `python --version` |
| Docker + NVIDIA Container Toolkit | Latest | `docker run --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi` |
| Ollama | Latest | `ollama --version` |
| ~40 GB disk space | For model weights + cache | |

## Quick Reference

```bash
make help               # See all available commands
make gpu-check          # Verify GPU is detected
make setup-local        # One-time setup
make validate           # Check training data

# --- Approach 1: RAG ---
make index              # Build FAISS index
make ollama-pull        # Download model into Ollama
# Terminal 1:
make ollama-start       # Start Ollama server
# Terminal 2:
make rag-api            # Start RAG API on :8080
# Terminal 3:
make rag-test           # Smoke test

# --- Approach 2: Fine-tuning ---
make prepare-data       # Convert to ChatML
make train              # QLoRA training (~30 min)
make merge              # Merge adapter into standalone model
# Terminal 1:
make vllm-start         # Start vLLM on :8000
# Terminal 2:
make finetune-api       # Start fine-tune API on :8081
# Terminal 3:
make finetune-test      # Smoke test
```

---

## Step 0: Verify Your GPU

```bash
make gpu-check
```

Expected output:

```
=== NVIDIA Driver ===
NVIDIA GeForce RTX 4080 SUPER, 560.xx, 16384 MiB, 16000 MiB

=== PyTorch CUDA ===
PyTorch: 2.x.x
CUDA available: True
CUDA version: 12.1
GPU 0: NVIDIA GeForce RTX 4080 SUPER — 16.0 GB
```

If `CUDA available: False`, check that your NVIDIA driver and CUDA toolkit are installed correctly.

## Step 1: Environment Setup

```bash
cd pyspark-query-generator

# Copy the local GPU config (tuned for 16 GB VRAM)
cp .env.local .env

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install all dependencies
make setup-local
```

### What `.env.local` changes vs the defaults

| Setting | Default (Azure) | Local (RTX 4080 Super) | Why |
|---|---|---|---|
| `BATCH_SIZE` | 4 | 2 | Fits in 16 GB VRAM |
| `GRADIENT_ACCUMULATION_STEPS` | 4 | 8 | Compensates for smaller batch |
| `VLLM_GPU_MEMORY_UTILIZATION` | 0.9 | 0.85 | Leaves headroom for OS/display |
| `VLLM_QUANTIZATION` | (empty) | awq | Enables 4-bit serving for 16 GB |

The training script also **auto-detects your GPU** and adjusts batch size/sequence length at runtime, so these .env values are safety nets.

---

## Approach 1: RAG (Local)

RAG doesn't require training. It retrieves similar examples from your dataset and includes them as context when prompting the LLM.

### 1a. Build the FAISS Index

```bash
make validate   # Verify data files are valid
make index      # Embed descriptions → build FAISS index
```

This downloads the `all-MiniLM-L6-v2` embedding model (~22 MB, runs on CPU) and creates `data/faiss_index/`.

### 1b. Start Ollama

```bash
# Pull the model (one-time, ~4 GB download)
make ollama-pull

# Start the server (keep this terminal open)
make ollama-start
```

Ollama quantizes the model automatically (typically Q4_K_M) so it fits comfortably in 16 GB VRAM with room to spare.

### 1c. Start the RAG API

In a second terminal:

```bash
make rag-api
```

The API starts on `http://localhost:8080`. Swagger docs at `http://localhost:8080/docs`.

### 1d. Test

```bash
# Health check + generation test
make rag-test
```

Or manually:

```bash
curl -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{
    "description": "Calculate average salary per department for active employees"
  }'
```

### 1e. Docker Alternative

If you prefer containers:

```bash
make docker-up-rag     # Starts Ollama + RAG API containers
make docker-logs-rag   # Watch logs
make docker-down       # Stop everything
```

### VRAM Usage (Approach 1)

| Component | VRAM |
|---|---|
| Ollama (DeepSeek 6.7B Q4) | ~5 GB |
| Embedding model | CPU only |
| **Total** | **~5 GB** |

Plenty of headroom on 16 GB.

---

## Approach 2: Fine-Tuning (Local)

Fine-tuning customizes the model on your PySpark examples. Three phases: train, merge, serve.

### 2a. Prepare Training Data

```bash
make prepare-data
```

Creates `data/samples/prepared/train.jsonl` and `val.jsonl` in ChatML conversation format.

### 2b. Train with QLoRA

**Stop Ollama first** to free GPU memory:

```bash
# In the terminal running Ollama, press Ctrl+C
# Or:
pkill ollama
```

Then train:

```bash
make train
```

The training script auto-detects your RTX 4080 Super and selects:

```
Detected GPU: NVIDIA GeForce RTX 4080 SUPER (16.0 GB VRAM)
Selected profile: T4/RTX 4080 (batch_size=1, grad_accum=16, max_seq_len=1536)
Effective batch size: 1 x 16 = 16
```

**Expected training time: ~30 minutes** for 3 epochs on 17 examples.

#### What happens during training

1. Downloads DeepSeek Coder 6.7B from HuggingFace (~13 GB, cached for reuse)
2. Loads it in 4-bit quantization (QLoRA) — fits in ~6 GB VRAM
3. Attaches small LoRA adapter layers (~4M trainable parameters out of 6.7B)
4. Trains for 3 passes over the dataset
5. Saves the adapter to `outputs/lora_adapter/` (~50-100 MB)

#### VRAM usage during training

| Component | VRAM |
|---|---|
| Base model (4-bit quantized) | ~4 GB |
| LoRA adapter + optimizer states | ~2 GB |
| Activations (with gradient checkpointing) | ~4-6 GB |
| **Total** | **~10-12 GB** |

Fits in 16 GB with margin.

#### If you get CUDA out-of-memory

Force even smaller settings:

```bash
python -m approach2_finetune.train_lora --no-auto-gpu
```

Then edit `.env` to use `BATCH_SIZE=1` and `GRADIENT_ACCUMULATION_STEPS=16`.

### 2c. Merge the Adapter

```bash
make merge
```

This combines the base model weights with your LoRA adapter into a single standalone model at `outputs/merged_model/`. Takes a few minutes and ~26 GB of disk (16-bit weights). The merge runs on CPU, so VRAM is not a constraint.

### 2d. Serve with vLLM

```bash
make vllm-start
```

vLLM loads the merged model and starts an OpenAI-compatible API on port 8000. With `VLLM_QUANTIZATION=awq` in your `.env`, it quantizes the model on-the-fly to fit in 16 GB.

**Alternative — serve base model + LoRA adapter directly (no merge needed):**

```bash
make vllm-start-lora
```

### 2e. Start the Fine-Tune API

In a second terminal:

```bash
make finetune-api
```

Runs on `http://localhost:8081` with the same API as Approach 1.

### 2f. Test

```bash
make finetune-test
```

### 2g. Docker Alternative

```bash
# Train inside Docker (prepare → train → merge)
make docker-train

# Start inference stack
make docker-up-finetune
make docker-logs-finetune
```

### VRAM Usage (Approach 2 — Inference)

| Component | VRAM |
|---|---|
| vLLM (DeepSeek 6.7B AWQ quantized) | ~6 GB |
| KV cache (for generation) | ~4-6 GB |
| **Total** | **~10-12 GB** |

---

## Running Both Approaches Side by Side

You **cannot** run Ollama and vLLM at the same time on a single 16 GB GPU — both need the GPU. To compare:

### Option A: Switch manually

```bash
# Test Approach 1
make ollama-start   # Terminal 1
make rag-api        # Terminal 2
make rag-test       # Terminal 3
# Ctrl+C to stop Ollama and RAG API

# Test Approach 2
make vllm-start     # Terminal 1
make finetune-api   # Terminal 2
make finetune-test  # Terminal 3
```

### Option B: Run RAG without GPU

Ollama can run in CPU-only mode (slower but works). This lets you run both APIs simultaneously:

```bash
# Terminal 1 — vLLM on GPU
make vllm-start

# Terminal 2 — Ollama on CPU
CUDA_VISIBLE_DEVICES="" ollama serve

# Terminal 3 — RAG API
make rag-api

# Terminal 4 — Fine-tune API
make finetune-api
```

RAG inference on CPU takes ~30-60s per request vs ~2-5s on GPU. Acceptable for testing.

---

## Evaluate Both Approaches

After running each approach, save predictions and compare:

```bash
# Generate predictions from RAG (with RAG API running on :8080)
python scripts/generate_predictions.py --api http://localhost:8080 --output outputs/rag_predictions.jsonl

# Generate predictions from fine-tuned model (with fine-tune API running on :8081)
python scripts/generate_predictions.py --api http://localhost:8081 --output outputs/ft_predictions.jsonl

# Compare
make evaluate-predictions PRED=outputs/rag_predictions.jsonl
make evaluate-predictions PRED=outputs/ft_predictions.jsonl
```

---

## Troubleshooting

### `CUDA out of memory` during training

- Ensure Ollama and any other GPU processes are stopped: `nvidia-smi` to check
- The auto-detect should handle this, but you can force smaller settings with `--no-auto-gpu` and `.env` overrides
- Reduce `max_seq_length` in `.env` to 1024

### `torch.cuda.OutOfMemoryError` during vLLM serving

- Reduce `VLLM_GPU_MEMORY_UTILIZATION` to `0.80` in `.env`
- Reduce `VLLM_MAX_MODEL_LEN` to `2048`
- Ensure no other process uses the GPU

### Ollama model download is slow

Ollama models download from ollama.ai. If on a restricted network, you can pre-download and import:

```bash
# Download the GGUF file manually, then:
ollama create deepseek-coder:6.7b-instruct -f Modelfile
```

### vLLM fails to start with AWQ quantization

Not all models ship with pre-quantized AWQ weights. If this happens:

```bash
# Option 1: Use the merged model without quantization (needs careful VRAM management)
VLLM_QUANTIZATION="" make vllm-start

# Option 2: Serve with Ollama instead of vLLM (it handles quantization internally)
ollama create pyspark-finetuned -f outputs/merged_model/Modelfile
ollama run pyspark-finetuned
```

### Training loss is not decreasing

- Ensure `prepare-data` ran successfully and `data/samples/prepared/train.jsonl` is not empty
- Try lowering the learning rate: set `LEARNING_RATE=1e-4` in `.env`
- Check that the training data is valid: `make validate`
