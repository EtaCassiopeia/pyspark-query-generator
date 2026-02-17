.PHONY: help setup setup-rag setup-finetune setup-all \
       validate index \
       ollama-start ollama-stop rag-api \
       prepare-data train merge vllm-start finetune-api \
       evaluate gpu-check \
       docker-up docker-down docker-logs \
       clean

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
PYTHON     ?= python
PIP        ?= pip
SHELL      := /bin/bash

# ──────────────────────────────────────────────
# Help
# ──────────────────────────────────────────────
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ──────────────────────────────────────────────
# Environment setup
# ──────────────────────────────────────────────
setup-local: ## Copy local GPU config and install all deps
	@test -f .env || cp .env.local .env
	$(PIP) install -e ".[rag,finetune,eval]"

setup-rag: ## Install RAG dependencies only
	$(PIP) install -e ".[rag]"

setup-finetune: ## Install fine-tuning dependencies only
	$(PIP) install -e ".[finetune]"

setup-all: ## Install everything (RAG + finetune + eval + dev)
	$(PIP) install -e ".[rag,finetune,eval,dev]"

# ──────────────────────────────────────────────
# GPU diagnostics
# ──────────────────────────────────────────────
gpu-check: ## Show GPU info and VRAM
	@echo "=== NVIDIA Driver ==="
	nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free \
		--format=csv,noheader || echo "nvidia-smi not found"
	@echo ""
	@echo "=== PyTorch CUDA ==="
	$(PYTHON) -c "\
		import torch; \
		print(f'PyTorch: {torch.__version__}'); \
		print(f'CUDA available: {torch.cuda.is_available()}'); \
		print(f'CUDA version: {torch.version.cuda}'); \
		[print(f'GPU {i}: {torch.cuda.get_device_name(i)} — {torch.cuda.get_device_properties(i).total_mem / 1024**3:.1f} GB') for i in range(torch.cuda.device_count())] \
	" 2>/dev/null || echo "PyTorch not installed or no CUDA"

# ──────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────
validate: ## Validate training & evaluation JSONL
	$(PYTHON) -m shared.data_validator --input data/samples/training_data.jsonl
	$(PYTHON) -m shared.data_validator --input data/samples/evaluation_data.jsonl

# ──────────────────────────────────────────────
# Approach 1: RAG
# ──────────────────────────────────────────────
index: ## Build FAISS index from training data
	$(PYTHON) -m approach1_rag.ingestion

ollama-pull: ## Pull the DeepSeek model into Ollama
	ollama pull deepseek-coder:6.7b-instruct

ollama-start: ## Start Ollama server (foreground)
	ollama serve

rag-api: ## Start RAG FastAPI server
	$(PYTHON) -m approach1_rag.api

rag-test: ## Quick smoke test against RAG API
	@curl -s http://localhost:8080/health | $(PYTHON) -m json.tool
	@echo ""
	@curl -s -X POST http://localhost:8080/generate \
		-H "Content-Type: application/json" \
		-d '{"description": "Get total sales per product category for the last 30 days"}' \
		| $(PYTHON) -m json.tool

# ──────────────────────────────────────────────
# Approach 2: Fine-tuning
# ──────────────────────────────────────────────
prepare-data: ## Convert JSONL to ChatML instruction format
	$(PYTHON) -m approach2_finetune.prepare_data

train: ## Run QLoRA fine-tuning locally
	$(PYTHON) -m approach2_finetune.train_lora

merge: ## Merge LoRA adapter into base model
	$(PYTHON) -m approach2_finetune.merge_adapter

vllm-start: ## Start vLLM server with merged model
	$(PYTHON) -m approach2_finetune.serve_vllm --model outputs/merged_model

vllm-start-lora: ## Start vLLM with base model + LoRA adapter
	$(PYTHON) -m approach2_finetune.serve_vllm \
		--model deepseek-ai/deepseek-coder-6.7b-instruct \
		--lora-adapter outputs/lora_adapter

finetune-api: ## Start fine-tune FastAPI server
	$(PYTHON) -m approach2_finetune.api

finetune-test: ## Quick smoke test against fine-tune API
	@curl -s http://localhost:8081/health | $(PYTHON) -m json.tool
	@echo ""
	@curl -s -X POST http://localhost:8081/generate \
		-H "Content-Type: application/json" \
		-d '{"description": "Get total sales per product category for the last 30 days"}' \
		| $(PYTHON) -m json.tool

# ──────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────
evaluate: ## Run evaluation (baseline — reference as predictions)
	$(PYTHON) -m shared.evaluation.evaluator \
		--eval-data data/samples/evaluation_data.jsonl

evaluate-predictions: ## Run evaluation against a predictions file (PRED=path)
	$(PYTHON) -m shared.evaluation.evaluator \
		--eval-data data/samples/evaluation_data.jsonl \
		--predictions $(PRED)

# ──────────────────────────────────────────────
# Docker (local GPU)
# ──────────────────────────────────────────────
docker-up-rag: ## Start RAG stack (Ollama + RAG API) — needs GPU
	docker compose -f docker-compose.local.yml --profile rag up -d

docker-up-finetune: ## Start fine-tune stack (vLLM + fine-tune API) — needs GPU
	docker compose -f docker-compose.local.yml --profile finetune up -d

docker-train: ## Run training inside Docker (prepare → train → merge) — needs GPU
	docker compose -f docker-compose.local.yml --profile train run --rm trainer

docker-down: ## Stop all local containers
	docker compose -f docker-compose.local.yml --profile all --profile rag --profile finetune --profile train down

docker-logs: ## Tail logs from running services
	docker compose -f docker-compose.local.yml --profile all logs -f

docker-logs-rag: ## Tail RAG stack logs
	docker compose -f docker-compose.local.yml --profile rag logs -f

docker-logs-finetune: ## Tail fine-tune stack logs
	docker compose -f docker-compose.local.yml --profile finetune logs -f

# ──────────────────────────────────────────────
# Cleanup
# ──────────────────────────────────────────────
clean: ## Remove generated artifacts
	rm -rf outputs/ data/faiss_index/ data/samples/prepared/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
