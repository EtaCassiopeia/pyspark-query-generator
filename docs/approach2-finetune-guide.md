# Approach 2: Fine-tuning with LoRA — Step-by-Step Guide

## Architecture

```
Training Pipeline:
  JSONL Data → Prepare (ChatML) → QLoRA Training → LoRA Adapter → Merge (optional)

Serving Pipeline:
  User Request → vLLM Server (base + LoRA) → PySpark Code
```

## Prerequisites

- NVIDIA GPU with 16GB+ VRAM (local) or Azure ML GPU cluster
- CUDA 12.1+
- ~30GB disk space for model weights

## Step 1: Prepare Training Data

Convert the JSONL examples to ChatML conversation format:

```bash
python -m approach2_finetune.prepare_data
```

This creates `data/samples/prepared/train.jsonl` and `val.jsonl` in the format:

```json
{
  "messages": [
    {"role": "system", "content": "You are a PySpark code generator..."},
    {"role": "user", "content": "Get total sales per category..."},
    {"role": "assistant", "content": "from pyspark.sql import..."}
  ]
}
```

## Step 2: Train (Local)

```bash
python -m approach2_finetune.train_lora \
  --output-dir outputs/lora_adapter
```

### Training Configuration

| Parameter | Default | Notes |
|---|---|---|
| Base model | deepseek-coder-6.7b-instruct | Open-weight, code-specialized |
| LoRA rank (r) | 16 | Higher = more parameters, better fit |
| LoRA alpha | 32 | Scaling factor (typically 2x rank) |
| Quantization | 4-bit NF4 (QLoRA) | Reduces memory from ~26GB to ~6GB |
| Epochs | 3 | Monitor eval loss for overfitting |
| Learning rate | 2e-4 | With cosine scheduler |
| Batch size | 4 | Effective batch = 4 * 4 (grad accum) = 16 |

### Expected Training Time

| Hardware | Time |
|---|---|
| A100 80GB | ~15 min |
| A10G 24GB | ~30 min |
| T4 16GB | ~60 min |
| RTX 4090 24GB | ~20 min |

## Step 2 (Alt): Train on Azure ML

```bash
# Setup workspace and compute (one-time)
python -m approach2_finetune.azure_ml.workspace_setup
python -m approach2_finetune.azure_ml.create_compute

# Submit training job
python -m approach2_finetune.azure_ml.submit_training_job
```

Monitor in Azure ML Studio. The job will:
1. Prepare training data
2. Run QLoRA training
3. Save the LoRA adapter to the job outputs

## Step 3: Merge Adapter (Optional)

For simpler deployment, merge the LoRA weights into the base model:

```bash
python -m approach2_finetune.merge_adapter \
  --adapter-path outputs/lora_adapter \
  --output-path outputs/merged_model
```

## Step 4: Serve with vLLM

### Option A: LoRA adapter (lower disk usage)

```bash
python -m approach2_finetune.serve_vllm \
  --model deepseek-ai/deepseek-coder-6.7b-instruct \
  --lora-adapter outputs/lora_adapter
```

### Option B: Merged model (simpler)

```bash
python -m approach2_finetune.serve_vllm \
  --model outputs/merged_model
```

## Step 5: Start the API

```bash
python -m approach2_finetune.api
```

Same interface as Approach 1 on `http://localhost:8081`.

## Step 6: Deploy on Azure ML

```bash
# Register model
python -m approach2_finetune.azure_ml.register_model \
  --model-path outputs/lora_adapter

# Deploy managed endpoint
python -m approach2_finetune.azure_ml.deploy_endpoint
```

## Evaluation

Run the evaluation suite against predictions from the fine-tuned model:

```bash
# Generate predictions for eval set (manual or via API)
# Then evaluate:
python -m shared.evaluation.evaluator \
  --eval-data data/samples/evaluation_data.jsonl \
  --predictions outputs/predictions.jsonl
```

## Retraining

When new examples are available:

1. Add examples to `data/samples/training_data.jsonl`
2. Re-run `prepare_data.py`
3. Train a new adapter (previous adapter can serve as starting point)
4. Evaluate on held-out set
5. If improved, register new model version and update endpoint
