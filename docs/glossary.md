# Technology Glossary

A plain-language guide to every technology, tool, and concept used in this project. Organized from foundational ideas up through specific tools, so you can read top-to-bottom or jump to any term.

---

## Table of Contents

1. [Core AI / Machine Learning Concepts](#1-core-ai--machine-learning-concepts)
2. [How LLMs Work Under the Hood](#2-how-llms-work-under-the-hood)
3. [Fine-Tuning Concepts](#3-fine-tuning-concepts)
4. [RAG (Retrieval-Augmented Generation)](#4-rag-retrieval-augmented-generation)
5. [Evaluation Metrics](#5-evaluation-metrics)
6. [Python Libraries & Tools](#6-python-libraries--tools)
7. [Model Serving & APIs](#7-model-serving--apis)
8. [Containerization](#8-containerization)
9. [Azure Infrastructure](#9-azure-infrastructure)
10. [Data Formats](#10-data-formats)
11. [Hardware](#11-hardware)
12. [How It All Connects](#12-how-it-all-connects)

---

## 1. Core AI / Machine Learning Concepts

### LLM (Large Language Model)

A computer program trained on massive amounts of text that can understand and generate human language. Think of it as an extremely sophisticated autocomplete — you give it some text, and it predicts what should come next. ChatGPT, Claude, and Llama are all LLMs.

In our project, we use an LLM to generate PySpark code from English descriptions.

### Base Model

The pre-trained LLM before we customize it. Our base model is **DeepSeek Coder 6.7B Instruct** — an open-source model built specifically for writing code. "6.7B" means it has 6.7 billion parameters (see below). "Instruct" means it was already trained to follow instructions, not just predict the next word.

### Open-Weight Model

A model whose internal data (the "weights") is publicly available for anyone to download, run, and modify. This is critical for our project because it means we can run the model entirely on our own servers — no data ever leaves our network. This is in contrast to closed models like GPT-4 which can only be accessed through an external API.

### Parameters

The numbers inside an LLM that were learned during training. They determine how the model responds to any given input. A 6.7B parameter model has 6.7 billion of these numbers. More parameters generally means more capable but also requires more memory and computing power.

### Tokens

How LLMs read text. Rather than processing letter-by-letter or word-by-word, LLMs break text into **tokens** — chunks that are roughly 3-4 characters on average. The word "PySpark" might be two tokens: "Py" and "Spark". The sentence "Get total sales" might be 3 tokens. Token count matters because:

- LLMs have a maximum number of tokens they can process at once (the **context window**)
- Generation speed is measured in tokens per second
- Cloud API pricing is typically per-token

### Inference

Using a trained model to produce an output. When a user asks "Get total sales per region" and the model generates PySpark code, that's inference. This is the day-to-day "running" of the model, as opposed to training it.

### Training vs. Inference

| | Training | Inference |
|---|---|---|
| **What** | Teaching the model from examples | Using the model to generate output |
| **When** | Once (or occasionally to update) | Every time a user makes a request |
| **Compute** | Very heavy (big GPU, hours) | Lighter (smaller GPU, seconds) |
| **Analogy** | Studying for an exam | Taking the exam |

### Prompt

The text you send to an LLM to get a response. In our project, a prompt includes:
- A **system prompt**: background instructions ("You are a PySpark code generator...")
- A **user prompt**: the actual request ("Get total sales per region for Q4")

Prompt engineering is the practice of crafting these inputs to get better outputs.

### Few-Shot Learning

Showing the model a few examples of what you want before asking it to do the task. Instead of just saying "generate PySpark code for X," you show it 3 examples of descriptions and their correct PySpark code, then ask it to handle a new one. This dramatically improves output quality without any training. The RAG approach in this project is built on this idea.

### Hallucination

When an LLM confidently generates something that's wrong — like referencing a table that doesn't exist or using a function with incorrect syntax. This is a key risk in code generation, which is why we include human review as a step.

---

## 2. How LLMs Work Under the Hood

### Tokenizer

The component that converts raw text into tokens (numbers) that the model can process, and converts the model's output tokens back into text. Each model has its own tokenizer. When our code calls `AutoTokenizer.from_pretrained(...)`, it downloads the tokenizer that matches our base model.

### Model Weights

The actual file(s) containing the billions of parameters. For a 6.7B model, the weights are typically 13-26 GB on disk depending on the number format used. Common file formats:

- `.safetensors` — the modern standard (safe to load, efficient)
- `.bin` — older PyTorch format
- `.gguf` — compressed format used by Ollama

### Causal Language Model (CausalLM)

The type of LLM architecture used for text generation. "Causal" means the model can only look at text that comes before the current position — it generates text left-to-right, one token at a time. This is what `AutoModelForCausalLM` in our code refers to. Almost all code-generation models use this architecture.

---

## 3. Fine-Tuning Concepts

### Fine-Tuning

Taking a pre-trained model and training it further on your own data so it specializes in your task. The model already knows how to write code in general; fine-tuning teaches it specifically how *we* write PySpark queries, with our tables, our naming conventions, and our patterns.

Think of it like hiring an experienced developer and onboarding them — they already know Python, but they need to learn your codebase.

### Full Fine-Tuning vs. Parameter-Efficient Fine-Tuning

**Full fine-tuning** updates all 6.7 billion parameters. This requires enormous GPU memory (50+ GB) and risks "catastrophic forgetting" — the model loses its general abilities while learning the new task.

**Parameter-efficient fine-tuning (PEFT)** only updates a tiny fraction of parameters (typically <1%), keeping the rest frozen. This requires far less memory, is much faster, and preserves the model's general knowledge. Our project uses PEFT.

### LoRA (Low-Rank Adaptation)

The specific PEFT technique we use. Instead of modifying the model's existing parameters, LoRA inserts small trainable matrices alongside certain layers of the model. Here's the intuition:

```
Original model layer:    Input → [Large Matrix (frozen)] → Output
                                      6.7B params

With LoRA:               Input → [Large Matrix (frozen)] → Output
                                         +
                                  [Small Matrix A] × [Small Matrix B]
                                      ~4M params (trainable)
```

The small matrices (the "adapter") learn the difference between generic code generation and our specific PySpark patterns. Key settings:

- **Rank (r=16)**: Size of the small matrices. Higher rank = more capacity to learn, but more memory. 16 is a good balance.
- **Alpha (=32)**: A scaling factor that controls how much influence the adapter has. Typically set to 2x the rank.
- **Dropout (=0.05)**: Randomly deactivates 5% of connections during training to prevent overfitting (see below).
- **Target modules** (`q_proj`, `k_proj`, `v_proj`, etc.): Which layers of the model get LoRA adapters. These are the "attention" layers where the model decides what parts of the input to focus on.

### QLoRA (Quantized LoRA)

LoRA with an extra trick: the frozen base model is compressed to 4-bit precision (see Quantization below) before adding the LoRA adapters. This slashes memory usage from ~26 GB to ~6 GB, letting you fine-tune a 6.7B model on a single consumer GPU. The trainable LoRA matrices remain in higher precision so training quality isn't affected.

### Quantization

Compressing model weights by using fewer bits per number. A standard model uses 16 bits per parameter (2 bytes). Quantization reduces this:

| Precision | Bits per parameter | 6.7B model size | Quality impact |
|---|---|---|---|
| fp16 / bf16 | 16 bits | ~13 GB | None (original) |
| int8 | 8 bits | ~7 GB | Minimal |
| int4 / NF4 | 4 bits | ~4 GB | Small, acceptable |

**NF4 (Normal Float 4-bit)** is a 4-bit format specifically designed for LLM weights. It's what QLoRA uses. "Double quantization" further compresses the metadata that describes the quantization itself.

### Adapter

The small set of LoRA weights that get saved after training. While the base model is 13+ GB, the adapter is only ~30-100 MB. To use the fine-tuned model, you load the base model + adapter together. You can also **merge** the adapter into the base model to create a single standalone model file.

### Epoch

One complete pass through the entire training dataset. If you have 100 training examples and train for 3 epochs, the model sees each example 3 times. More epochs means more learning, but too many can cause overfitting.

### Overfitting

When the model memorizes the training examples instead of learning general patterns. An overfitted model produces perfect results for inputs it's seen before but fails on new ones. We prevent this with:

- A validation set (data held out from training) to monitor quality
- Dropout (randomly disabling connections during training)
- Early stopping (stopping training when validation quality stops improving)

### Batch Size & Gradient Accumulation

Training processes examples in groups called **batches**. Our batch size is 4, meaning the model sees 4 examples at once before updating its weights.

**Gradient accumulation** (=4 in our config) simulates larger batches without needing more memory. Instead of processing 16 examples at once (impossible on a single GPU), we process 4 batches of 4 and accumulate the learning signal before updating. Effective batch size = 4 x 4 = 16.

### Learning Rate

How big a step the model takes when updating its weights after each batch. Too high and the model overshoots and never settles; too low and training is too slow or gets stuck. Our learning rate of `2e-4` (0.0002) is standard for LoRA fine-tuning.

### Learning Rate Scheduler (Cosine)

The learning rate doesn't stay fixed — it follows a schedule. A **cosine scheduler** starts at the full learning rate, then gradually decreases it following a cosine curve. This lets the model make big adjustments early in training and fine-grained adjustments later.

### Warmup

The first portion of training (10% in our case, set by `warmup_ratio=0.1`) where the learning rate gradually increases from near-zero to its full value. This prevents the model from making wild updates when it first starts learning.

### Gradient Checkpointing

A memory optimization. During training, the model normally stores intermediate calculations for every layer to use when computing updates. Gradient checkpointing discards some of these and recomputes them when needed, trading a small amount of speed for significantly less memory usage.

### Eval Loss

The model's prediction error measured on the validation set (data it hasn't trained on). Lower is better. We monitor this to detect overfitting and to select the best model checkpoint.

### ChatML Format

A specific way to structure training conversations so the model learns to behave as a chat assistant:

```
<|im_start|>system
You are a PySpark code generator...<|im_end|>
<|im_start|>user
Get total sales per region<|im_end|>
<|im_start|>assistant
from pyspark.sql import functions as F
...<|im_end|>
```

The `<|im_start|>` and `<|im_end|>` tags are special tokens that tell the model where each speaker's turn begins and ends. DeepSeek Coder uses this format natively.

### Alpaca Format

An alternative instruction format (from Stanford's Alpaca project):

```
### Instruction:
Generate PySpark code for: Get total sales per region

### Response:
from pyspark.sql import functions as F
...
```

Simpler than ChatML but doesn't support multi-turn conversations. We include it as an option.

### Instruction Tuning

The general technique of fine-tuning a model on instruction-response pairs (as opposed to raw text). This is what makes a model actually follow user requests instead of just predicting the next word. Both ChatML and Alpaca are formats for instruction tuning.

---

## 4. RAG (Retrieval-Augmented Generation)

### RAG

A technique that enhances LLM generation by first searching a knowledge base for relevant information, then including that information in the prompt. In our project:

1. User asks: "Get total sales per region"
2. System **retrieves** the 3 most similar examples from our training data
3. System builds a prompt with those examples as context
4. LLM **generates** PySpark code, guided by the relevant examples

RAG doesn't modify the model — it improves results by giving the model better context.

### Embedding

A way to represent text as a list of numbers (a "vector") that captures its meaning. Similar texts produce similar vectors. For example, "total revenue by region" and "sum of sales per area" would have very similar embeddings, even though they share few words.

Our embedding model (`all-MiniLM-L6-v2`) converts each description into a vector of 384 numbers.

### Embedding Model (all-MiniLM-L6-v2)

A small, fast model designed specifically for creating embeddings. It's separate from the LLM — its only job is to convert text into numerical vectors for similarity search. At 22 MB, it runs on CPU with no GPU needed.

### Vector Store / Vector Database

A database optimized for storing and searching through embeddings. Instead of searching by keywords (like a traditional database), you search by meaning: "find the 3 stored descriptions most similar to this new description."

### FAISS (Facebook AI Similarity Search)

The specific vector store library we use. Built by Meta, it's a file-based library (not a separate server to run) that efficiently searches through millions of vectors. We use `IndexFlatIP` which computes exact cosine similarity — no approximation, 100% accurate. For our dataset size (<1000 examples), this is fast enough.

### Cosine Similarity

A mathematical measure of how similar two vectors are, regardless of their length. A score of 1.0 means identical direction (very similar meaning), 0.0 means unrelated, and -1.0 means opposite. Our FAISS index uses this to rank which training examples are most relevant to a user's query.

### Top-k Retrieval

Selecting the `k` most similar results from the vector search. In our config, `k=3` means we retrieve the 3 most similar training examples to use as context in the prompt.

---

## 5. Evaluation Metrics

### BLEU Score

Originally created for evaluating language translations, BLEU measures how much of the generated text overlaps with the reference text at the word/phrase level. Score ranges from 0 to 1 — higher is better. For code, BLEU captures whether the generated code uses the same functions, variables, and structure as the reference.

### Operation Coverage

A custom metric for this project. It checks: of all the PySpark operations used in the reference code (e.g., `filter`, `groupBy`, `join`, `sum`), what fraction also appears in the generated code? A score of 1.0 means the generated code uses all the same operations. This is more meaningful than BLEU for PySpark specifically.

### Structural Similarity

Another custom metric. It extracts "structural patterns" from both the reference and generated code — method calls, table names, column names, function calls — and measures the overlap (Jaccard similarity). This tells us whether the generated code has the right overall shape, even if the exact syntax differs.

### Exact Match

Simply: does the generated code match the reference exactly (after normalizing whitespace)? This is a strict metric. Scoring 100% here isn't realistic, but it provides a useful upper-bound reference.

---

## 6. Python Libraries & Tools

### PyTorch

The deep learning framework that everything runs on. PyTorch handles the actual math of running and training neural networks on GPUs. Think of it as the "engine" — all the other ML libraries in this project are built on top of it.

### HuggingFace Transformers

The standard library for working with LLMs in Python. It provides:
- `AutoModelForCausalLM` — load any text generation model with one line of code
- `AutoTokenizer` — load the matching tokenizer
- `TrainingArguments` — configure training (learning rate, epochs, batch size, etc.)

It supports thousands of open-source models and is the de facto standard in the ML community.

### PEFT (Parameter-Efficient Fine-Tuning)

A HuggingFace library specifically for LoRA and other parameter-efficient methods. It provides:
- `LoraConfig` — define the LoRA adapter configuration (rank, alpha, which layers)
- `get_peft_model` — attach LoRA adapters to a model
- `PeftModel` — load a base model with a saved adapter

### TRL (Transformer Reinforcement Learning)

A HuggingFace library for training LLMs with various techniques. We use its `SFTTrainer` (Supervised Fine-Tuning Trainer), which handles:
- Loading and formatting conversation data
- Running the training loop
- Saving checkpoints
- Evaluating on validation data

It wraps all the complexity of fine-tuning into a single, well-tested class.

### BitsAndBytes

The library that enables 4-bit and 8-bit quantization. When we call `BitsAndBytesConfig(load_in_4bit=True, ...)`, this library compresses the model weights from 16 bits to 4 bits on the fly, reducing memory by ~4x. This is what makes QLoRA possible on affordable hardware.

### Accelerate

A HuggingFace library that handles distributing model loading across available hardware. When you see `device_map="auto"`, Accelerate figures out how to split the model across available GPUs (or CPU + GPU if the model is too large for a single GPU).

### sentence-transformers

The library for creating text embeddings. Wraps transformer models that have been specially trained for similarity tasks. We use it to embed descriptions before storing them in FAISS.

### Datasets (HuggingFace)

A library for loading and preprocessing training data. Our code uses `load_dataset("json", data_files=...)` to load the JSONL files into a format that the trainer expects.

### Pydantic

A Python library for data validation. We define the shape of our data as Python classes (e.g., `TrainingExample` must have a `description` string and a `pyspark_code` string), and Pydantic automatically validates incoming data against these rules. If someone provides a training example without a description, Pydantic raises a clear error.

### pydantic-settings

An extension of Pydantic for application configuration. It reads settings from `.env` files and environment variables, validates them, and provides typed access. This is how `shared/config.py` works.

### FastAPI

A modern Python web framework for building APIs. We use it to create the HTTP endpoints (`/generate`, `/health`) that users and applications call to generate PySpark code. Features we rely on:
- Automatic input validation (via Pydantic)
- Auto-generated API docs (Swagger UI at `/docs`)
- Async support for handling multiple concurrent requests

### uvicorn

The web server that runs FastAPI applications. FastAPI defines what the API does; uvicorn handles the networking (listening for HTTP requests, managing connections). The command `uvicorn app:app --port 8080` starts the server on port 8080.

### httpx

An HTTP client library for Python (similar to `requests` but with async support). Our RAG approach uses it to call the LLM backend's API. The LLM client sends a POST request to the Ollama/vLLM `/chat/completions` endpoint and receives the generated code.

### NLTK

Natural Language Toolkit — a long-established Python library for text processing. We use it solely for BLEU score calculation in our evaluation metrics.

### pyproject.toml

The standard Python project configuration file. It defines:
- Project name, version, and description
- Dependencies (what libraries the project needs)
- Optional dependency groups (`[rag]`, `[finetune]`, `[azure]`, `[eval]`) so you only install what you need
- Tool configurations (linter settings, type checker settings)

---

## 7. Model Serving & APIs

### Ollama

A desktop application that makes it easy to download and run open-source LLMs locally. Think of it as "Docker for LLMs" — you run `ollama pull deepseek-coder:6.7b-instruct` and it downloads and serves the model. It exposes an API compatible with OpenAI's format, so our code can talk to it the same way it would talk to any other LLM API.

We use Ollama for **development and the RAG approach** because it's easy to set up.

### vLLM

A high-performance LLM serving engine. While Ollama is optimized for ease of use, vLLM is optimized for production workloads:
- Higher throughput (more requests per second)
- Native LoRA adapter support (can serve the base model + adapter without merging)
- Efficient GPU memory management
- OpenAI-compatible API

We use vLLM for **production deployment of the fine-tuned model**.

### OpenAI-Compatible API

A standardized API format that many LLM tools now support. The key endpoint is `/v1/chat/completions`, which accepts messages in the format `[{"role": "system", ...}, {"role": "user", ...}]` and returns generated text. Both Ollama and vLLM implement this format, so our code works with either backend without changes.

### Managed Online Endpoint (Azure ML)

An Azure service that hosts ML models behind an HTTPS API without you managing the underlying servers. You provide the model and a serving script; Azure handles scaling, load balancing, health monitoring, and networking. We use this for the production deployment of Approach 2.

---

## 8. Containerization

### Docker

A tool that packages an application and all its dependencies into a portable unit called a **container**. A container includes the operating system, Python, all libraries, and your code — guaranteeing it runs the same way everywhere. Our `Dockerfile` describes how to build this package.

### Docker Compose

A tool for running multiple Docker containers together. Our `docker-compose.yml` defines two services:
- **ollama**: runs the LLM
- **rag-api**: runs our FastAPI application

Docker Compose handles networking between them (the API can reach Ollama at `http://ollama:11434`) and starts them in the right order.

### Container Registry (Azure ACR)

A storage service for Docker container images. After building a Docker image locally, you push it to the registry so Azure services can pull and run it. Think of it as "GitHub for Docker images."

---

## 9. Azure Infrastructure

### VNet (Virtual Network)

A logically isolated network in Azure. All resources inside a VNet can communicate with each other, but nothing outside can reach them unless you explicitly allow it. This is how we ensure **no data leaves our infrastructure** — the LLM, the API, and all training data live inside the VNet.

Think of it as a private office building with a single locked entrance.

### Subnet

A subdivision of a VNet. We create separate subnets for different purposes:
- **Compute subnet** (10.0.1.0/24): VMs and containers that run our application
- **ML subnet** (10.0.2.0/24): Azure ML training clusters
- **Endpoints subnet** (10.0.3.0/24): Private access points for Azure services

Subnets let you apply different security rules to different groups of resources.

### Private Endpoint

A way to access Azure services (storage, ML workspace, container registry) through a private IP address inside your VNet, instead of going over the public internet. Even though Azure Storage has a public URL, our private endpoint means traffic from our VMs to storage never leaves the Azure backbone network.

### NSG (Network Security Group)

A firewall rule set attached to a subnet or individual VM. NSGs control which network traffic is allowed in and out. For example: "only the compute subnet can talk to the ML subnet on port 443."

### Resource Group

A logical container for Azure resources. All resources for this project (VMs, storage, networking, ML workspace) go in one resource group called `rg-pyspark-generator`. This makes it easy to manage permissions, track costs, and clean up when done.

### Azure ML Workspace

The central hub for all machine learning activities in Azure. It contains:
- **Compute clusters** — the machines that run training jobs
- **Model registry** — versioned storage for trained models
- **Managed endpoints** — hosted APIs for model serving
- **Experiment tracking** — logs and metrics from training runs

You can manage everything through a web UI called **ML Studio** or through Python code.

### Compute Cluster

A pool of VMs in Azure ML that runs training jobs. Our cluster:
- Uses **low-priority (spot) instances** — spare Azure capacity available at ~60% discount, with the tradeoff that Azure can reclaim them if demand spikes
- **Scales to zero** — when no training job is running, the cluster shuts down completely and costs nothing
- Scales up to 1 node when a job is submitted

### Model Registry

A versioned store for trained models within Azure ML. After training, we register the LoRA adapter as version 1 of "pyspark-query-generator." If we retrain later, it becomes version 2. This provides a clear history and makes it easy to roll back.

### Key Vault

Azure's secrets management service. API keys, passwords, and connection strings go here instead of in config files or environment variables. Applications retrieve secrets at runtime through a secure API.

### Storage Account

Azure's general-purpose storage. We use it for training data files and model artifacts. Storage Account types:
- **Blob storage** — files of any kind (our JSONL data, model weights)
- **Standard LRS** — locally redundant (3 copies within one datacenter), cheapest option

---

## 10. Data Formats

### JSONL (JSON Lines)

A file format where each line is a valid JSON object. Unlike a regular JSON array, JSONL can be read line-by-line without loading the entire file into memory. Our training data looks like:

```
{"description": "Get total sales...", "pyspark_code": "from pyspark.sql...", ...}
{"description": "Join customers...", "pyspark_code": "from pyspark.sql...", ...}
```

### .env File

A configuration file containing environment variables as `KEY=VALUE` pairs. Our `.env.example` shows all available settings. The actual `.env` file (not committed to git) contains your real values. `pydantic-settings` reads this file automatically.

### Pickle (.pkl)

A Python-specific binary format for saving objects to disk. We use it to store the metadata that accompanies the FAISS index (the list of descriptions and their associated PySpark code).

---

## 11. Hardware

### GPU (Graphics Processing Unit)

Originally designed for rendering graphics, GPUs are now essential for AI because they can do thousands of math operations in parallel. Training and running LLMs is almost entirely matrix multiplication — exactly what GPUs excel at.

### VRAM (Video RAM)

The memory on a GPU. This is the primary constraint for LLMs — the model weights, input data, and intermediate calculations must all fit in VRAM. A 6.7B parameter model in 16-bit precision needs ~13 GB of VRAM just for the weights, plus more for computation.

### CUDA

NVIDIA's platform for running general-purpose computations on their GPUs. All the ML libraries (PyTorch, etc.) use CUDA under the hood. "CUDA 12.1" refers to the version — we need 12.1+ for compatibility with the latest libraries.

### GPU Types Used in This Project

| GPU | VRAM | Typical Use | Azure VM | Cost/hr |
|---|---|---|---|---|
| **A100** | 80 GB | Training (fast, lots of memory) | NC24ads_A100_v4 | ~$3.67 (spot) |
| **T4** | 16 GB | Inference (cost-effective for serving) | NC4as_T4_v3 | ~$0.53 |

The A100 is used for training because it's fast and has plenty of memory. The T4 is used for inference because a quantized 6.7B model fits in 16 GB and the T4 is much cheaper for 24/7 operation.

### Spot / Low-Priority Instances

Azure VMs offered at a steep discount (~60% off) because they use spare capacity. The tradeoff: Azure can deallocate them with 30 seconds notice if demand spikes. Fine for training (jobs can be checkpointed and resumed), but not for production inference.

---

## 12. How It All Connects

### Approach 1 (RAG) — Data Flow

```
User: "Get total sales per region"
            │
            ▼
  ┌─── sentence-transformers ───┐
  │  Convert text to 384-dim     │
  │  numerical vector            │
  └──────────┬──────────────────┘
             │
             ▼
  ┌─── FAISS ───────────────────┐
  │  Search index for 3 most    │
  │  similar training examples   │
  └──────────┬──────────────────┘
             │
             ▼
  ┌─── Prompt Template ─────────┐
  │  System prompt               │
  │  + 3 retrieved examples      │
  │  + user's description        │
  └──────────┬──────────────────┘
             │
             ▼
  ┌─── Ollama / vLLM ──────────┐
  │  DeepSeek Coder generates   │
  │  PySpark code based on the  │
  │  prompt with examples        │
  └──────────┬──────────────────┘
             │
             ▼
  ┌─── FastAPI ─────────────────┐
  │  Returns JSON response with  │
  │  the generated PySpark code  │
  └─────────────────────────────┘
```

### Approach 2 (Fine-tuned) — Training Flow

```
  ┌─── training_data.jsonl ─────┐
  │  17 examples of description  │
  │  → PySpark code pairs        │
  └──────────┬──────────────────┘
             │
             ▼
  ┌─── prepare_data.py ─────────┐
  │  Convert to ChatML format    │
  │  Split into train/val        │
  └──────────┬──────────────────┘
             │
             ▼
  ┌─── train_lora.py ───────────┐
  │  Load DeepSeek Coder (4-bit) │  ← BitsAndBytes compresses to 4-bit
  │  Attach LoRA adapters        │  ← PEFT adds small trainable matrices
  │  Train with SFTTrainer       │  ← TRL runs the training loop
  │  on Azure ML A100 GPU        │
  └──────────┬──────────────────┘
             │
             ▼
  ┌─── LoRA Adapter (~50 MB) ───┐
  │  Small file with learned     │
  │  adjustments to the model    │
  └──────────┬──────────────────┘
             │
             ▼
  ┌─── Azure ML Model Registry ─┐
  │  Stored as versioned artifact│
  └─────────────────────────────┘
```

### Approach 2 (Fine-tuned) — Inference Flow

```
User: "Get total sales per region"
            │
            ▼
  ┌─── FastAPI ─────────────────┐
  │  Receives HTTP POST request  │
  └──────────┬──────────────────┘
             │
             ▼
  ┌─── vLLM ────────────────────┐
  │  DeepSeek Coder base model   │
  │  + LoRA adapter loaded       │
  │  Generates PySpark code      │
  │  (no examples needed — the   │
  │   patterns are baked into    │
  │   the adapter weights)       │
  └──────────┬──────────────────┘
             │
             ▼
  ┌─── FastAPI ─────────────────┐
  │  Returns JSON response       │
  └─────────────────────────────┘
```

### The Key Difference

- **RAG** gives the model good examples at query time (the model itself is unchanged)
- **Fine-tuning** changes the model itself so it already knows our patterns (no examples needed at query time)

Both produce the same output format and are served through the same API interface — the user doesn't need to know which approach is running.
