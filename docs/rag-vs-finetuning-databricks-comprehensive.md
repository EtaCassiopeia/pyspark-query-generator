# PySpark Query Generation: RAG vs Fine-Tuning on Databricks (Azure)

## A Comprehensive Guide for Decision-Makers

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [Approach 1: Retrieval-Augmented Generation (RAG)](#3-approach-1-retrieval-augmented-generation-rag)
4. [Approach 2: Fine-Tuning](#4-approach-2-fine-tuning)
5. [Available Models & Infrastructure on Databricks Azure](#5-available-models--infrastructure-on-databricks-azure)
6. [Architecture: RAG on Databricks](#6-architecture-rag-on-databricks)
7. [Architecture: Fine-Tuning on Databricks](#7-architecture-fine-tuning-on-databricks)
8. [Step-by-Step Setup: RAG](#8-step-by-step-setup-rag)
9. [Step-by-Step Setup: Fine-Tuning](#9-step-by-step-setup-fine-tuning)
10. [Open-Source Alternatives](#10-open-source-alternatives)
11. [Head-to-Head Comparison](#11-head-to-head-comparison)
12. [Recommendation & Phased Roadmap](#12-recommendation--phased-roadmap)
13. [Appendix: Glossary & References](#13-appendix-glossary--references)

---

## 1. Executive Summary

We need to automate PySpark query generation from natural language descriptions. Two proven approaches exist:

| | RAG | Fine-Tuning |
|---|---|---|
| **What it does** | Retrieves similar examples at query time, feeds them to an LLM as context | Trains (specializes) a model on our data so it learns PySpark patterns directly |
| **Time to first result** | Hours | Days to weeks |
| **Ongoing data updates** | Add new examples instantly | Requires retraining |
| **Quality ceiling** | High (depends on example coverage) | Very high (model internalizes patterns) |
| **Cost model** | Pay-per-query (inference only) | Upfront training cost + inference cost |

**Our recommendation:** Start with RAG (fast time-to-value, low risk), evaluate quality, then layer in fine-tuning for queries where RAG falls short.

---

## 2. Problem Statement

Our data engineering teams spend significant time writing PySpark queries for recurring intake requests. Each request involves:
- Understanding the business requirement
- Identifying relevant Delta tables and columns
- Writing correct PySpark transformations (filters, joins, aggregations, window functions)
- Validating output

**Goal:** Given a natural language description of a data request, automatically generate a correct PySpark query.

**Constraints:**
- We run on **Databricks on Azure** (our primary compute and data platform)
- External model downloads (e.g., from HuggingFace) are blocked by corporate policy
- We use **Nexus** as our package/model repository -- only models available in Nexus are accessible
- We have access to **Azure OpenAI** through Databricks External Models
- We have ~300 curated training examples (description + PySpark code pairs)

---

## 3. Approach 1: Retrieval-Augmented Generation (RAG)

### How It Works

RAG keeps a base LLM unchanged and enhances it at query time by injecting relevant examples into the prompt.

```
                         +---------------------+
User Request ──────────> | 1. Embed the query  |
  "Get total sales       +---------------------+
   by region for Q4"              |
                                  v
                         +---------------------+
                         | 2. Search Vector DB |  ──> Top-5 similar examples
                         |    (similarity)      |      with PySpark code
                         +---------------------+
                                  |
                                  v
                         +---------------------+
                         | 3. Build Prompt     |
                         |  System prompt      |
                         |  + Retrieved examples|
                         |  + User request     |
                         +---------------------+
                                  |
                                  v
                         +---------------------+
                         | 4. LLM generates    | ──> PySpark query
                         |    PySpark code      |
                         +---------------------+
```

### Key Characteristics

- **No model training required** -- uses an existing LLM as-is
- **Few-shot learning** -- the retrieved examples teach the model by demonstration
- **Dynamic knowledge** -- add new examples to the vector store at any time, no retraining
- **Transparent** -- you can inspect which examples were retrieved and why
- **Quality depends on** -- (a) coverage of example library, (b) embedding quality, (c) base LLM capability

### When RAG Excels

- New queries are similar to existing examples in the training set
- The domain evolves (new tables, new patterns) and you need to adapt quickly
- You want explainability (show users "here are the similar examples we used")
- You have a small dataset (<1000 examples)
- You need fast iteration and experimentation

### When RAG Struggles

- Queries require novel combinations not well-represented in examples
- The LLM needs deep understanding of your specific schema/naming conventions
- You need consistent output formatting that general LLMs don't follow
- Latency is critical (retrieval adds ~100-500ms per query)

---

## 4. Approach 2: Fine-Tuning

### How It Works

Fine-tuning takes a pre-trained LLM and further trains it on your specific data, so the model learns your PySpark patterns, table names, and coding conventions.

```
                    TRAINING PHASE (one-time)
                    =========================
Training Data ──> +------------------------+
  300 examples    | Base Model             |
  (desc + code)   | (e.g., Llama 3.1 8B)  |
                  +------------------------+
                           |
                    QLoRA / LoRA fine-tuning
                    (train adapter weights)
                           |
                           v
                  +------------------------+
                  | Fine-Tuned Model       |
                  | (specialized for       |
                  |  PySpark generation)   |
                  +------------------------+
                           |
                    Register to MLflow/UC
                           |
                           v
                    Deploy to Serving Endpoint


                    INFERENCE PHASE (every query)
                    ==============================
User Request ──────────> +------------------------+
  "Get total sales       | Fine-Tuned Model       | ──> PySpark query
   by region for Q4"     | (no retrieval needed)  |
                         +------------------------+
```

### Key Characteristics

- **Model learns your domain** -- table names, coding style, common patterns
- **Simpler inference** -- no retrieval step, just prompt → response
- **Higher quality ceiling** -- model internalizes patterns rather than copying examples
- **Training cost** -- requires GPU compute for training (hours to days)
- **Static knowledge** -- new patterns require retraining
- **Less transparent** -- harder to explain why the model produced a specific output

### When Fine-Tuning Excels

- You need consistent output format and coding style
- Queries require understanding of company-specific conventions
- You have enough training data (300+ examples, ideally 1000+)
- You want to use a smaller, faster model (fine-tuned 8B can outperform generic 70B)
- Inference latency must be minimal

### When Fine-Tuning Struggles

- Training data is limited or low quality
- The domain changes frequently (new tables added weekly)
- You need rapid iteration (each training run takes hours)
- You lack GPU compute budget for training

---

## 5. Available Models & Infrastructure on Databricks Azure

### 5.1 LLMs Available Through Foundation Model APIs

These models are available **out-of-the-box** on Databricks -- no download or Nexus required:

| Model | Parameters | Context Window | Access Mode | Best For |
|-------|-----------|---------------|-------------|----------|
| **Meta Llama 3.3 70B Instruct** | 70B | 128K tokens | Pay-per-token | High-quality generation (RAG) |
| **Meta Llama 3.1 8B Instruct** | 8B | 128K tokens | Provisioned throughput | Fine-tuning base model |
| **Meta Llama 3.1 70B Instruct** | 70B | 128K tokens | Pay-per-token / Provisioned | RAG generation, fine-tuning |
| **Meta Llama 3.2 3B Instruct** | 3B | 128K tokens | Provisioned throughput | Lightweight fine-tuning |
| **DBRX Instruct** | 132B (MoE) | 32K tokens | Pay-per-token | Alternative for RAG |
| **Mixtral 8x7B Instruct** | ~47B (MoE) | 32K tokens | Pay-per-token | Cost-effective generation |
| **Qwen3-Next-80B-A3B** | 80B (MoE) | Large | Pay-per-token | Efficient RAG generation |

### 5.2 LLMs Available Through Azure OpenAI (External Models)

Via Databricks **External Models** proxy to Azure OpenAI:

| Model | Notes |
|-------|-------|
| **GPT-4o** | Strong code generation, high cost |
| **GPT-4o-mini** | Good code generation, lower cost |
| **GPT-3.5 Turbo** | Fast, cost-effective for simpler queries |

> **Setup:** Create an External Model endpoint in Databricks that proxies to your Azure OpenAI deployment. This gives you a unified Databricks API for both open-source and OpenAI models.

### 5.3 Embedding Models

| Model | Source | Dimensions | Context | Cost | Notes |
|-------|--------|-----------|---------|------|-------|
| **databricks-bge-large-en** | Databricks-hosted (BAAI) | 1024 | 512 tokens | Pay-per-token | Pre-deployed, no setup |
| **databricks-gte-large-en** | Databricks-hosted | 1024 | 512 tokens | Pay-per-token | Pre-deployed, no setup |
| **Qwen3-Embedding-0.6B** | Databricks-hosted | Up to 1024 | ~32K tokens | Pay-per-token | Public Preview, long context |
| **text-embedding-ada-002** | Azure OpenAI | 1536 | 8191 tokens | Per Azure pricing | Via External Models |
| **text-embedding-3-small** | Azure OpenAI | 1536 | 8191 tokens | Per Azure pricing | Via External Models, cost-effective |
| **text-embedding-3-large** | Azure OpenAI | 3072 | 8191 tokens | Per Azure pricing | Via External Models, highest quality |
| **all-MiniLM-L6-v2** | Open source (sentence-transformers) | 384 | 256 tokens | Free (self-hosted) | Requires Nexus or cluster install |

### 5.4 Models Available for Fine-Tuning on Databricks

Through **Mosaic AI Model Training** (managed fine-tuning):

| Model | Parameters | Notes |
|-------|-----------|-------|
| **Meta Llama 3.3 70B Instruct** | 70B | High quality, needs large cluster |
| **Meta Llama 3.1 70B Instruct** | 70B | Proven for code tasks |
| **Meta Llama 3.1 8B Instruct** | 8B | **Recommended starting point** -- good quality/cost balance |
| **Meta Llama 3.2 3B Instruct** | 3B | Fastest training, suitable for simpler tasks |
| **Mistral 7B Instruct v0.3** | 7B | Strong code generation |
| **CodeLlama 13B Instruct** | 13B | Purpose-built for code |

### 5.5 Vector Search Infrastructure

| Feature | Databricks Vector Search | FAISS (Self-Managed) |
|---------|------------------------|---------------------|
| **Type** | Managed serverless | Library on cluster |
| **Persistence** | Persists across sessions | Rebuilt on restart |
| **Auto-sync with Delta** | Yes (Change Data Feed) | Manual rebuild |
| **Embedding** | Auto-embeds using Foundation Model APIs | Manual embedding |
| **Scalability** | Serverless, auto-scales | Limited by cluster memory |
| **Permissions needed** | Vector Search endpoint access | None (runs on cluster) |
| **Setup time** | 5-10 min for endpoint | Seconds |

### 5.6 Nexus Considerations

Since HuggingFace downloads are blocked:

- **Databricks-hosted models** (Foundation Model APIs) are accessible without any download -- they run on Databricks infrastructure
- **Azure OpenAI models** are accessible through External Models -- no download needed
- **Open-source models** (e.g., `all-MiniLM-L6-v2`, custom HuggingFace models) must be:
  1. Uploaded to Nexus first by an admin
  2. Or uploaded to a Unity Catalog Volume on Databricks
  3. Or pre-installed on a custom Docker image for Databricks clusters
- **Recommendation:** Prefer Databricks-hosted models to avoid Nexus dependency

---

## 6. Architecture: RAG on Databricks

### Option A: Managed RAG (Databricks Vector Search + Foundation Model APIs)

```
┌─────────────────────────────────────────────────────────────────────┐
│                     DATABRICKS WORKSPACE                            │
│                                                                     │
│  ┌──────────────┐     ┌──────────────────────┐                      │
│  │ Delta Table  │     │ Vector Search        │                      │
│  │ (training    │────>│ Endpoint             │                      │
│  │  examples)   │ CDF │  ┌─────────────────┐ │                      │
│  │              │sync │  │ Delta Sync Index │ │                      │
│  └──────────────┘     │  │ (auto-embedded   │ │                      │
│                       │  │  with BGE/GTE)   │ │                      │
│                       │  └─────────────────┘ │                      │
│                       └──────────┬───────────┘                      │
│                                  │ similarity search                │
│                                  v                                  │
│  ┌──────────────┐     ┌──────────────────────┐                      │
│  │ User Request │────>│ RAG Orchestrator     │                      │
│  │ (notebook or │     │  1. Query Vector DB  │                      │
│  │  API)        │     │  2. Build prompt     │                      │
│  └──────────────┘     │  3. Call LLM         │                      │
│                       └──────────┬───────────┘                      │
│                                  │                                  │
│                                  v                                  │
│                       ┌──────────────────────┐                      │
│                       │ Foundation Model API │                      │
│                       │ (Llama 3.3 70B or    │                      │
│                       │  Azure OpenAI GPT-4o)│                      │
│                       └──────────────────────┘                      │
│                                  │                                  │
│                                  v                                  │
│                            Generated PySpark                        │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ MLflow: Tracing, Evaluation, Experiment Tracking             │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

**Key components:**
- **Delta Table** stores training examples (description + PySpark code)
- **Vector Search** auto-embeds and indexes using `databricks-bge-large-en`
- **Foundation Model API** generates PySpark from prompt + retrieved examples
- **MLflow** tracks experiments and evaluates quality

### Option B: Self-Managed RAG (FAISS + Foundation Model APIs)

```
┌─────────────────────────────────────────────────────────────────────┐
│                     DATABRICKS NOTEBOOK                              │
│                                                                     │
│  ┌──────────────┐     ┌──────────────────────┐                      │
│  │ Delta Table  │────>│ In-Memory FAISS      │                      │
│  │ (training    │ load│ Index                │                      │
│  │  examples)   │     │ (sentence-transformers│                      │
│  └──────────────┘     │  embeddings)         │                      │
│                       └──────────┬───────────┘                      │
│                                  │ similarity search                │
│                                  v                                  │
│  ┌──────────────┐     ┌──────────────────────┐                      │
│  │ User Request │────>│ RAG Pipeline         │                      │
│  └──────────────┘     │  (retrieve + prompt) │                      │
│                       └──────────┬───────────┘                      │
│                                  │                                  │
│                                  v                                  │
│                       ┌──────────────────────┐                      │
│                       │ Foundation Model API │                      │
│                       └──────────────────────┘                      │
│                                  │                                  │
│                                  v                                  │
│                            Generated PySpark                        │
└─────────────────────────────────────────────────────────────────────┘
```

**When to use Option B:** No Vector Search endpoint permissions, prototyping, or small datasets (<1000 examples).

---

## 7. Architecture: Fine-Tuning on Databricks

### Option A: Managed Fine-Tuning (Mosaic AI Model Training)

```
┌─────────────────────────────────────────────────────────────────────┐
│                     TRAINING PHASE                                   │
│                                                                     │
│  ┌──────────────┐     ┌──────────────────────┐                      │
│  │ Training Data│     │ Mosaic AI Model      │                      │
│  │ (Delta Table │────>│ Training             │                      │
│  │  or JSONL in │     │                      │                      │
│  │  UC Volume)  │     │ - Base: Llama 3.1 8B │                      │
│  └──────────────┘     │ - SFT on your data   │                      │
│                       │ - Managed GPU cluster │                      │
│                       │ - Up to 2x faster    │                      │
│                       └──────────┬───────────┘                      │
│                                  │                                  │
│                                  v                                  │
│                       ┌──────────────────────┐                      │
│                       │ MLflow + Unity Catalog│                      │
│                       │ (model checkpoints   │                      │
│                       │  auto-registered)    │                      │
│                       └──────────┬───────────┘                      │
│                                  │                                  │
│                                  v                                  │
│                       ┌──────────────────────┐                      │
│                       │ Model Serving        │                      │
│                       │ Endpoint             │                      │
│                       │ (Provisioned         │                      │
│                       │  Throughput)         │                      │
│                       └──────────────────────┘                      │
│                                                                     │
│                     INFERENCE PHASE                                   │
│                                                                     │
│  ┌──────────────┐     ┌──────────────────────┐                      │
│  │ User Request │────>│ Model Serving        │──> Generated PySpark  │
│  └──────────────┘     │ Endpoint             │                      │
│                       └──────────────────────┘                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Option B: Manual Fine-Tuning (HuggingFace + PEFT on Databricks Cluster)

```
┌─────────────────────────────────────────────────────────────────────┐
│                     DATABRICKS GPU CLUSTER                           │
│                                                                     │
│  ┌──────────────┐     ┌──────────────────────┐                      │
│  │ Training Data│     │ Custom Training      │                      │
│  │ (Delta Table)│────>│ Script               │                      │
│  └──────────────┘     │                      │                      │
│                       │ - HuggingFace        │                      │
│                       │   Transformers       │                      │
│                       │ - PEFT (LoRA/QLoRA)  │                      │
│                       │ - TRL (SFTTrainer)   │                      │
│                       │ - 4-bit quantization │                      │
│                       └──────────┬───────────┘                      │
│                                  │                                  │
│                                  v                                  │
│                       ┌──────────────────────┐                      │
│                       │ LoRA Adapter         │                      │
│                       │ (saved to UC Volume  │                      │
│                       │  or MLflow)          │                      │
│                       └──────────┬───────────┘                      │
│                                  │ merge                            │
│                                  v                                  │
│                       ┌──────────────────────┐                      │
│                       │ Merged Model         │──> Deploy to         │
│                       │ (registered to UC)   │    Model Serving     │
│                       └──────────────────────┘                      │
└─────────────────────────────────────────────────────────────────────┘
```

**When to use Option B:** Mosaic AI fine-tuning has permission issues, you need custom training logic, or you want finer control over hyperparameters.

---

## 8. Step-by-Step Setup: RAG

### 8.1 Data Preparation

```
Step 1: Create Delta Table with Training Examples
─────────────────────────────────────────────────
Notebook: databricks/01_data_prep.py

Input:  CSV file with columns:
        - intake_number (ID)
        - title (short description)
        - intake_description (natural language requirement)
        - pyspark_query (ground truth PySpark code)

Output: Delta tables in Unity Catalog:
        - {catalog}.{schema}.training_examples  (all data)
        - {catalog}.{schema}.training_set       (85% for retrieval)
        - {catalog}.{schema}.validation_set     (15% for evaluation)
```

### 8.2 Option A: Managed Vector Search Setup

```
Step 2a: Create Vector Search Endpoint
──────────────────────────────────────
Notebook: databricks/02b_rag_databricks.py

1. Enable Change Data Feed on training_set table:
   ALTER TABLE {catalog}.{schema}.training_set
   SET TBLPROPERTIES (delta.enableChangeDataFeed = true);

2. Create Vector Search endpoint:
   from databricks.vector_search.client import VectorSearchClient
   vsc = VectorSearchClient()
   vsc.create_endpoint(
       name="pyspark-gen-vs-endpoint",
       endpoint_type="STANDARD"
   )
   # Wait 5-10 minutes for provisioning

3. Create Delta Sync index:
   vsc.create_delta_sync_index(
       endpoint_name="pyspark-gen-vs-endpoint",
       index_name="{catalog}.{schema}.training_vs_index",
       source_table_name="{catalog}.{schema}.training_set",
       primary_key="intake_number",
       pipeline_type="TRIGGERED",
       embedding_source_columns=[
           EmbeddingSourceColumn(
               name="intake_description",
               model_endpoint_name="databricks-bge-large-en"
           )
       ],
       columns_to_sync=["intake_number", "title",
                        "intake_description", "pyspark_query"]
   )

4. Sync the index (initial load):
   index.sync()
```

```
Step 3a: Query and Generate
───────────────────────────
1. Similarity search:
   results = index.similarity_search(
       query_text="Get total sales by region for Q4",
       columns=["intake_description", "pyspark_query"],
       num_results=5
   )

2. Build few-shot prompt with retrieved examples

3. Call Foundation Model API:
   from databricks.sdk import WorkspaceClient
   w = WorkspaceClient()
   response = w.serving_endpoints.query(
       name="databricks-meta-llama-3-3-70b-instruct",
       messages=[
           {"role": "system", "content": system_prompt},
           {"role": "user", "content": user_prompt_with_examples}
       ]
   )
```

### 8.3 Option B: FAISS Setup (No Permissions Needed)

```
Step 2b: Build FAISS Index In-Memory
─────────────────────────────────────
Notebook: databricks/02b_rag_faiss.py

1. Load training data:
   df = spark.sql(f"SELECT * FROM {catalog}.{schema}.training_set")
   examples = [row.asDict() for row in df.collect()]

2. Embed all descriptions:
   from sentence_transformers import SentenceTransformer
   model = SentenceTransformer("all-MiniLM-L6-v2")
   embeddings = model.encode([e["intake_description"] for e in examples])

3. Build FAISS index:
   import faiss
   index = faiss.IndexFlatIP(embeddings.shape[1])
   faiss.normalize_L2(embeddings)
   index.add(embeddings)

4. Query: same as above but using FAISS similarity search
5. Generate: same Foundation Model API call
```

### 8.4 Evaluation

```
Step 4: Evaluate RAG Quality
────────────────────────────
Option A - Mosaic AI Agent Evaluation (recommended):
   import mlflow
   results = mlflow.evaluate(
       model=rag_chain,
       data=eval_dataset,
       model_type="databricks-agent"
   )
   # Automated LLM-judge metrics:
   # - Answer correctness
   # - Groundedness (hallucination detection)
   # - Retrieval relevance

Option B - Custom metrics (already implemented):
   # BLEU-4, Exact Match, Operation Coverage, Structural Similarity
   from shared.evaluation.evaluator import evaluate_predictions
```

### 8.5 Adding New Examples

```
Step 5: Updating the Knowledge Base
────────────────────────────────────
Managed Vector Search:
  1. INSERT new rows into Delta table
  2. Vector Search auto-syncs (triggered or continuous)
  3. No code changes needed

FAISS:
  1. INSERT new rows into Delta table
  2. Re-run embedding + index build cells
  3. ~5 seconds for 300 rows
```

---

## 9. Step-by-Step Setup: Fine-Tuning

### 9.1 Data Preparation

```
Step 1: Prepare Training Data in Chat Format
─────────────────────────────────────────────
Notebook: databricks/01_data_prep.py

The data must be in chat completion format (messages array):
[
  {"role": "system", "content": "You are a PySpark expert..."},
  {"role": "user", "content": "Generate PySpark for: <description>"},
  {"role": "assistant", "content": "<pyspark_code>"}
]

Output formats:
  - Delta table with `messages` column (for Mosaic AI)
  - JSONL files in UC Volume (for Mosaic AI)
  - JSONL files on DBFS (for manual training)
```

### 9.2 Option A: Managed Fine-Tuning (Mosaic AI)

```
Step 2a: Create Fine-Tuning Run
────────────────────────────────
Notebook: databricks/02a_finetune.py

from databricks.model_training import foundation_model as fm

run = fm.create(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    train_data_path="{catalog}.{schema}.training_set",
    task_type="CHAT_COMPLETION",
    register_to="{catalog}.{schema}",
    training_duration="3ep",           # 3 epochs
    learning_rate="5e-5",
    eval_data_path="{catalog}.{schema}.validation_set"
)

# Monitor progress:
run.get_events()

# Training typically takes 1-4 hours depending on data size and model
```

```
Step 3a: Deploy Fine-Tuned Model
─────────────────────────────────
The model is automatically registered to Unity Catalog.

Deploy to Model Serving:
  1. Go to Serving > Create Serving Endpoint
  2. Select the fine-tuned model from Unity Catalog
  3. Choose "Provisioned Throughput" for production
  4. Or use API:

  from databricks.sdk import WorkspaceClient
  w = WorkspaceClient()
  w.serving_endpoints.create(
      name="pyspark-gen-finetuned",
      config=EndpointCoreConfigInput(
          served_entities=[ServedEntityInput(
              entity_name="{catalog}.{schema}.pyspark-gen-llama-8b",
              entity_version="1",
              scale_to_zero_enabled=True
          )]
      )
  )
```

### 9.3 Option B: Manual Fine-Tuning (HuggingFace + PEFT)

```
Step 2b: Run Custom Training Script
────────────────────────────────────
Notebook: databricks/02a_finetune_manual.py

Prerequisites:
  - GPU cluster (A100 recommended, A10G minimum)
  - Libraries: transformers, peft, trl, bitsandbytes, accelerate
    (install via %pip or cluster init script)
  - Base model accessible (Databricks-managed or in UC Volume)

Training configuration (auto-detected by GPU):
  ┌──────────────┬─────────┬───────────────┬─────────┐
  │ GPU          │ Batch   │ Grad Accum    │ Seq Len │
  ├──────────────┼─────────┼───────────────┼─────────┤
  │ A100 80GB    │ 4       │ 4             │ 2048    │
  │ A100 40GB    │ 4       │ 4             │ 2048    │
  │ A10G (24GB)  │ 2       │ 8             │ 2048    │
  │ T4 (16GB)    │ 1       │ 16            │ 1536    │
  └──────────────┴─────────┴───────────────┴─────────┘

QLoRA Configuration:
  - Quantization: 4-bit NormalFloat (NF4)
  - LoRA rank: 16
  - LoRA alpha: 32
  - Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
  - Dropout: 0.05

Training: 3 epochs, cosine LR scheduler, eval every epoch
Output: LoRA adapter saved to UC Volume + registered to MLflow
```

```
Step 3b: Merge and Deploy
─────────────────────────
1. Merge LoRA adapter into base model:
   merged_model = PeftModel.from_pretrained(base_model, adapter_path)
   merged_model = merged_model.merge_and_unload()

2. Register merged model to Unity Catalog via MLflow:
   mlflow.transformers.log_model(
       transformers_model={"model": merged_model, "tokenizer": tokenizer},
       registered_model_name="{catalog}.{schema}.pyspark-gen-merged"
   )

3. Deploy to Model Serving endpoint (same as Step 3a)
```

### 9.4 Evaluation

```
Step 4: Evaluate Fine-Tuned Model
──────────────────────────────────
1. Run inference on validation set (15% held out)
2. Compare with RAG using same metrics:
   - BLEU-4 (token-level similarity)
   - Exact Match (normalized string equality)
   - Operation Coverage (PySpark operations present)
   - Structural Similarity (code pattern matching)
3. Use Mosaic AI Agent Evaluation for LLM-judge metrics
4. A/B test: run same queries through RAG and fine-tuned model
```

### 9.5 Retraining

```
Step 5: Updating the Fine-Tuned Model
──────────────────────────────────────
When new training examples are added:
  1. Add to Delta table
  2. Re-run fine-tuning (incremental or full)
  3. Register new model version to UC
  4. Update serving endpoint to new version
  5. Run evaluation on validation set

Recommended cadence: monthly or when >50 new examples are added
```

---

## 10. Open-Source Alternatives

For scenarios where Databricks-managed services are unavailable or cost-prohibitive:

### 10.1 Embedding Models (Open Source)

| Model | Size | Dims | Notes | Nexus Required? |
|-------|------|------|-------|-----------------|
| `all-MiniLM-L6-v2` | 80MB | 384 | Fast, good for prototyping | Yes |
| `BAAI/bge-large-en-v1.5` | 1.3GB | 1024 | High quality, same as Databricks-hosted | Yes |
| `intfloat/e5-large-v2` | 1.3GB | 1024 | Strong multilingual support | Yes |

> **Note:** These require Nexus upload since HuggingFace is blocked. Prefer `databricks-bge-large-en` (same model, no download needed).

### 10.2 LLMs (Open Source)

| Model | Size | Notes | Nexus Required? |
|-------|------|-------|-----------------|
| `deepseek-coder-6.7b-instruct` | ~4GB (4-bit) | Strong code generation | Yes |
| `codellama-13b-instruct` | ~8GB (4-bit) | Meta's code-specific model | Yes |
| `Llama 3.1 8B Instruct` | ~5GB (4-bit) | General + code | Available on Databricks |

### 10.3 Vector Databases (Open Source)

| Solution | Notes |
|----------|-------|
| **FAISS** | In-memory, no server needed, great for <100K docs |
| **ChromaDB** | Persistent, local, Python-native |
| **LanceDB** | Embedded, serverless, good for Delta Lake integration |

### 10.4 Fine-Tuning Frameworks (Open Source)

| Framework | Notes |
|-----------|-------|
| **PEFT** (HuggingFace) | LoRA, QLoRA adapters |
| **TRL** (HuggingFace) | SFTTrainer, DPO, RLHF |
| **Axolotl** | Simplified fine-tuning config |
| **LitGPT** | Lightning AI's fine-tuning tool |

> All of these are already used in our `approach2_finetune/` code and work on Databricks GPU clusters.

---

## 11. Head-to-Head Comparison

### 11.1 Feature Comparison

| Dimension | RAG | Fine-Tuning |
|-----------|-----|-------------|
| **Setup complexity** | Low -- create index, write prompt template | Medium-High -- data prep, training, deployment |
| **Time to first result** | Hours | Days |
| **Training cost** | $0 | $50-500+ per run (depends on model/GPU) |
| **Inference cost** | Higher (retrieval + generation) | Lower (generation only) |
| **Inference latency** | ~1-3 seconds (retrieval + generation) | ~0.5-1.5 seconds (generation only) |
| **Quality with small data** (<300 examples) | Good | Fair (risk of overfitting) |
| **Quality with large data** (>1000 examples) | Good | Excellent |
| **Adapts to new data** | Instantly (add to vector store) | Requires retraining (hours) |
| **Output consistency** | Variable (depends on LLM mood) | Consistent (learned patterns) |
| **Explainability** | High (show retrieved examples) | Low (black box) |
| **Hallucination risk** | Lower (grounded in examples) | Higher (model may generate plausible but incorrect code) |
| **Databricks integration** | Native (Vector Search + Foundation Model APIs) | Native (Mosaic AI Training + Model Serving) |
| **Permissions needed** | Vector Search endpoint (or none for FAISS) | GPU cluster + Model Training permissions |

### 11.2 Cost Comparison (Estimated)

| Cost Item | RAG (Managed) | RAG (FAISS) | Fine-Tuning (Managed) | Fine-Tuning (Manual) |
|-----------|---------------|-------------|----------------------|---------------------|
| **Embedding (one-time)** | Included in Vector Search | Free (on cluster) | N/A | N/A |
| **Vector Search endpoint** | ~$0.50/hr | $0 | N/A | N/A |
| **Training compute** | $0 | $0 | ~$50-200/run | ~$20-100/run |
| **Inference (per 1K queries)** | ~$2-10 (LLM API) | ~$2-10 (LLM API) | ~$1-5 (own endpoint) | ~$1-5 (own endpoint) |
| **Serving endpoint** | Uses shared Foundation Model APIs | Uses shared Foundation Model APIs | ~$1-4/hr (provisioned) | ~$1-4/hr (provisioned) |

> Costs are approximate and depend on model size, query length, and throughput.

### 11.3 Quality Expectations

Based on our evaluation framework (300 examples, PySpark generation):

| Metric | RAG (Expected) | Fine-Tuned (Expected) | Notes |
|--------|----------------|----------------------|-------|
| **BLEU-4** | 0.25-0.40 | 0.35-0.55 | Token overlap with reference |
| **Operation Coverage** | 0.70-0.85 | 0.80-0.95 | Correct PySpark operations |
| **Structural Similarity** | 0.50-0.70 | 0.65-0.85 | Code pattern matching |
| **Exact Match** | 0.05-0.15 | 0.10-0.25 | Identical output (strict) |

> These are estimates. Actual results depend on data quality and model selection. Run both approaches on the validation set to get real numbers.

### 11.4 Risk Assessment

| Risk | RAG | Fine-Tuning |
|------|-----|-------------|
| **Data quality issues** | Low impact -- bad examples diluted by good ones | High impact -- model learns bad patterns |
| **Model deprecation** | Low risk -- swap LLM endpoint | Medium risk -- retrain on new base model |
| **Scaling to new domains** | Easy -- add examples | Hard -- collect data + retrain |
| **Compliance/governance** | Simple -- data stays in Delta tables | Complex -- model weights contain training data |
| **Team expertise needed** | Basic Python + Databricks | ML engineering + GPU knowledge |

---

## 12. Recommendation & Phased Roadmap

### Phase 1: RAG MVP (Weeks 1-3)

**Goal:** Deliver a working prototype with measurable quality metrics.

```
Week 1: Data & Infrastructure
  ├── Curate and clean training examples in Delta table
  ├── Set up Vector Search endpoint + Delta Sync index
  └── Validate embedding quality with sample queries

Week 2: RAG Pipeline
  ├── Implement RAG orchestrator (retrieve → prompt → generate)
  ├── Choose LLM: start with Llama 3.3 70B (Foundation Model API)
  ├── Tune retrieval (top-k, similarity threshold)
  └── Build interactive notebook for demos

Week 3: Evaluation & Baseline
  ├── Run evaluation on validation set (all 4 metrics)
  ├── Document baseline quality numbers
  ├── Identify failure categories (where RAG falls short)
  └── Present results to stakeholders
```

**Deliverables:**
- Working RAG pipeline on Databricks
- Quality baseline metrics
- Interactive demo notebook
- List of failure categories for Phase 2

### Phase 2: Fine-Tuning Evaluation (Weeks 4-7)

**Goal:** Determine if fine-tuning improves quality enough to justify the added complexity.

```
Week 4-5: Training
  ├── Prepare data in chat completion format
  ├── Run Mosaic AI fine-tuning on Llama 3.1 8B
  ├── If managed fine-tuning blocked: use manual QLoRA approach
  └── Register model to Unity Catalog

Week 6: Evaluation
  ├── Run same evaluation on fine-tuned model
  ├── Compare RAG vs fine-tuned on all metrics
  ├── Focus on failure categories from Phase 1
  └── Measure latency and cost differences

Week 7: Decision
  ├── If fine-tuning wins by >10% on key metrics: plan production deployment
  ├── If marginal improvement: stick with RAG (simpler, cheaper)
  ├── If mixed: use hybrid (RAG for common queries, fine-tuned for complex)
  └── Present comparison to stakeholders
```

### Phase 3: Production & Hybrid (Weeks 8-12)

**Goal:** Deploy the chosen approach (or hybrid) for production use.

```
  ├── Deploy chosen approach to Model Serving endpoint
  ├── Build API layer for integration with existing tools
  ├── Set up monitoring (MLflow Tracing, quality dashboards)
  ├── Implement feedback loop (users rate generated queries)
  ├── Establish retraining cadence (monthly for fine-tuning)
  └── Consider hybrid: RAG with fine-tuned LLM as generator
```

### The Hybrid Approach (Best of Both Worlds)

The most powerful setup combines both approaches:

```
User Request
     │
     v
┌─────────────────┐
│ Vector Search   │──> Retrieve similar examples
│ (RAG retrieval) │
└────────┬────────┘
         │
         v
┌─────────────────┐
│ Fine-Tuned LLM  │──> Generate PySpark
│ (specialized    │    (with examples as additional context)
│  generator)     │
└─────────────────┘
```

**Why this works:** The fine-tuned model already knows PySpark conventions, and the retrieved examples provide specific context for the current query. This typically outperforms either approach alone.

---

## 13. Appendix: Glossary & References

### Glossary

| Term | Definition |
|------|-----------|
| **RAG** | Retrieval-Augmented Generation -- enhance LLM with external knowledge at query time |
| **Fine-tuning** | Further training a pre-trained model on domain-specific data |
| **LoRA** | Low-Rank Adaptation -- efficient fine-tuning that trains small adapter matrices |
| **QLoRA** | Quantized LoRA -- combines 4-bit model quantization with LoRA for memory efficiency |
| **Vector Search** | Database that finds similar items by comparing numerical embeddings |
| **Embedding** | Dense numerical representation of text that captures semantic meaning |
| **Foundation Model APIs** | Databricks' managed endpoints for serving pre-trained LLMs |
| **Unity Catalog (UC)** | Databricks' unified governance layer for data and AI assets |
| **MLflow** | Open-source platform for ML lifecycle management (tracking, registry, deployment) |
| **Delta Table** | Databricks' transactional data lake table format |
| **Mosaic AI** | Databricks' suite of AI/ML tools (training, serving, evaluation) |
| **FAISS** | Facebook AI Similarity Search -- open-source vector search library |
| **Provisioned Throughput** | Dedicated compute for model serving (vs. pay-per-token) |
| **Change Data Feed (CDF)** | Delta Lake feature that tracks row-level changes for incremental processing |

### Key Databricks Documentation

- [Foundation Model APIs - Supported Models](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/foundation-model-apis/supported-models)
- [Vector Search - Create Endpoints and Indexes](https://learn.microsoft.com/en-us/azure/databricks/vector-search/create-vector-search)
- [Foundation Model Fine-Tuning](https://learn.microsoft.com/en-us/azure/databricks/large-language-models/foundation-model-training/)
- [Mosaic AI Agent Framework](https://docs.databricks.com/en/generative-ai/agent-framework/index.html)
- [Mosaic AI Agent Evaluation](https://docs.databricks.com/en/generative-ai/agent-evaluation/index.html)
- [External Models](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/external-models/)
- [MLflow on Databricks](https://docs.databricks.com/en/mlflow/index.html)

### Existing Project Resources

- `databricks/01_data_prep.py` -- Data preparation notebook
- `databricks/02a_finetune.py` -- Managed fine-tuning notebook
- `databricks/02a_finetune_manual.py` -- Manual fine-tuning notebook (QLoRA)
- `databricks/02b_rag_databricks.py` -- Managed RAG notebook (Vector Search)
- `databricks/02b_rag_faiss.py` -- Self-managed RAG notebook (FAISS)
- `databricks/03_interactive_app.py` -- Interactive demo notebook
- `docs/approach1-rag-guide.md` -- Detailed RAG guide
- `docs/approach2-finetune-guide.md` -- Detailed fine-tuning guide
