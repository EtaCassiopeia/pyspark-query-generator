# Databricks notebook source
# MAGIC %md
# MAGIC # Step 2B-i: RAG with FAISS + sentence-transformers (Open Source)
# MAGIC
# MAGIC Uses open-source `all-MiniLM-L6-v2` for embeddings and FAISS for in-memory
# MAGIC vector search. No extra permissions or managed services needed.
# MAGIC
# MAGIC **Trade-off:** Index is rebuilt on every notebook restart (~5 sec for 300 rows).
# MAGIC
# MAGIC **Prerequisites:** Run notebook `01_data_prep` first.

# COMMAND ----------

# DBTITLE 1,Configuration - EDIT THESE
CATALOG = "your_catalog"
SCHEMA = "pyspark_gen"
SOURCE_TABLE = f"{CATALOG}.{SCHEMA}.training_examples"

# Foundation model for generation (choose one available in your workspace)
# Check: Workspace > Serving > Foundation Model APIs
GENERATION_MODEL = "databricks-meta-llama-3-1-70b-instruct"  # or "databricks-dbrx-instruct"

# Number of similar examples to retrieve
TOP_K = 5

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Install dependencies (one-time per session)

# COMMAND ----------

%pip install sentence-transformers faiss-cpu

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load data from Delta table

# COMMAND ----------

# Re-declare config after restartPython (kernel state is cleared)
CATALOG = "your_catalog"
SCHEMA = "pyspark_gen"
SOURCE_TABLE = f"{CATALOG}.{SCHEMA}.training_examples"
GENERATION_MODEL = "databricks-meta-llama-3-1-70b-instruct"
TOP_K = 5

# COMMAND ----------

# Load training examples into pandas (300 rows fits easily in memory)
pdf = spark.table(SOURCE_TABLE).toPandas()
print(f"Loaded {len(pdf)} examples")
pdf[["intake_number", "title", "intake_description"]].head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Build in-memory vector index
# MAGIC
# MAGIC Uses `all-MiniLM-L6-v2` — a small (~80MB) embedding model that runs
# MAGIC locally on the driver node. No API calls, no permissions needed.

# COMMAND ----------

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Load embedding model (downloads ~80MB on first run, cached after)
print("Loading embedding model 'all-MiniLM-L6-v2'...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model ready.")

# Embed all intake descriptions
descriptions = pdf["intake_description"].tolist()
print(f"Embedding {len(descriptions)} intake descriptions...")
embeddings = embed_model.encode(descriptions, normalize_embeddings=True, show_progress_bar=True)
embeddings = embeddings.astype("float32")

# Build FAISS index (inner product on normalized vectors = cosine similarity)
dim = embeddings.shape[1]
faiss_index = faiss.IndexFlatIP(dim)
faiss_index.add(embeddings)
print(f"FAISS index built: {faiss_index.ntotal} vectors, dim={dim}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Test retrieval

# COMMAND ----------

def retrieve_examples(requirement: str, top_k: int = TOP_K) -> list:
    """Find the top-k most similar intake examples using FAISS."""
    query_vec = embed_model.encode([requirement], normalize_embeddings=True).astype("float32")
    scores, indices = faiss_index.search(query_vec, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        row = pdf.iloc[idx]
        results.append({
            "intake_number": row["intake_number"],
            "title": row["title"],
            "intake_description": row["intake_description"],
            "pyspark_query": row["pyspark_query"],
            "score": float(score),
        })
    return results

# COMMAND ----------

# Test retrieval
test_results = retrieve_examples("Get average order value per customer segment")

for r in test_results:
    print(f"Score: {r['score']:.3f} | {r['intake_number']} | {r['title']}")
    print(f"  {r['intake_description'][:100]}...")
    print("-" * 50)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. RAG pipeline — retrieve + generate

# COMMAND ----------

import requests

DATABRICKS_HOST = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
DATABRICKS_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

SYSTEM_PROMPT = """You are a PySpark query generator. Given an intake ticket with a title and description, produce a complete, runnable PySpark query.

Follow the exact coding style shown in the reference examples below.

Rules:
- Use PySpark DataFrame API (not RDD)
- Use spark.table() to reference tables
- Import pyspark.sql.functions as F
- Use F.col() for column references
- Output only the code, no explanations
- Include comments only for non-obvious logic"""


def build_prompt(requirement: str, examples: list) -> str:
    """Format retrieved intake examples into the user message."""
    examples_text = ""
    for i, ex in enumerate(examples, 1):
        examples_text += f"--- Example {i} ({ex['intake_number']}) ---\n"
        examples_text += f"Title: {ex['title']}\n"
        examples_text += f"Description: {ex['intake_description']}\n"
        examples_text += f"PySpark Query:\n```python\n{ex['pyspark_query']}\n```\n\n"

    return f"""=== REFERENCE EXAMPLES FROM PAST INTAKES ===

{examples_text}
=== NEW REQUIREMENT ===

{requirement}

Generate the PySpark query:"""


def generate_query(requirement: str, top_k: int = TOP_K) -> str:
    """Full RAG pipeline: retrieve similar intakes, then generate."""
    # 1. Retrieve similar examples via FAISS
    examples = retrieve_examples(requirement, top_k)

    # 2. Build prompt with examples as context
    user_message = build_prompt(requirement, examples)

    # 3. Generate via Foundation Model API
    response = requests.post(
        f"{DATABRICKS_HOST}/serving-endpoints/{GENERATION_MODEL}/invocations",
        headers={"Authorization": f"Bearer {DATABRICKS_TOKEN}"},
        json={
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            "max_tokens": 1024,
            "temperature": 0.1,
        },
    )
    return response.json()["choices"][0]["message"]["content"]

# COMMAND ----------

# DBTITLE 1,Test the full RAG pipeline
result = generate_query("Get total revenue by product category for the last 30 days")
print(result)

# COMMAND ----------

# DBTITLE 1,Test with more examples
test_requirements = [
    "Find the top 5 customers by total spend in 2024",
    "Calculate the rolling 7-day average of daily active users",
    "Join inventory with sales data and find items below reorder threshold",
    "Pivot monthly sales data by region",
]

for req in test_requirements:
    print(f"\n{'='*60}")
    print(f"REQUIREMENT: {req}")
    print(f"{'='*60}")
    result = generate_query(req)
    print(result)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Evaluate on validation set

# COMMAND ----------

val_rows = spark.table(f"{CATALOG}.{SCHEMA}.validation_set").collect()

results = []
for row in val_rows:
    generated = generate_query(row["intake_description"])
    results.append({
        "intake_number": row["intake_number"],
        "title": row["title"],
        "description": row["intake_description"],
        "expected": row["pyspark_query"],
        "generated": generated,
    })

results_df = spark.createDataFrame(results)
results_df.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.rag_faiss_eval_results")
display(results_df)
