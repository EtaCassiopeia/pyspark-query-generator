# Databricks notebook source
# MAGIC %md
# MAGIC # Step 2B: RAG Approach with Databricks Vector Search
# MAGIC
# MAGIC Uses **Databricks Vector Search** to store your 300 examples and
# MAGIC **Foundation Model APIs** (DBRX / Llama / Mixtral) to generate queries.
# MAGIC
# MAGIC This is faster to set up than fine-tuning and still gives strong results.
# MAGIC
# MAGIC **Prerequisites:** Run notebook `01_data_prep` first.

# COMMAND ----------

# DBTITLE 1,Configuration - EDIT THESE
CATALOG = "your_catalog"
SCHEMA = "pyspark_gen"
SOURCE_TABLE = f"{CATALOG}.{SCHEMA}.training_examples"

# Vector search
VS_ENDPOINT_NAME = "pyspark_gen_vs_endpoint"  # vector search endpoint
VS_INDEX_NAME = f"{CATALOG}.{SCHEMA}.examples_index"

# Foundation model for generation (choose one available in your workspace)
# Check available models: Workspace > Serving > Foundation Model APIs
GENERATION_MODEL = "databricks-meta-llama-3-1-70b-instruct"  # or "databricks-dbrx-instruct"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2B.1 Enable Change Data Feed on the source table
# MAGIC
# MAGIC Required for Databricks Vector Search to sync from Delta.

# COMMAND ----------

spark.sql(f"ALTER TABLE {SOURCE_TABLE} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
print(f"CDF enabled on {SOURCE_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2B.2 Create Vector Search endpoint

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient()

# Create endpoint (one-time operation — skip if it already exists)
try:
    vsc.create_endpoint(name=VS_ENDPOINT_NAME, endpoint_type="STANDARD")
    print(f"Creating endpoint '{VS_ENDPOINT_NAME}'... (takes 5-10 minutes)")
except Exception as e:
    if "already exists" in str(e):
        print(f"Endpoint '{VS_ENDPOINT_NAME}' already exists.")
    else:
        raise

# COMMAND ----------

# Wait for endpoint to be ready (re-run until status is ONLINE)
endpoint = vsc.get_endpoint(VS_ENDPOINT_NAME)
print(f"Endpoint status: {endpoint}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2B.3 Create Vector Search index
# MAGIC
# MAGIC This embeds the `description` column and creates a searchable index.
# MAGIC Uses Databricks' built-in embedding model — no external API needed.

# COMMAND ----------

try:
    index = vsc.create_delta_sync_index(
        endpoint_name=VS_ENDPOINT_NAME,
        index_name=VS_INDEX_NAME,
        source_table_name=SOURCE_TABLE,
        pipeline_type="TRIGGERED",          # manual refresh; use "CONTINUOUS" for auto-sync
        primary_key="id",
        embedding_source_columns=["description"],  # Databricks embeds this column automatically
        embedding_model_endpoint_name="databricks-bge-large-en",  # built-in embedding model
    )
    print(f"Index '{VS_INDEX_NAME}' created. Syncing...")
except Exception as e:
    if "already exists" in str(e):
        print(f"Index '{VS_INDEX_NAME}' already exists.")
        index = vsc.get_index(VS_ENDPOINT_NAME, VS_INDEX_NAME)
    else:
        raise

# COMMAND ----------

# Trigger initial sync and check status
index.sync()
print(index.describe())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2B.4 Test retrieval

# COMMAND ----------

# Search for similar examples
results = index.similarity_search(
    query_text="Get average order value per customer segment",
    columns=["id", "description", "pyspark_query"],
    num_results=5,
)

for row in results["result"]["data_array"]:
    print(f"Score: {row[-1]:.3f}")
    print(f"Description: {row[1]}")
    print(f"Query: {row[2][:100]}...")
    print("-" * 50)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2B.5 RAG pipeline — retrieve + generate

# COMMAND ----------

import requests
import json

DATABRICKS_HOST = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
DATABRICKS_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

SYSTEM_PROMPT = """You are a PySpark query generator. Given a natural language requirement, produce a complete, runnable PySpark query.

Follow the exact coding style shown in the reference examples below.

Rules:
- Use PySpark DataFrame API (not RDD)
- Use spark.table() to reference tables
- Import pyspark.sql.functions as F
- Use F.col() for column references
- Output only the code, no explanations
- Include comments only for non-obvious logic"""


def retrieve_examples(requirement: str, top_k: int = 5) -> list[dict]:
    """Retrieve the most similar examples from Vector Search."""
    results = index.similarity_search(
        query_text=requirement,
        columns=["description", "pyspark_query"],
        num_results=top_k,
    )
    examples = []
    for row in results["result"]["data_array"]:
        examples.append({
            "description": row[0],
            "pyspark_query": row[1],
            "score": row[-1],
        })
    return examples


def build_prompt(requirement: str, examples: list[dict]) -> str:
    """Format retrieved examples into the user message."""
    examples_text = ""
    for i, ex in enumerate(examples, 1):
        examples_text += f"--- Example {i} ---\n"
        examples_text += f"Requirement: {ex['description']}\n"
        examples_text += f"PySpark Query:\n```python\n{ex['pyspark_query']}\n```\n\n"

    return f"""=== REFERENCE EXAMPLES ===

{examples_text}
=== NEW REQUIREMENT ===

{requirement}

Generate the PySpark query:"""


def generate_query(requirement: str, top_k: int = 5) -> str:
    """Full RAG pipeline: retrieve similar examples, then generate."""
    # 1. Retrieve
    examples = retrieve_examples(requirement, top_k)

    # 2. Build prompt
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
# MAGIC ## 2B.6 Evaluate on validation set

# COMMAND ----------

val_df = spark.table(f"{CATALOG}.{SCHEMA}.validation_set").collect()

results = []
for row in val_df:
    generated = generate_query(row["description"])
    results.append({
        "description": row["description"],
        "expected": row["pyspark_query"],
        "generated": generated,
    })

results_df = spark.createDataFrame(results)
results_df.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.rag_eval_results")
display(results_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Done!
# MAGIC
# MAGIC Your RAG pipeline is live. To use it from other notebooks:
# MAGIC ```python
# MAGIC # In any Databricks notebook:
# MAGIC %run ./02b_rag
# MAGIC result = generate_query("your requirement here")
# MAGIC print(result)
# MAGIC ```
