# Databricks notebook source
# MAGIC %md
# MAGIC # Step 2B-ii: RAG with Databricks Vector Search (Preferred)
# MAGIC
# MAGIC Uses **Databricks Vector Search** for managed embeddings and vector index.
# MAGIC Auto-syncs with your Delta table — no re-embedding on notebook restart.
# MAGIC
# MAGIC **Requirements:**
# MAGIC - `databricks-vectorsearch` package (`%pip install databricks-vectorsearch`)
# MAGIC - Unity Catalog enabled workspace
# MAGIC - Permission to create Vector Search endpoints
# MAGIC
# MAGIC **Prerequisites:** Run notebook `01_data_prep` first.

# COMMAND ----------

# DBTITLE 1,Configuration - EDIT THESE
CATALOG = "your_catalog"
SCHEMA = "pyspark_gen"
SOURCE_TABLE = f"{CATALOG}.{SCHEMA}.training_examples"

# Vector search
VS_ENDPOINT_NAME = "pyspark_gen_vs_endpoint"
VS_INDEX_NAME = f"{CATALOG}.{SCHEMA}.examples_index"

# Foundation model for generation (choose one available in your workspace)
# Check: Workspace > Serving > Foundation Model APIs
GENERATION_MODEL = "databricks-meta-llama-3-1-70b-instruct"  # or "databricks-dbrx-instruct"

# Number of similar examples to retrieve
TOP_K = 5

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Install dependencies

# COMMAND ----------

%pip install databricks-vectorsearch

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# Re-declare config after restartPython (kernel state is cleared)
CATALOG = "your_catalog"
SCHEMA = "pyspark_gen"
SOURCE_TABLE = f"{CATALOG}.{SCHEMA}.training_examples"
VS_ENDPOINT_NAME = "pyspark_gen_vs_endpoint"
VS_INDEX_NAME = f"{CATALOG}.{SCHEMA}.examples_index"
GENERATION_MODEL = "databricks-meta-llama-3-1-70b-instruct"
TOP_K = 5

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Enable Change Data Feed on the source table
# MAGIC
# MAGIC Required for Vector Search to sync from Delta.

# COMMAND ----------

spark.sql(f"ALTER TABLE {SOURCE_TABLE} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
print(f"CDF enabled on {SOURCE_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Create Vector Search endpoint

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient()

try:
    vsc.create_endpoint(name=VS_ENDPOINT_NAME, endpoint_type="STANDARD")
    print(f"Creating endpoint '{VS_ENDPOINT_NAME}'... (takes 5-10 minutes)")
except Exception as e:
    if "already exists" in str(e):
        print(f"Endpoint '{VS_ENDPOINT_NAME}' already exists.")
    else:
        raise

# COMMAND ----------

# Wait for endpoint to be ready (re-run until ONLINE)
endpoint = vsc.get_endpoint(VS_ENDPOINT_NAME)
print(f"Endpoint status: {endpoint}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Create Vector Search index
# MAGIC
# MAGIC Embeds `intake_description` using Databricks' built-in embedding model.
# MAGIC Index persists and auto-syncs — no re-embedding on restart.

# COMMAND ----------

try:
    index = vsc.create_delta_sync_index(
        endpoint_name=VS_ENDPOINT_NAME,
        index_name=VS_INDEX_NAME,
        source_table_name=SOURCE_TABLE,
        pipeline_type="TRIGGERED",
        primary_key="id",
        embedding_source_columns=["intake_description"],
        embedding_model_endpoint_name="databricks-bge-large-en",
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
# MAGIC ## 5. Test retrieval

# COMMAND ----------

def retrieve_examples(requirement: str, top_k: int = TOP_K) -> list:
    """Find the top-k most similar intake examples using Databricks Vector Search."""
    results = index.similarity_search(
        query_text=requirement,
        columns=["intake_number", "title", "intake_description", "pyspark_query"],
        num_results=top_k,
    )
    examples = []
    for row in results["result"]["data_array"]:
        examples.append({
            "intake_number": row[0],
            "title": row[1],
            "intake_description": row[2],
            "pyspark_query": row[3],
            "score": row[-1],
        })
    return examples

# COMMAND ----------

# Test retrieval
test_results = retrieve_examples("Get average order value per customer segment")

for r in test_results:
    print(f"Score: {r['score']:.3f} | {r['intake_number']} | {r['title']}")
    print(f"  {r['intake_description'][:100]}...")
    print("-" * 50)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. RAG pipeline — retrieve + generate

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
    # 1. Retrieve similar examples via Vector Search
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
# MAGIC ## 7. Evaluate on validation set

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
results_df.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.rag_vs_eval_results")
display(results_df)
