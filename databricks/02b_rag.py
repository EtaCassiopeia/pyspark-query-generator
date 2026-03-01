# Databricks notebook source
# MAGIC %md
# MAGIC # Step 2B: RAG Approach with Databricks Vector Search
# MAGIC
# MAGIC Uses **Databricks Vector Search** to store your 300 intake examples and
# MAGIC **Foundation Model APIs** (DBRX / Llama / Mixtral) to generate queries.
# MAGIC
# MAGIC Searches by `intake_description` similarity, retrieves full context
# MAGIC (intake_number, title, description, query), and passes to LLM.
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

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2B.1 Enable Change Data Feed on the source table

# COMMAND ----------

spark.sql(f"ALTER TABLE {SOURCE_TABLE} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
print(f"CDF enabled on {SOURCE_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2B.2 Create Vector Search endpoint

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
# MAGIC ## 2B.3 Create Vector Search index
# MAGIC
# MAGIC Embeds `intake_description` for semantic search.
# MAGIC Retrieves all columns (intake_number, title, description, query).

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

index.sync()
print(index.describe())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2B.4 Test retrieval

# COMMAND ----------

results = index.similarity_search(
    query_text="Get average order value per customer segment",
    columns=["intake_number", "title", "intake_description", "pyspark_query"],
    num_results=5,
)

for row in results["result"]["data_array"]:
    print(f"Score: {row[-1]:.3f}")
    print(f"Intake: {row[0]} | Title: {row[1]}")
    print(f"Description: {row[2][:100]}...")
    print(f"Query: {row[3][:80]}...")
    print("-" * 50)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2B.5 RAG pipeline — retrieve + generate

# COMMAND ----------

import requests
import json

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


def retrieve_examples(requirement: str, top_k: int = 5) -> list[dict]:
    """Retrieve the most similar intake examples from Vector Search."""
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


def build_prompt(requirement: str, examples: list[dict]) -> str:
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


def generate_query(requirement: str, top_k: int = 5) -> str:
    """Full RAG pipeline: retrieve similar intakes, then generate."""
    examples = retrieve_examples(requirement, top_k)

    user_message = build_prompt(requirement, examples)

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
    generated = generate_query(row["intake_description"])
    results.append({
        "intake_number": row["intake_number"],
        "title": row["title"],
        "description": row["intake_description"],
        "expected": row["pyspark_query"],
        "generated": generated,
    })

results_df = spark.createDataFrame(results)
results_df.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.rag_eval_results")
display(results_df)
