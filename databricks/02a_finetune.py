# Databricks notebook source
# MAGIC %md
# MAGIC # Step 2A: Fine-Tune a Model on Databricks
# MAGIC
# MAGIC Uses **Mosaic AI Fine-Tuning** to train a model on your 300 examples.
# MAGIC
# MAGIC Reads data via `SELECT *` (which you have access to), converts to JSONL,
# MAGIC saves to DBFS, and passes file paths to `fm.create` (avoids table permission issues).
# MAGIC
# MAGIC Supported base models (check your Databricks workspace for latest list):
# MAGIC - `meta-llama/Meta-Llama-3.1-8B-Instruct` (recommended - good at code)
# MAGIC - `meta-llama/Meta-Llama-3.1-70B-Instruct` (better quality, more expensive)
# MAGIC - `mistralai/Mistral-7B-Instruct-v0.3`
# MAGIC - `codellama/CodeLlama-13b-Instruct-hf`
# MAGIC
# MAGIC **Prerequisites:** Run notebook `01_data_prep` first.

# COMMAND ----------

# DBTITLE 1,Configuration - EDIT THESE
CATALOG = "your_catalog"
SCHEMA = "pyspark_gen"

# Pick your base model — start with the 8B, it's fast to train and good at code
BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# Fine-tuned model will be registered here
REGISTERED_MODEL_NAME = f"{CATALOG}.{SCHEMA}.pyspark_query_generator"

# DBFS paths for JSONL files (fm.create can read these without table permissions)
TRAIN_JSONL_PATH = f"dbfs:/FileStore/pyspark_gen/training_set.jsonl"
VAL_JSONL_PATH = f"dbfs:/FileStore/pyspark_gen/validation_set.jsonl"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2A.1 Read data via SELECT and export to JSONL
# MAGIC
# MAGIC Since `fm.create` cannot access the tables directly, we read the data
# MAGIC ourselves (which works) and save as JSONL files on DBFS.

# COMMAND ----------

import json

SYSTEM_PROMPT = """You are a PySpark query generator. Given an intake ticket with a title and description, produce a complete, runnable PySpark query.

Rules:
- Use PySpark DataFrame API (not RDD)
- Use spark.table() to reference tables
- Import pyspark.sql.functions as F
- Use F.col() for column references
- Output only the code, no explanations
- Include comments only for non-obvious logic"""


def to_chat_jsonl(row):
    """Convert a row to chat completion format for fine-tuning."""
    user_content = f"Intake: {row['intake_number']}\nTitle: {row['title']}\nDescription: {row['intake_description']}"
    return json.dumps({
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": row["pyspark_query"]},
        ]
    })

# COMMAND ----------

# DBTITLE 1,Read via SELECT and write JSONL to DBFS
for split_name, dbfs_path in [("training_set", TRAIN_JSONL_PATH), ("validation_set", VAL_JSONL_PATH)]:
    rows = spark.sql(f"SELECT intake_number, title, intake_description, pyspark_query FROM {CATALOG}.{SCHEMA}.{split_name}").collect()
    lines = [to_chat_jsonl(row) for row in rows]
    dbutils.fs.put(dbfs_path, "\n".join(lines), overwrite=True)
    print(f"Wrote {len(lines)} examples to {dbfs_path}")

# COMMAND ----------

# Verify the files are readable
for path in [TRAIN_JSONL_PATH, VAL_JSONL_PATH]:
    head = dbutils.fs.head(path, 500)
    print(f"\n--- {path} (first 500 chars) ---")
    print(head)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2A.2 Launch fine-tuning run
# MAGIC
# MAGIC Uses DBFS file paths instead of table references to avoid permission issues.

# COMMAND ----------

from databricks.model_training import foundation_model as fm

# Launch the fine-tuning run using JSONL files on DBFS
run = fm.create(
    model=BASE_MODEL,
    train_data_path=TRAIN_JSONL_PATH,
    eval_data_path=VAL_JSONL_PATH,
    register_to=REGISTERED_MODEL_NAME,
    training_duration="5ep",  # 5 epochs (good for 300 examples)
    learning_rate="5e-6",     # conservative LR for small dataset
)

print(f"Fine-tuning run launched!")
print(f"Run name: {run.name}")
print(f"Track progress in the Experiments UI")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2A.3 Monitor training
# MAGIC
# MAGIC Training takes ~30-90 minutes depending on the model size and cluster.
# MAGIC You can monitor in the **Experiments** tab, or poll here:

# COMMAND ----------

# Check status (re-run this cell periodically)
status = fm.get(run.name)
print(f"Status: {status.status}")
print(f"Details: {status}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2A.4 Deploy as serving endpoint
# MAGIC
# MAGIC Once training completes, deploy the model as an API endpoint.

# COMMAND ----------

import requests

# Get workspace URL and token
DATABRICKS_HOST = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
DATABRICKS_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

ENDPOINT_NAME = "pyspark-query-generator"

# Get the latest model version
from mlflow import MlflowClient
client = MlflowClient()
latest_version = client.get_registered_model(REGISTERED_MODEL_NAME).latest_versions[0].version

# Create serving endpoint
endpoint_config = {
    "name": ENDPOINT_NAME,
    "config": {
        "served_entities": [
            {
                "entity_name": REGISTERED_MODEL_NAME,
                "entity_version": latest_version,
                "min_provisioned_throughput": 0,
                "max_provisioned_throughput": 100,
            }
        ]
    },
}

response = requests.post(
    f"{DATABRICKS_HOST}/api/2.0/serving-endpoints",
    headers={"Authorization": f"Bearer {DATABRICKS_TOKEN}"},
    json=endpoint_config,
)
print(response.json())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2A.5 Test the fine-tuned model

# COMMAND ----------

# DBTITLE 1,Query the fine-tuned model
def generate_query(intake_number: str, title: str, intake_description: str) -> str:
    """Call the fine-tuned model serving endpoint."""
    user_content = f"Intake: {intake_number}\nTitle: {title}\nDescription: {intake_description}"
    response = requests.post(
        f"{DATABRICKS_HOST}/serving-endpoints/{ENDPOINT_NAME}/invocations",
        headers={"Authorization": f"Bearer {DATABRICKS_TOKEN}"},
        json={
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "max_tokens": 1024,
            "temperature": 0.1,
        },
    )
    return response.json()["choices"][0]["message"]["content"]


# Test it
result = generate_query(
    intake_number="INT-NEW-001",
    title="Revenue by Product Category",
    intake_description="Get total revenue by product category for the last 30 days"
)
print(result)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2A.6 Evaluate on validation set

# COMMAND ----------

val_df = spark.sql(f"SELECT * FROM {CATALOG}.{SCHEMA}.validation_set").collect()

results = []
for row in val_df:
    generated = generate_query(row["intake_number"], row["title"], row["intake_description"])
    results.append({
        "intake_number": row["intake_number"],
        "title": row["title"],
        "description": row["intake_description"],
        "expected": row["pyspark_query"],
        "generated": generated,
    })

results_df = spark.createDataFrame(results)
results_df.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.finetune_eval_results")
display(results_df)
