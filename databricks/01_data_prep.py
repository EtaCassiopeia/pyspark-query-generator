# Databricks notebook source
# MAGIC %md
# MAGIC # Step 1: Data Preparation
# MAGIC
# MAGIC Prepare the 300 intake tickets for both fine-tuning and RAG.
# MAGIC Run this notebook first — both approaches depend on it.
# MAGIC
# MAGIC Expected CSV columns:
# MAGIC - `intake_number` — ticket/intake ID (e.g., INT-001)
# MAGIC - `title` — short summary of the request
# MAGIC - `intake_description` — detailed requirement description
# MAGIC - `pyspark_query` — the PySpark code that fulfills the requirement

# COMMAND ----------

# DBTITLE 1,Configuration - EDIT THESE
CATALOG = "your_catalog"          # run SHOW CATALOGS to find yours
SCHEMA = "pyspark_gen"            # schema to create
TABLE_NAME = "training_examples"  # table to store examples
CSV_PATH = "/Volumes/your_catalog/your_schema/your_volume/your_data.csv"  # path to uploaded CSV

# COMMAND ----------

# DBTITLE 1,Create schema
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

# COMMAND ----------

# DBTITLE 1,Load CSV and save as Delta table
from pyspark.sql import functions as F

# Load raw CSV
raw_df = spark.read.option("header", True).option("multiLine", True).option("escape", '"').csv(CSV_PATH)

# Validate required columns
required_cols = {"intake_number", "title", "intake_description", "pyspark_query"}
missing = required_cols - set(raw_df.columns)
assert not missing, f"CSV missing columns: {missing}. Found: {raw_df.columns}"

print(f"CSV columns found: {raw_df.columns}")

# Clean
df = raw_df.select(
    F.monotonically_increasing_id().alias("id"),
    F.trim(F.col("intake_number")).alias("intake_number"),
    F.trim(F.col("title")).alias("title"),
    F.trim(F.col("intake_description")).alias("intake_description"),
    F.trim(F.col("pyspark_query")).alias("pyspark_query"),
).filter(
    F.col("intake_description").isNotNull() & F.col("pyspark_query").isNotNull()
).filter(
    (F.length("intake_description") > 10) & (F.length("pyspark_query") > 10)
)

print(f"Rows after cleaning: {df.count()}")
df.show(5, truncate=80)

# COMMAND ----------

# DBTITLE 1,Save as Delta table
full_table = f"{CATALOG}.{SCHEMA}.{TABLE_NAME}"
df.write.mode("overwrite").saveAsTable(full_table)
print(f"Saved to {full_table}")

# COMMAND ----------

# DBTITLE 1,Train/validation split (85/15)
train_df, val_df = df.randomSplit([0.85, 0.15], seed=42)

train_df.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.training_set")
val_df.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.validation_set")

print(f"Training set:   {train_df.count()} rows")
print(f"Validation set: {val_df.count()} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.2 Prepare fine-tuning format (JSONL for chat completions)

# COMMAND ----------

SYSTEM_PROMPT = """You are a PySpark query generator. Given an intake ticket with a title and description, produce a complete, runnable PySpark query.

Rules:
- Use PySpark DataFrame API (not RDD)
- Use spark.table() to reference tables
- Import pyspark.sql.functions as F
- Use F.col() for column references
- Output only the code, no explanations
- Include comments only for non-obvious logic"""

# COMMAND ----------

# DBTITLE 1,Convert to chat JSONL format
import json

def to_chat_jsonl(row):
    """Convert a row to chat format for fine-tuning."""
    user_content = f"Intake: {row['intake_number']}\nTitle: {row['title']}\nDescription: {row['intake_description']}"
    return json.dumps({
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": row["pyspark_query"]},
        ]
    })

# Collect and write JSONL files to DBFS (accessible without Volume permissions)
for split_name in ["training_set", "validation_set"]:
    split_df = spark.sql(f"SELECT intake_number, title, intake_description, pyspark_query FROM {CATALOG}.{SCHEMA}.{split_name}")
    rows = split_df.collect()

    jsonl_path = f"dbfs:/FileStore/pyspark_gen/{split_name}.jsonl"
    lines = [to_chat_jsonl(row) for row in rows]

    dbutils.fs.put(jsonl_path, "\n".join(lines), overwrite=True)
    print(f"Wrote {len(lines)} examples to {jsonl_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Done!
# MAGIC
# MAGIC You now have:
# MAGIC - Delta table with columns: `id`, `intake_number`, `title`, `intake_description`, `pyspark_query`
# MAGIC - `training_set` and `validation_set` tables (85/15 split)
# MAGIC - JSONL files ready for fine-tuning
# MAGIC
# MAGIC **Next:** Run notebook `02a_finetune` for fine-tuning OR `02b_rag_faiss` / `02b_rag_databricks` for RAG
