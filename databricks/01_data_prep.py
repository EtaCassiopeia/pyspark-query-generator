# Databricks notebook source
# MAGIC %md
# MAGIC # Step 1: Data Preparation
# MAGIC
# MAGIC Prepare the 300 (description, query) pairs for both fine-tuning and RAG.
# MAGIC Run this notebook first — both approaches depend on it.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.1 Upload your CSV
# MAGIC
# MAGIC Upload your CSV to Databricks (Unity Catalog volume, DBFS, or workspace files).
# MAGIC Expected columns: `description`, `pyspark_query`
# MAGIC
# MAGIC Option A: Use the UI — click "File > Upload Data" in the workspace
# MAGIC Option B: Use the code below if the CSV is already in a volume

# COMMAND ----------

# DBTITLE 1,Configuration - EDIT THESE
CATALOG = "your_catalog"          # your Unity Catalog name
SCHEMA = "pyspark_gen"            # schema to create
TABLE_NAME = "training_examples"  # table to store examples
CSV_PATH = "/Volumes/your_catalog/your_schema/your_volume/your_data.csv"  # path to uploaded CSV

# COMMAND ----------

# DBTITLE 1,Create schema
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

# COMMAND ----------

# DBTITLE 1,Load CSV and save as Delta table
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType

# Load raw CSV
raw_df = spark.read.option("header", True).option("multiLine", True).option("escape", '"').csv(CSV_PATH)

# Validate required columns
assert "description" in raw_df.columns, "CSV must have 'description' column"
assert "pyspark_query" in raw_df.columns, "CSV must have 'pyspark_query' column"

# Clean
df = raw_df.select(
    F.monotonically_increasing_id().alias("id"),
    F.trim(F.col("description")).alias("description"),
    F.trim(F.col("pyspark_query")).alias("pyspark_query"),
).filter(
    F.col("description").isNotNull() & F.col("pyspark_query").isNotNull()
).filter(
    (F.length("description") > 10) & (F.length("pyspark_query") > 10)
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

SYSTEM_PROMPT = """You are a PySpark query generator. Given a natural language requirement, produce a complete, runnable PySpark query.

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
    """Convert a row to OpenAI-compatible chat format (also used by Databricks fine-tuning)."""
    return json.dumps({
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": row["description"]},
            {"role": "assistant", "content": row["pyspark_query"]},
        ]
    })

# Collect and write JSONL files
for split_name in ["training_set", "validation_set"]:
    split_df = spark.table(f"{CATALOG}.{SCHEMA}.{split_name}")
    rows = split_df.select("description", "pyspark_query").collect()

    jsonl_path = f"/Volumes/{CATALOG}/{SCHEMA}/data/{split_name}.jsonl"
    lines = [to_chat_jsonl(row) for row in rows]

    # Write via dbutils
    dbutils.fs.put(jsonl_path, "\n".join(lines), overwrite=True)
    print(f"Wrote {len(lines)} examples to {jsonl_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Done!
# MAGIC
# MAGIC You now have:
# MAGIC - Delta table `training_examples` with all 300 rows
# MAGIC - `training_set` and `validation_set` tables (85/15 split)
# MAGIC - JSONL files ready for fine-tuning
# MAGIC
# MAGIC **Next:** Run notebook `02a_finetune` for fine-tuning OR `02b_rag` for RAG approach
