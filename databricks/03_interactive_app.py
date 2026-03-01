# Databricks notebook source
# MAGIC %md
# MAGIC # Step 3: Interactive App
# MAGIC
# MAGIC A simple UI within Databricks to use the query generator interactively.
# MAGIC Supports three backends:
# MAGIC
# MAGIC | Mode | Backend | When to use |
# MAGIC |---|---|---|
# MAGIC | `rag_faiss` | FAISS + sentence-transformers | **Default.** No extra permissions needed |
# MAGIC | `rag_databricks` | Databricks Vector Search | When you have `databricks-vectorsearch` access |
# MAGIC | `finetune` | Fine-tuned model serving endpoint | After running `02a_finetune` and deploying |
# MAGIC
# MAGIC **Prerequisites:** Run `01_data_prep` first, then the notebook matching your chosen mode.

# COMMAND ----------

# DBTITLE 1,Select mode — EDIT THIS
# Set MODE to one of: "rag_faiss", "rag_databricks", "finetune"
MODE = "rag_faiss"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load the selected backend

# COMMAND ----------

if MODE == "rag_faiss":
    pass  # loaded via %run below

# COMMAND ----------

# MAGIC %run ./02b_rag_faiss

# COMMAND ----------

# MAGIC %md
# MAGIC ### Override functions if using a different backend
# MAGIC
# MAGIC The cell above always runs `02b_rag_faiss`. If a different MODE is selected,
# MAGIC the cells below redefine `retrieve_examples` and `generate_query` to match.

# COMMAND ----------

if MODE == "rag_databricks":
    # --- Databricks Vector Search backend ---
    # Requires: %pip install databricks-vectorsearch (run in 02b_rag_databricks first)
    from databricks.vector_search.client import VectorSearchClient

    CATALOG = "your_catalog"
    SCHEMA = "pyspark_gen"
    VS_ENDPOINT_NAME = "pyspark_gen_vs_endpoint"
    VS_INDEX_NAME = f"{CATALOG}.{SCHEMA}.examples_index"

    vsc = VectorSearchClient()
    index = vsc.get_index(VS_ENDPOINT_NAME, VS_INDEX_NAME)

    def retrieve_examples(requirement, top_k=5):
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

    # generate_query and build_prompt from 02b_rag_faiss still work — same signature

elif MODE == "finetune":
    # --- Fine-tuned model backend ---
    import requests

    CATALOG = "your_catalog"
    SCHEMA = "pyspark_gen"
    ENDPOINT_NAME = "pyspark-query-generator"

    DATABRICKS_HOST = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
    DATABRICKS_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

    FT_SYSTEM_PROMPT = """You are a PySpark query generator. Given an intake ticket with a title and description, produce a complete, runnable PySpark query.

Rules:
- Use PySpark DataFrame API (not RDD)
- Use spark.table() to reference tables
- Import pyspark.sql.functions as F
- Use F.col() for column references
- Output only the code, no explanations
- Include comments only for non-obvious logic"""

    def generate_query_finetune(intake_number, title, intake_description):
        user_content = f"Intake: {intake_number}\nTitle: {title}\nDescription: {intake_description}"
        response = requests.post(
            f"{DATABRICKS_HOST}/serving-endpoints/{ENDPOINT_NAME}/invocations",
            headers={"Authorization": f"Bearer {DATABRICKS_TOKEN}"},
            json={
                "messages": [
                    {"role": "system", "content": FT_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                "max_tokens": 1024,
                "temperature": 0.1,
            },
        )
        return response.json()["choices"][0]["message"]["content"]

print(f"Active mode: {MODE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Interactive query generator

# COMMAND ----------

# DBTITLE 1,Widgets
dbutils.widgets.text("requirement", "", "Enter your requirement")

if MODE != "finetune":
    dbutils.widgets.dropdown("num_examples", "5", ["3", "5", "7", "10"], "Reference examples")

if MODE == "finetune":
    dbutils.widgets.text("intake_number", "INT-NEW", "Intake number")
    dbutils.widgets.text("title", "", "Short title")

# COMMAND ----------

# DBTITLE 1,Generate PySpark query
requirement = dbutils.widgets.get("requirement")

if not requirement:
    print("Enter a requirement in the widget above and re-run this cell.")

elif MODE == "finetune":
    # --- Fine-tuned model: no retrieval, direct generation ---
    intake_number = dbutils.widgets.get("intake_number")
    title = dbutils.widgets.get("title")

    print(f"Intake: {intake_number}")
    print(f"Title: {title}")
    print(f"Description: {requirement}\n")

    print(f"{'='*60}")
    print("GENERATED PYSPARK QUERY:")
    print(f"{'='*60}\n")
    result = generate_query_finetune(intake_number, title, requirement)
    print(result)

else:
    # --- RAG mode (faiss or databricks): retrieve + generate ---
    num_examples = int(dbutils.widgets.get("num_examples"))

    print(f"Requirement: {requirement}\n")
    print(f"Retrieving {num_examples} similar examples...\n")

    examples = retrieve_examples(requirement, num_examples)
    print("Similar past intakes found:")
    for i, ex in enumerate(examples, 1):
        print(f"  {i}. (score: {ex['score']:.3f}) [{ex['intake_number']}] {ex['title']} — {ex['intake_description'][:80]}")

    print(f"\n{'='*60}")
    print("GENERATED PYSPARK QUERY:")
    print(f"{'='*60}\n")
    result = generate_query(requirement, num_examples)
    print(result)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Batch processing
# MAGIC
# MAGIC Process a list of new requirements at once.

# COMMAND ----------

# DBTITLE 1,Batch generate
new_requirements = [
    {"intake_number": "INT-BATCH-001", "title": "Customer Lifetime Value",    "intake_description": "Calculate customer lifetime value based on order history"},
    {"intake_number": "INT-BATCH-002", "title": "Frequently Bought Together", "intake_description": "Find products frequently bought together"},
    {"intake_number": "INT-BATCH-003", "title": "Daily Sales YoY",            "intake_description": "Generate a daily sales summary report with YoY comparison"},
]

batch_results = []
for req in new_requirements:
    if MODE == "finetune":
        query = generate_query_finetune(req["intake_number"], req["title"], req["intake_description"])
    else:
        query = generate_query(req["intake_description"])

    batch_results.append({
        "intake_number": req["intake_number"],
        "title": req["title"],
        "requirement": req["intake_description"],
        "generated_query": query,
    })

batch_df = spark.createDataFrame(batch_results)
display(batch_df)
