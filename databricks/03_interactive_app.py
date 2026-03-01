# Databricks notebook source
# MAGIC %md
# MAGIC # Step 3: Interactive App
# MAGIC
# MAGIC A simple UI within Databricks to use the query generator interactively.
# MAGIC Works with either the fine-tuned model (02a) or the RAG approach (02b).
# MAGIC
# MAGIC **Prerequisites:** Run either `02a_finetune` or `02b_rag` first.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Option A: Use with RAG (recommended for POC)

# COMMAND ----------

# MAGIC %run ./02b_rag

# COMMAND ----------

# DBTITLE 1,Interactive widget
dbutils.widgets.text("requirement", "", "Enter your requirement")
dbutils.widgets.dropdown("num_examples", "5", ["3", "5", "7", "10"], "Reference examples")

# COMMAND ----------

requirement = dbutils.widgets.get("requirement")
num_examples = int(dbutils.widgets.get("num_examples"))

if requirement:
    print(f"Requirement: {requirement}\n")
    print(f"Retrieving {num_examples} similar examples...\n")

    # Show retrieved examples
    examples = retrieve_examples(requirement, num_examples)
    print("Similar past tickets found:")
    for i, ex in enumerate(examples, 1):
        print(f"  {i}. (score: {ex['score']:.3f}) {ex['description'][:80]}")

    # Generate query
    print(f"\n{'='*60}")
    print("GENERATED PYSPARK QUERY:")
    print(f"{'='*60}\n")
    result = generate_query(requirement, num_examples)
    print(result)
else:
    print("Enter a requirement in the widget above and re-run this cell.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Option B: Batch processing
# MAGIC
# MAGIC Process a list of new requirements at once.

# COMMAND ----------

# DBTITLE 1,Batch generate from a table of new requirements
# Create a table of new requirements or load from a source
new_requirements = [
    "Calculate customer lifetime value based on order history",
    "Find products frequently bought together",
    "Generate a daily sales summary report with YoY comparison",
]

batch_results = []
for req in new_requirements:
    query = generate_query(req)
    batch_results.append({"requirement": req, "generated_query": query})

batch_df = spark.createDataFrame(batch_results)
display(batch_df)
