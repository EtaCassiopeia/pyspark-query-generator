# Databricks notebook source
# MAGIC %md
# MAGIC # Step 2A (Manual): Fine-Tune with Hugging Face Transformers
# MAGIC
# MAGIC Bypasses `fm.create` entirely — loads data via `spark.sql()` (which you have
# MAGIC access to) and fine-tunes directly on the cluster GPU using LoRA.
# MAGIC
# MAGIC **No table permission issues** — all data loading happens in your notebook.
# MAGIC
# MAGIC **Cluster requirements:**
# MAGIC - GPU cluster (A10G, A100, or V100) — single node is fine for 8B model with LoRA
# MAGIC - Databricks Runtime ML (e.g., 14.3 LTS ML or newer)
# MAGIC
# MAGIC **Prerequisites:** Run notebook `01_data_prep` first.

# COMMAND ----------

# DBTITLE 1,Configuration - EDIT THESE
CATALOG = "your_catalog"
SCHEMA = "pyspark_gen"

# Base model — 8B is the sweet spot for single-GPU LoRA fine-tuning
BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# Where to save the fine-tuned model (MLflow)
REGISTERED_MODEL_NAME = f"{CATALOG}.{SCHEMA}.pyspark_query_generator"

# Training parameters
EPOCHS = 3
LEARNING_RATE = 2e-4
BATCH_SIZE = 4
MAX_SEQ_LENGTH = 2048

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Install dependencies

# COMMAND ----------

%pip install peft trl accelerate bitsandbytes

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# Re-declare config after restart
CATALOG = "your_catalog"
SCHEMA = "pyspark_gen"
BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
REGISTERED_MODEL_NAME = f"{CATALOG}.{SCHEMA}.pyspark_query_generator"
EPOCHS = 3
LEARNING_RATE = 2e-4
BATCH_SIZE = 4
MAX_SEQ_LENGTH = 2048

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load data via spark.sql (no permission issues)

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


def load_split(split_name):
    """Read data via spark.sql and convert to list of chat messages."""
    rows = spark.sql(
        f"SELECT intake_number, title, intake_description, pyspark_query FROM {CATALOG}.{SCHEMA}.{split_name}"
    ).collect()

    examples = []
    for row in rows:
        user_content = f"Intake: {row['intake_number']}\nTitle: {row['title']}\nDescription: {row['intake_description']}"
        examples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": row["pyspark_query"]},
            ]
        })
    return examples


train_data = load_split("training_set")
val_data = load_split("validation_set")

print(f"Training examples:   {len(train_data)}")
print(f"Validation examples: {len(val_data)}")
print(f"\nSample message:\n{json.dumps(train_data[0]['messages'][1], indent=2)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Convert to Hugging Face Dataset

# COMMAND ----------

from datasets import Dataset

train_dataset = Dataset.from_list(train_data)
val_dataset = Dataset.from_list(val_data)

print(train_dataset)
print(val_dataset)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Load model with QLoRA (4-bit quantization)
# MAGIC
# MAGIC QLoRA lets you fine-tune an 8B model on a single GPU by:
# MAGIC - Loading the base model in 4-bit precision (~4GB VRAM instead of ~16GB)
# MAGIC - Training only small adapter layers (LoRA) on top

# COMMAND ----------

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

print(f"Loading tokenizer for {BASE_MODEL}...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print(f"Loading model {BASE_MODEL} in 4-bit...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

model = prepare_model_for_kbit_training(model)
print("Model loaded.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Configure LoRA adapters

# COMMAND ----------

lora_config = LoraConfig(
    r=16,                          # rank — higher = more capacity, more VRAM
    lora_alpha=32,                 # scaling factor
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # attention layers
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Format data for training

# COMMAND ----------

def format_chat(example):
    """Apply the chat template to convert messages to a single text string."""
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}

train_formatted = train_dataset.map(format_chat)
val_formatted = val_dataset.map(format_chat)

print("Sample formatted text (first 500 chars):")
print(train_formatted[0]["text"][:500])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Train

# COMMAND ----------

from trl import SFTTrainer
from transformers import TrainingArguments

output_dir = "/tmp/pyspark_gen_finetune"

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    bf16=True,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    report_to="mlflow",
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_formatted,
    eval_dataset=val_formatted,
    tokenizer=tokenizer,
    max_seq_length=MAX_SEQ_LENGTH,
)

print("Starting training...")
trainer.train()
print("Training complete!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Save and register model

# COMMAND ----------

import mlflow

# Save the LoRA adapter
adapter_path = f"{output_dir}/final_adapter"
model.save_pretrained(adapter_path)
tokenizer.save_pretrained(adapter_path)
print(f"Adapter saved to {adapter_path}")

# Log to MLflow and register in Unity Catalog
mlflow.set_registry_uri("databricks-uc")

with mlflow.start_run(run_name="pyspark_query_generator_lora") as run:
    mlflow.log_params({
        "base_model": BASE_MODEL,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "lora_r": 16,
        "lora_alpha": 32,
        "training_examples": len(train_data),
    })

    # Log the adapter as an MLflow model
    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=mlflow.pyfunc.PythonModel(),
        artifacts={"adapter": adapter_path},
        registered_model_name=REGISTERED_MODEL_NAME,
    )

    print(f"Model registered as {REGISTERED_MODEL_NAME}")
    print(f"MLflow run ID: {run.info.run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Test the fine-tuned model (local inference on this cluster)

# COMMAND ----------

from peft import PeftModel

def generate_query(intake_number: str, title: str, intake_description: str) -> str:
    """Generate a PySpark query using the fine-tuned model."""
    user_content = f"Intake: {intake_number}\nTitle: {title}\nDescription: {intake_description}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the generated part (skip the input prompt)
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)

# COMMAND ----------

# DBTITLE 1,Test it
result = generate_query(
    intake_number="INT-NEW-001",
    title="Revenue by Product Category",
    intake_description="Get total revenue by product category for the last 30 days"
)
print(result)

# COMMAND ----------

# DBTITLE 1,Test with more examples
test_cases = [
    ("INT-TEST-001", "Top Customers", "Find the top 5 customers by total spend in 2024"),
    ("INT-TEST-002", "Rolling DAU", "Calculate the rolling 7-day average of daily active users"),
    ("INT-TEST-003", "Inventory Reorder", "Join inventory with sales data and find items below reorder threshold"),
]

for intake_num, title, desc in test_cases:
    print(f"\n{'='*60}")
    print(f"[{intake_num}] {title}")
    print(f"Description: {desc}")
    print(f"{'='*60}\n")
    result = generate_query(intake_num, title, desc)
    print(result)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Evaluate on validation set

# COMMAND ----------

val_rows = spark.sql(f"SELECT * FROM {CATALOG}.{SCHEMA}.validation_set").collect()

results = []
for row in val_rows:
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

# COMMAND ----------

# MAGIC %md
# MAGIC ## Optional: Deploy as serving endpoint
# MAGIC
# MAGIC If you want to serve this model as an API (like `fm.create` would),
# MAGIC you can deploy it via Model Serving after registering it above.
# MAGIC
# MAGIC Go to **Serving** in the sidebar > **Create serving endpoint** >
# MAGIC select your registered model `pyspark_query_generator`.
