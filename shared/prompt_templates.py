"""Prompt templates for RAG and fine-tuning approaches."""

SYSTEM_PROMPT = """You are a PySpark code generator. Given a natural language description of a data transformation or report, produce clean, correct PySpark code.

Rules:
- Use PySpark DataFrame API (not RDD).
- Use `spark.table("schema.table")` to read tables.
- Include necessary imports (from pyspark.sql import functions as F, Window, etc.).
- Write production-quality code: proper column aliases, comments for complex logic.
- Output ONLY the PySpark code, no explanations."""

RAG_FEW_SHOT_TEMPLATE = """Below are examples of natural language descriptions and their corresponding PySpark code.

{examples}

Now generate PySpark code for the following description:

Description: {description}

PySpark Code:"""

RAG_EXAMPLE_TEMPLATE = """Description: {description}
PySpark Code:
{pyspark_code}
---"""


def build_rag_prompt(description: str, examples: list[dict[str, str]]) -> str:
    """Build a RAG prompt with retrieved few-shot examples."""
    formatted_examples = "\n".join(
        RAG_EXAMPLE_TEMPLATE.format(
            description=ex["description"], pyspark_code=ex["pyspark_code"]
        )
        for ex in examples
    )
    return RAG_FEW_SHOT_TEMPLATE.format(examples=formatted_examples, description=description)


# --- Instruction-tuning formats ---

ALPACA_TEMPLATE = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Generate PySpark code for the following:
{description}

### Response:
{pyspark_code}"""

CHATML_TEMPLATE = """<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{description}<|im_end|>
<|im_start|>assistant
{pyspark_code}<|im_end|>"""


def format_for_chatml(description: str, pyspark_code: str = "") -> str:
    """Format a single example in ChatML format for fine-tuning."""
    return CHATML_TEMPLATE.format(
        system_prompt=SYSTEM_PROMPT,
        description=description,
        pyspark_code=pyspark_code,
    )


def format_for_alpaca(description: str, pyspark_code: str = "") -> str:
    """Format a single example in Alpaca format for fine-tuning."""
    return ALPACA_TEMPLATE.format(description=description, pyspark_code=pyspark_code)
