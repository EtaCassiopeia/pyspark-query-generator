"""Pydantic models for training data, requests, and responses."""

from pydantic import BaseModel, Field


class TrainingExample(BaseModel):
    """A single training example mapping a natural language description to PySpark code."""

    description: str = Field(..., description="Natural language description of the desired query")
    pyspark_code: str = Field(..., description="The corresponding PySpark code")
    tables: list[str] = Field(default_factory=list, description="Tables referenced in the query")
    operations: list[str] = Field(
        default_factory=list,
        description="PySpark operations used (e.g., filter, groupBy, join)",
    )
    complexity: str = Field(
        default="medium", description="Query complexity: simple, medium, complex"
    )


class GenerationRequest(BaseModel):
    """Request to generate PySpark code from a natural language description."""

    description: str = Field(..., description="Natural language description of the desired query")
    max_tokens: int = Field(default=2048, ge=64, le=4096)
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)


class GenerationResponse(BaseModel):
    """Response containing the generated PySpark code."""

    description: str = Field(..., description="Original description from the request")
    pyspark_code: str = Field(..., description="Generated PySpark code")
    model: str = Field(..., description="Model used for generation")
    approach: str = Field(..., description="'rag' or 'finetune'")
    retrieved_examples: list[str] = Field(
        default_factory=list,
        description="Descriptions of retrieved examples (RAG only)",
    )
