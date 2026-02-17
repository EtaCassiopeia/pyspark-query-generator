"""End-to-end RAG pipeline: retrieve examples, build prompt, generate PySpark code."""

import re

from shared.config import settings
from shared.data_models import GenerationRequest, GenerationResponse
from shared.prompt_templates import SYSTEM_PROMPT, build_rag_prompt

from .llm_client import LLMClient
from .retriever import Retriever


class RAGGenerator:
    """Orchestrates the full RAG generation pipeline."""

    def __init__(self, retriever: Retriever | None = None, llm: LLMClient | None = None) -> None:
        self.retriever = retriever or Retriever()
        self.llm = llm or LLMClient()

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate PySpark code from a natural language description."""
        # 1. Retrieve similar examples
        similar = self.retriever.search(request.description)

        # 2. Build few-shot prompt
        user_prompt = build_rag_prompt(
            description=request.description,
            examples=similar,
        )

        # 3. Call LLM
        raw_output = await self.llm.generate(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )

        # 4. Post-process
        code = _extract_code(raw_output)

        return GenerationResponse(
            description=request.description,
            pyspark_code=code,
            model=self.llm.model,
            approach="rag",
            retrieved_examples=[ex["description"] for ex in similar],
        )


def _extract_code(text: str) -> str:
    """Extract PySpark code from LLM output, stripping markdown fences if present."""
    # Try to extract from code blocks first
    match = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()
