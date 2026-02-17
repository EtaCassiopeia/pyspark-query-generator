"""FastAPI server for the RAG-based PySpark query generator."""

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI, HTTPException

from shared.data_models import GenerationRequest, GenerationResponse

from .generator import RAGGenerator
from .llm_client import LLMClient
from .retriever import Retriever

generator: RAGGenerator | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global generator
    retriever = Retriever()
    llm = LLMClient()
    generator = RAGGenerator(retriever=retriever, llm=llm)
    print("RAG generator initialized")
    yield


app = FastAPI(
    title="PySpark Query Generator — RAG",
    description="Generate PySpark code from natural language using RAG with local LLM",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict[str, str]:
    assert generator is not None
    llm_ok = await generator.llm.health_check()
    status = "healthy" if llm_ok else "degraded"
    return {"status": status, "approach": "rag", "model": generator.llm.model}


@app.post("/generate", response_model=GenerationResponse)
async def generate(request: GenerationRequest) -> GenerationResponse:
    if generator is None:
        raise HTTPException(status_code=503, detail="Generator not initialized")
    return await generator.generate(request)


def main() -> None:
    import uvicorn

    uvicorn.run("approach1_rag.api:app", host="0.0.0.0", port=8080, reload=True)


if __name__ == "__main__":
    main()
