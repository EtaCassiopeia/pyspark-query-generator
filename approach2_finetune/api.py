"""FastAPI server for the fine-tuned model approach (same interface as RAG)."""

import re
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

import httpx
from fastapi import FastAPI, HTTPException

from shared.config import settings
from shared.data_models import GenerationRequest, GenerationResponse
from shared.prompt_templates import SYSTEM_PROMPT

client: httpx.AsyncClient | None = None
model_name: str = ""


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global client, model_name
    model_name = settings.base_model
    client = httpx.AsyncClient(timeout=120.0)
    print(f"Fine-tune API initialized, targeting vLLM at {settings.vllm_host}:{settings.vllm_port}")
    yield
    if client:
        await client.aclose()


app = FastAPI(
    title="PySpark Query Generator — Fine-tuned",
    description="Generate PySpark code from natural language using a fine-tuned model",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict[str, str]:
    vllm_url = f"http://{settings.vllm_host}:{settings.vllm_port}"
    try:
        assert client is not None
        resp = await client.get(f"{vllm_url}/v1/models")
        status = "healthy" if resp.status_code == 200 else "degraded"
    except httpx.HTTPError:
        status = "degraded"
    return {"status": status, "approach": "finetune", "model": model_name}


@app.post("/generate", response_model=GenerationResponse)
async def generate(request: GenerationRequest) -> GenerationResponse:
    if client is None:
        raise HTTPException(status_code=503, detail="Client not initialized")

    vllm_url = f"http://{settings.vllm_host}:{settings.vllm_port}"
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": request.description},
        ],
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
    }

    resp = await client.post(f"{vllm_url}/v1/chat/completions", json=payload)
    resp.raise_for_status()

    raw = resp.json()["choices"][0]["message"]["content"]

    # Strip markdown code fences if present
    match = re.search(r"```(?:python)?\s*\n(.*?)```", raw, re.DOTALL)
    code = match.group(1).strip() if match else raw.strip()

    return GenerationResponse(
        description=request.description,
        pyspark_code=code,
        model=model_name,
        approach="finetune",
    )


def main() -> None:
    import uvicorn

    uvicorn.run("approach2_finetune.api:app", host="0.0.0.0", port=8081, reload=True)


if __name__ == "__main__":
    main()
