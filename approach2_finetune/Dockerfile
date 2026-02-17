FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3.11 python3-pip && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir ".[finetune,rag]" vllm

COPY shared/ shared/
COPY approach2_finetune/ approach2_finetune/
COPY data/ data/

EXPOSE 8081

CMD ["python3", "-m", "approach2_finetune.api"]
