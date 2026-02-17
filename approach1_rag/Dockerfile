FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir ".[rag]"

COPY shared/ shared/
COPY approach1_rag/ approach1_rag/
COPY data/ data/

EXPOSE 8080

CMD ["python", "-m", "approach1_rag.api"]
