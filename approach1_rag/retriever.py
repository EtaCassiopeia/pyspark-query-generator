"""FAISS-based retriever: embed query and find top-k similar training examples."""

import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from shared.config import settings


class Retriever:
    """Wraps a FAISS index and metadata for similarity search."""

    def __init__(
        self,
        index_dir: Path | None = None,
        model_name: str | None = None,
    ) -> None:
        index_dir = Path(index_dir or settings.faiss_index_path)
        model_name = model_name or settings.embedding_model

        self.index = faiss.read_index(str(index_dir / "index.faiss"))
        with open(index_dir / "metadata.pkl", "rb") as f:
            self.metadata: list[dict[str, str]] = pickle.load(f)

        self.model = SentenceTransformer(model_name)

    def search(self, query: str, top_k: int | None = None) -> list[dict[str, str]]:
        """Return the top-k most similar training examples for a query.

        Each result dict has keys: description, pyspark_code, score.
        """
        top_k = top_k or settings.rag_top_k

        embedding = self.model.encode([query], normalize_embeddings=True)
        embedding = np.array(embedding, dtype=np.float32)

        scores, indices = self.index.search(embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            entry = self.metadata[idx].copy()
            entry["score"] = float(score)
            results.append(entry)

        return results
