"""Load JSONL training data, embed descriptions, and build a FAISS index."""

import json
import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from shared.config import settings
from shared.data_validator import validate_jsonl


def build_index(
    data_path: Path | None = None,
    index_dir: Path | None = None,
    model_name: str | None = None,
) -> None:
    """Build a FAISS index from training data and save to disk.

    Saves:
        - index.faiss: the FAISS index
        - metadata.pkl: list of dicts with description + pyspark_code for each example
    """
    data_path = data_path or settings.training_data_path
    index_dir = index_dir or settings.faiss_index_path
    model_name = model_name or settings.embedding_model

    # Validate and load examples
    examples = validate_jsonl(data_path)
    print(f"Loaded {len(examples)} training examples from {data_path}")

    # Embed descriptions
    model = SentenceTransformer(model_name)
    descriptions = [ex.description for ex in examples]
    embeddings = model.encode(descriptions, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype=np.float32)

    # Build FAISS index (inner product on normalized vectors = cosine similarity)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    # Save index and metadata
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(index_dir / "index.faiss"))

    metadata = [
        {"description": ex.description, "pyspark_code": ex.pyspark_code}
        for ex in examples
    ]
    with open(index_dir / "metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    print(f"Index saved to {index_dir} (dimension={dimension}, vectors={index.ntotal})")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build FAISS index from training data")
    parser.add_argument("--data", help="Path to training JSONL")
    parser.add_argument("--index-dir", help="Output directory for FAISS index")
    args = parser.parse_args()

    build_index(
        data_path=Path(args.data) if args.data else None,
        index_dir=Path(args.index_dir) if args.index_dir else None,
    )


if __name__ == "__main__":
    main()
