"""
PySpark Query Generator - RAG Engine

Loads (description, query) pairs, embeds them in-memory using a local model,
retrieves the most similar examples for a new request, and builds a prompt
ready to paste into ChatGPT.

No API keys required - embeddings run 100% locally.
"""

import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pyperclip
import textwrap
import sys
import os


class PySparkRAG:
    def __init__(self, csv_path: str, model_name: str = "all-MiniLM-L6-v2", top_k: int = 5):
        """
        Args:
            csv_path:   Path to CSV with columns: description, pyspark_query
            model_name: Sentence-transformer model (runs locally, ~80MB download on first use)
            top_k:      Number of similar examples to retrieve
        """
        self.top_k = top_k
        self.df = self._load_data(csv_path)
        self.model = self._load_model(model_name)
        self.index = self._build_index()

    # ------------------------------------------------------------------ data
    def _load_data(self, csv_path: str) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        required = {"description", "pyspark_query"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"CSV missing columns: {missing}")
        df = df.dropna(subset=["description", "pyspark_query"])
        print(f"Loaded {len(df)} examples from {csv_path}")
        return df.reset_index(drop=True)

    # --------------------------------------------------------------- model
    def _load_model(self, model_name: str) -> SentenceTransformer:
        print(f"Loading embedding model '{model_name}' (local, no API key needed)...")
        model = SentenceTransformer(model_name)
        print("Model ready.")
        return model

    # --------------------------------------------------------------- index
    def _build_index(self) -> faiss.IndexFlatIP:
        """Embed all descriptions and build a FAISS inner-product index."""
        descriptions = self.df["description"].tolist()
        print(f"Embedding {len(descriptions)} descriptions...")
        embeddings = self.model.encode(descriptions, normalize_embeddings=True, show_progress_bar=True)
        self.embeddings = embeddings.astype("float32")

        # Inner product on normalized vectors = cosine similarity
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(self.embeddings)
        print(f"FAISS index built: {index.ntotal} vectors, dim={dim}")
        return index

    # -------------------------------------------------------------- search
    def retrieve(self, query: str, top_k: int = None) -> list[dict]:
        """Find the top-k most similar (description, query) pairs."""
        k = top_k or self.top_k
        query_vec = self.model.encode([query], normalize_embeddings=True).astype("float32")
        scores, indices = self.index.search(query_vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            row = self.df.iloc[idx]
            results.append({
                "description": row["description"],
                "pyspark_query": row["pyspark_query"],
                "similarity": float(score),
            })
        return results

    # --------------------------------------------------------- prompt build
    def build_prompt(self, request: str, top_k: int = None) -> str:
        """Retrieve similar examples and format a complete prompt for ChatGPT."""
        examples = self.retrieve(request, top_k)

        examples_block = ""
        for i, ex in enumerate(examples, 1):
            examples_block += f"--- Example {i} (similarity: {ex['similarity']:.2f}) ---\n"
            examples_block += f"Requirement: {ex['description']}\n"
            examples_block += f"PySpark Query:\n```python\n{ex['pyspark_query']}\n```\n\n"

        prompt = textwrap.dedent(f"""\
        You are a PySpark query generator. Given a requirement, produce a complete, runnable PySpark query.

        Follow the exact coding style shown in the reference examples below.

        Rules:
        - Use PySpark DataFrame API (not RDD)
        - Use spark.table() to reference tables
        - Import pyspark.sql.functions as F
        - Use F.col() for column references
        - Output only the code, no explanations

        === REFERENCE EXAMPLES ===

        {examples_block}
        === NEW REQUIREMENT ===

        {request}

        Generate the PySpark query:
        """)
        return prompt

    # ----------------------------------------------------------- clipboard
    def generate(self, request: str, top_k: int = None, copy: bool = True) -> str:
        """Build prompt, print it, and optionally copy to clipboard."""
        prompt = self.build_prompt(request, top_k)

        print("\n" + "=" * 70)
        print("GENERATED PROMPT (paste this into ChatGPT)")
        print("=" * 70)
        print(prompt)
        print("=" * 70)

        if copy:
            try:
                pyperclip.copy(prompt)
                print("\n>> Prompt copied to clipboard. Paste into ChatGPT.")
            except pyperclip.PyperclipException:
                print("\n>> Could not copy to clipboard. Copy the prompt above manually.")

        return prompt


# -------------------------------------------------------------------- main
def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "sample_data.csv"

    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found")
        sys.exit(1)

    rag = PySparkRAG(csv_path, top_k=5)

    print("\n" + "-" * 50)
    print("PySpark RAG Query Generator")
    print("Type a requirement and press Enter.")
    print("Type 'quit' to exit.")
    print("-" * 50)

    while True:
        try:
            request = input("\nRequirement: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not request or request.lower() in ("quit", "exit", "q"):
            break

        rag.generate(request)


if __name__ == "__main__":
    main()
