from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Iterable, List

import chromadb
from chromadb.api.models.Collection import Collection

DB_DIR = Path("chroma_db")
COLLECTION_NAME = "rag_chunks_v1"
EMBED_DIM = 256


class LocalHashEmbeddingFunction:
    """Deterministic local embeddings to avoid remote model downloads."""

    def name(self) -> str:
        # Chroma validates embedding function identity using this method.
        return "local_hash_embedding_v1"

    def __call__(self, input: Iterable[str]) -> List[List[float]]:
        return [self._embed(text) for text in input]

    def _embed(self, text: str) -> List[float]:
        vec = [0.0] * EMBED_DIM
        tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
        if not tokens:
            return vec

        for tok in tokens:
            idx = hash(tok) % EMBED_DIM
            vec[idx] += 1.0

        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec


def get_collection() -> Collection:
    DB_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(DB_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
        embedding_function=LocalHashEmbeddingFunction(),
    )


def reset_collection() -> None:
    client = chromadb.PersistentClient(path=str(DB_DIR))
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass


def list_indexed_sources() -> list[dict]:
    """Return a list of {source, chunk_count} for each unique file in the index."""
    collection = get_collection()
    total = collection.count()
    if total == 0:
        return []

    payload = collection.get(include=["metadatas"])
    metas = payload.get("metadatas") or []

    counts: dict[str, int] = {}
    for m in metas:
        src = m.get("source", "unknown")
        counts[src] = counts.get(src, 0) + 1

    return [{"source": src, "chunks": cnt} for src, cnt in sorted(counts.items())]


def delete_source(source_name: str) -> int:
    """Delete all chunks belonging to a specific source file. Returns count deleted."""
    collection = get_collection()
    payload = collection.get(include=["metadatas"])
    ids = payload.get("ids") or []
    metas = payload.get("metadatas") or []

    to_delete = [
        doc_id for doc_id, m in zip(ids, metas)
        if m.get("source") == source_name
    ]

    if to_delete:
        collection.delete(ids=to_delete)
    return len(to_delete)
