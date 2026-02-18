from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.store import get_collection


@dataclass
class RetrievedChunk:
    doc_id: str
    text: str
    metadata: dict
    score: float


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def _bm25_scores(query: str, docs: list[str], k1: float = 1.5, b: float = 0.75) -> np.ndarray:
    tokenized_docs = [_tokenize(d) for d in docs]
    q_tokens = _tokenize(query)

    if not docs or not q_tokens:
        return np.zeros(len(docs), dtype=float)

    N = len(tokenized_docs)
    avgdl = sum(len(d) for d in tokenized_docs) / max(1, N)

    doc_freq = {}
    for d in tokenized_docs:
        for t in set(d):
            doc_freq[t] = doc_freq.get(t, 0) + 1

    scores = np.zeros(N, dtype=float)
    for i, d in enumerate(tokenized_docs):
        tf = {}
        for t in d:
            tf[t] = tf.get(t, 0) + 1

        dl = len(d)
        for t in q_tokens:
            if t not in tf:
                continue
            df = doc_freq.get(t, 0)
            idf = math.log(1 + (N - df + 0.5) / (df + 0.5))
            denom = tf[t] + k1 * (1 - b + b * (dl / max(1e-9, avgdl)))
            scores[i] += idf * ((tf[t] * (k1 + 1)) / max(1e-9, denom))

    return scores


def _semantic_scores(query: str, docs: list[str]) -> np.ndarray:
    if not docs:
        return np.array([])

    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vec.fit_transform(docs + [query])
    doc_vecs = matrix[:-1]
    q_vec = matrix[-1]
    sims = cosine_similarity(doc_vecs, q_vec).flatten()
    return sims


def _normalize(scores: np.ndarray) -> np.ndarray:
    if len(scores) == 0:
        return scores
    low, high = float(np.min(scores)), float(np.max(scores))
    if abs(high - low) < 1e-12:
        return np.ones_like(scores) * 0.5
    return (scores - low) / (high - low)


def _heuristic_rerank(query: str, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
    q_terms = set(_tokenize(query))

    def score(c: RetrievedChunk) -> float:
        doc_terms = set(_tokenize(c.text))
        overlap = len(q_terms & doc_terms) / max(1, len(q_terms))
        return 0.8 * c.score + 0.2 * overlap

    return sorted(candidates, key=score, reverse=True)


def _cross_encoder_rerank(query: str, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
    if os.getenv("ENABLE_CROSS_ENCODER", "0") != "1":
        return _heuristic_rerank(query, candidates)

    try:
        from sentence_transformers import CrossEncoder  # type: ignore
    except Exception:
        return _heuristic_rerank(query, candidates)

    model_name = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

    try:
        model = CrossEncoder(model_name)
        pairs = [(query, c.text) for c in candidates]
        scores = model.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: float(x[1]), reverse=True)
        return [c for c, _ in ranked]
    except Exception:
        return _heuristic_rerank(query, candidates)


def retrieve_hybrid(query: str, top_k: int = 5) -> List[RetrievedChunk]:
    collection = get_collection()
    payload = collection.get(include=["documents", "metadatas"])
    docs = payload.get("documents") or []
    metas = payload.get("metadatas") or []
    ids = payload.get("ids") or []

    if not docs:
        return []

    bm25 = _normalize(_bm25_scores(query, docs))
    semantic = _normalize(_semantic_scores(query, docs))

    if len(semantic) != len(bm25):
        semantic = np.zeros_like(bm25)

    hybrid = 0.45 * bm25 + 0.55 * semantic

    candidate_count = min(len(docs), max(top_k * 4, 8))
    idxs = np.argsort(-hybrid)[:candidate_count]

    candidates = [
        RetrievedChunk(doc_id=ids[i], text=docs[i], metadata=metas[i], score=float(hybrid[i]))
        for i in idxs
    ]

    reranked = _cross_encoder_rerank(query, candidates)
    return reranked[:top_k]
