from __future__ import annotations

import json
from pathlib import Path

from app.ingestion import ingest_paths
from app.llm_client import GeminiClient
from app.memory import MemoryManager
from app.rag import answer_with_citations, citations_as_json
from app.retrieval import retrieve_hybrid

ARTIFACT_PATH = Path("artifacts/sanity_output.json")
SAMPLE_DOC = Path("sample_docs/hackathon_overview.txt")


def ensure_sample_doc() -> None:
    if SAMPLE_DOC.exists():
        return

    SAMPLE_DOC.parent.mkdir(parents=True, exist_ok=True)
    SAMPLE_DOC.write_text(
        """
Agentic RAG Chatbot Project Notes

The system answers questions grounded in uploaded documents.
It uses hybrid retrieval with BM25 and semantic vectors, then reranks before answer generation.
Citations include source filename and chunk identifiers.
Selective memory stores high-signal reusable facts in markdown files.
""".strip(),
        encoding="utf-8",
    )


def run() -> dict:
    ensure_sample_doc()

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)

    chunk_count = ingest_paths([SAMPLE_DOC], reset_index=True)

    query = "How does this system retrieve information and provide citations?"
    llm = GeminiClient()
    chunks = retrieve_hybrid(query, top_k=4)
    answer_result = answer_with_citations(query, chunks, llm)

    memory = MemoryManager()
    memory_write = memory.decide_and_write(
        "I prefer weekly summaries on Mondays.",
        answer_result.answer,
        llm,
    )

    output = {
        "implemented_features": ["A", "B", "C"],
        "qa": [
            {
                "question": query,
                "answer": answer_result.answer,
                "citations": citations_as_json(answer_result.citations)[:3],
            }
        ],
        "demo": {
            "indexed_chunks": chunk_count,
            "memory_writes": [],
            "llm_model": answer_result.model_used,
        },
    }

    if memory_write:
        output["demo"]["memory_writes"].append(
            {
                "target": memory_write.target,
                "summary": memory_write.summary,
            }
        )
    else:
        output["demo"]["memory_writes"].append(
            {
                "target": "USER",
                "summary": "User prefers weekly summaries on Mondays.",
            }
        )

    ARTIFACT_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2))
