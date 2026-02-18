from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Tuple

from app.store import get_collection, reset_collection


@dataclass
class ParsedDocument:
    source: str
    pages: List[Tuple[int, str]]


def _parse_txt_or_md(path: Path) -> ParsedDocument:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return ParsedDocument(source=path.name, pages=[(1, text)])


def _parse_pdf(path: Path) -> ParsedDocument:
    import fitz  # pymupdf

    doc = fitz.open(path)
    pages: List[Tuple[int, str]] = []
    for i, page in enumerate(doc):
        pages.append((i + 1, page.get_text("text")))
    return ParsedDocument(source=path.name, pages=pages)


def parse_document(path: Path) -> ParsedDocument:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return _parse_txt_or_md(path)
    if suffix == ".pdf":
        return _parse_pdf(path)
    raise ValueError(f"Unsupported file type: {suffix}")


def _normalize_whitespace(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _split_sections(text: str) -> List[str]:
    rough = re.split(r"(?=\n#{1,3}\s)|(?=\n[A-Z][A-Za-z\s]{3,80}:)|(?=\n\d+\.?\s+[A-Z])", text)
    out = []
    for block in rough:
        n = _normalize_whitespace(block)
        if n:
            out.append(n)
    return out or [_normalize_whitespace(text)]


def _word_chunks(text: str, chunk_words: int = 320, overlap_words: int = 60) -> Iterable[str]:
    words = text.split()
    if not words:
        return
    start = 0
    step = max(1, chunk_words - overlap_words)
    while start < len(words):
        end = min(len(words), start + chunk_words)
        yield " ".join(words[start:end])
        if end == len(words):
            break
        start += step


def chunk_document(parsed: ParsedDocument) -> list[dict]:
    chunks: list[dict] = []
    chunk_index = 0

    for page_no, page_text in parsed.pages:
        for section_text in _split_sections(page_text):
            for piece in _word_chunks(section_text):
                chunk_index += 1
                chunks.append(
                    {
                        "chunk_id": f"{parsed.source}-chunk-{chunk_index}",
                        "text": piece,
                        "source": parsed.source,
                        "page": page_no,
                    }
                )
    return chunks


def ingest_paths(paths: list[Path], reset_index: bool = False) -> int:
    if reset_index:
        reset_collection()

    collection = get_collection()
    now = datetime.now(timezone.utc).isoformat()
    total = 0

    for path in paths:
        parsed = parse_document(path)
        chunks = chunk_document(parsed)

        ids = []
        docs = []
        metas = []

        for chunk in chunks:
            ids.append(f"{chunk['chunk_id']}-{uuid.uuid4().hex[:10]}")
            docs.append(chunk["text"])
            metas.append(
                {
                    "source": chunk["source"],
                    "page": str(chunk["page"]),
                    "chunk_id": chunk["chunk_id"],
                    "ingested_at": now,
                }
            )

        if ids:
            collection.add(ids=ids, documents=docs, metadatas=metas)
            total += len(ids)

    return total
