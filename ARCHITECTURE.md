# Architecture Overview

**Author:** Sreehari Rayannagari
**Stack:** Python 3.9 · Streamlit · ChromaDB · Gemini 2.0 Flash (`google-genai` SDK)

---

## High-Level Flow

```
  Login / Register  ──  app/user_store.py  (SQLite, hashed passwords)
          |
          v
  Upload file  -->  [ Ingestion ]  app/ingestion.py
                    - pymupdf (PDF) / UTF-8 (TXT, MD)
                    - Section-aware split -> 320-word chunks, 60-word overlap
                    - Metadata: source, page, chunk_id, ingested_at
          |
          v
  [ Indexing ]  app/store.py -> ChromaDB (per-user collection)
                    - Deterministic 256-dim hash embeddings (no model download)
                    - Collection: rag_chunks_v1_<user_id>
          |
          v
  Ask question  -->  [ Hybrid Retrieval ]  app/retrieval.py
                    - BM25 lexical (pure Python) + TF-IDF cosine (scikit-learn)
                    - Weighted: 0.45 x BM25 + 0.55 x semantic -> top-5
                    - Heuristic reranker (query-term overlap boost)
          |
          v
  [ Answer Generation ]  app/rag.py
                    - 9-rule strict system prompt (grounding + anti-injection)
                    - Gemini 2.0 Flash, streaming via st.write_stream()
                    - Inline [1],[2] citation markers
                    - Fallback: extractive quotes from top-3 chunks
          |
          v
  [ Memory Decision ]  app/memory.py
                    - LLM JSON decision: {should_write, target, summary, confidence}
                    - Writes only if confidence >= 0.85, <= 30 words, not sensitive, not duplicate
                    - Appended to per-user USER_MEMORY.md / COMPANY_MEMORY.md
```

---

## 1) Ingestion (Upload -> Parse -> Chunk)

- **Supported inputs:** PDF, TXT, MD (multi-file upload via Streamlit sidebar)
- **Parsing:** `pymupdf` for PDF (page-by-page text extraction); plain UTF-8 read for TXT/MD
- **Chunking:** Section-aware splitting on headings / numbered items, then sliding word window (~320 words, 60-word overlap)
- **Metadata per chunk:** `source` (filename), `page` (page number), `chunk_id` (`filename-chunk-N`), `ingested_at` (UTC)

## 2) Indexing / Storage

- **Vector store:** ChromaDB `PersistentClient` -> `chroma_db/` directory on disk
- **Embeddings:** Custom `LocalHashEmbeddingFunction` — 256-dim normalized token-frequency vector. No internet, no model download, deterministic
- **Per-user isolation:** Each authenticated user gets a separate ChromaDB collection (`rag_chunks_v1_<uid>`), private uploads dir, and memory files under `data/users/<uid>/`
- **BM25 index:** Built at query time from collection contents (pure Python, no external index)

## 3) Retrieval + Grounded Answering

- **Retrieval:** Hybrid BM25 + TF-IDF cosine, both min-max normalized, merged 0.45/0.55 weight, then top-20 re-ranked by heuristic query-term overlap -> top-5 returned
- **Citations:** Each chunk becomes a `Citation(source, locator, snippet)` object. Locator = "page N, filename-chunk-M". Snippet = first 240 chars
- **Inline citation markers:** `[1]`, `[2]` etc. in the generated answer, mapped to the citations list
- **Failure behavior:** If retrieval returns zero chunks -> hard-coded refusal: *"I couldn't find relevant information in the uploaded documents."* No hallucination path exists

## 4) Memory System (Selective)

- **High-signal = reusable facts:** User preferences, roles, document summary facts, org processes
- **NOT stored:** Raw transcripts, verbose answers, anything matching `api_key|password|secret|token|ssn|credit card`
- **Decision flow:** Rule-based fast-path (regex on user message) -> LLM JSON decision -> confidence gate (>= 0.85) -> length guard (<= 30 words) -> sensitivity filter -> fuzzy dedupe (70% token overlap rejection)
- **Storage:** Append-only markdown — `data/users/<uid>/USER_MEMORY.md` and `COMPANY_MEMORY.md`
- **Clear Memory:** Sidebar button resets files to headers only; dedupe correctly checks only `- timestamp | fact` entry lines

## 5) Safe Tooling — Open-Meteo Weather Analytics

- **Interface:** Sidebar expander -> Location + date range -> "Run Analysis" button
- **Flow:** Geocode location -> fetch hourly temperature time series -> compute analytics (mean, volatility, rolling avg, anomaly flags)
- **AST sandbox:** `run_restricted_python()` parses code into AST, rejects `Import`, `Attribute`, `Lambda`, `ClassDef`, `With`, `Try`. Only whitelisted builtins (`len`, `sum`, `min`, `max`, `abs`, `round`)
- **Timeouts:** Geocode 20s, weather fetch 30s
- **Fallback:** Deterministic synthetic data if network unavailable

## 6) Multi-User Support

- **Backend:** `app/user_store.py` — SQLite database (`data/users.db`) with SHA-256 + per-user salt password hashing
- **Auth flow:** Login/Register tabs shown before any app access; `st.session_state.user` gates the main app
- **Per-user isolation:**

  | Resource | Location |
  |----------|----------|
  | Documents (ChromaDB) | Collection `rag_chunks_v1_<uid>` |
  | Uploaded files | `data/users/<uid>/uploads/` |
  | User memory | `data/users/<uid>/USER_MEMORY.md` |
  | Company memory | `data/users/<uid>/COMPANY_MEMORY.md` |
  | Chat history | `data/users/<uid>/chat_history.json` |

- **Persistence:** Chat history is written to disk after every exchange and restored on login. Documents, memories, and chunks all survive logout/login cycles

---

## Tradeoffs & Next Steps

### Why this design?

| Decision | Chose | Gave up | Reason |
|----------|-------|---------|--------|
| Hash embeddings | Zero-dependency, deterministic | Synonym / concept matching | No model download at judge time |
| Heuristic reranker | Fast, no model | Neural reranker quality | Cross-encoder opt-in via `ENABLE_CROSS_ENCODER=1` |
| ChromaDB local | Single command, works offline | Scalability | Perfect for demo scope |
| Flat markdown memory | Human-readable, inspectable | Structured queries | Judges can `cat USER_MEMORY.md` to verify |
| Streamlit | One-command launch | Rich SPA features | Fastest path to working UI |
| SQLite user store | No infra, single file | Concurrent multi-server | Adequate for local demo |

### What I would improve with more time:

1. **Transformer embeddings** (`text-embedding-004` or `all-MiniLM-L6-v2`) — biggest retrieval quality gain
2. **Cross-encoder reranker** enabled by default
3. **Semantic chunking** on sentence boundaries instead of fixed word counts
4. **Content-addressed dedup** on ingest to prevent duplicate chunks on re-upload
5. **Vector memory search** — embed stored memories and retrieve relevant ones at query time
6. **Table/image extraction** from PDFs via `camelot` or `pdfplumber`
7. **Memory editing UI** — edit/delete individual entries, not just clear-all
8. **OAuth / SSO** for production-grade authentication
