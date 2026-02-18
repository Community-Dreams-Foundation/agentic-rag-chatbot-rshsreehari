from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from app.ingestion import ingest_paths
from app.llm_client import GeminiClient
from app.memory import MemoryManager
from app.rag import answer_with_citations
from app.retrieval import retrieve_hybrid
from app.sandbox import analyze_weather
from app.store import get_collection, list_indexed_sources, delete_source

load_dotenv()

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_DIR = Path("sample_docs")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="RAG Chatbot", page_icon="�", layout="wide")

# ── Custom CSS for dark polished look ────────────────────────────────────────
st.markdown("""
<style>
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #1a1a2e;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span {
        color: #e0e0e0 !important;
    }

    /* Indexed chunks counter */
    .chunk-counter {
        font-size: 2.5rem;
        font-weight: bold;
        color: #4fc3f7;
        margin-top: 0;
    }

    /* Memory section styling */
    .memory-entry {
        background: #16213e;
        border-radius: 6px;
        padding: 8px 12px;
        margin: 4px 0;
        font-size: 0.85rem;
        color: #b0bec5;
        border-left: 3px solid #4fc3f7;
    }

    /* Chat message tweaks */
    .stChatMessage {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ── Session state init ───────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "indexed_chunks" not in st.session_state:
    # Count existing chunks in the DB
    try:
        st.session_state.indexed_chunks = get_collection().count()
    except Exception:
        st.session_state.indexed_chunks = 0

# ── Initialize LLM + Memory ─────────────────────────────────────────────────
llm = GeminiClient()
memory = MemoryManager()

# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("Document Upload")
    st.markdown("**Upload documents**")
    st.caption("Drag and drop files here\nLimit 200MB file · TXT, PDF, MD")

    files = st.file_uploader(
        "Browse files",
        type=["txt", "md", "pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    # Show uploaded file names
    if files:
        for f in files:
            st.markdown(f"📄 **{f.name}** — {f.size / 1024:.1f}KB")

    # ── Ingest Documents button ──────────────────────────────────────────────
    if st.button("Ingest Documents", use_container_width=True, type="primary"):
        if not files:
            st.warning("Upload at least one file first.")
        else:
            paths: list[Path] = []
            for f in files:
                target = UPLOAD_DIR / f.name
                target.write_bytes(f.getvalue())
                paths.append(target)

            with st.spinner("Parsing, chunking & indexing..."):
                try:
                    count = ingest_paths(paths, reset_index=False)
                    st.session_state.indexed_chunks = get_collection().count()
                    st.success(f"✅ Indexed {count} chunks from {len(paths)} file(s).")
                except Exception as exc:
                    st.error(f"Ingestion failed: {exc}")

    st.divider()

    # ── Load Sample Documents ────────────────────────────────────────────────
    if st.button("Load Sample Documents", use_container_width=True):
        # Filter out system reference docs that would pollute retrieval
        SYSTEM_DOCS = {"hackathon_overview.txt", "README.md"}
        sample_files = list(SAMPLE_DIR.glob("*.*"))
        sample_files = [
            f for f in sample_files
            if f.suffix.lower() in {".txt", ".md", ".pdf"} and f.name not in SYSTEM_DOCS
        ]
        if not sample_files:
            st.warning("No user-facing sample documents found in sample_docs/")
        else:
            with st.spinner("Indexing sample documents..."):
                try:
                    count = ingest_paths(sample_files, reset_index=False)
                    st.session_state.indexed_chunks = get_collection().count()
                    st.success(f"✅ Loaded {len(sample_files)} sample doc(s), {count} chunks.")
                except Exception as exc:
                    st.error(f"Failed: {exc}")

    # ── Indexed Chunks counter ───────────────────────────────────────────────
    st.markdown("**Indexed Chunks**")
    st.markdown(
        f'<p class="chunk-counter">{st.session_state.indexed_chunks}</p>',
        unsafe_allow_html=True,
    )

    # ── Indexed Files panel (file management) ────────────────────────────────
    indexed_sources = list_indexed_sources()
    if indexed_sources:
        with st.expander(f"📂 Indexed Files ({len(indexed_sources)})", expanded=False):
            for src_info in indexed_sources:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"📄 **{src_info['source']}** — {src_info['chunks']} chunks")
                with col2:
                    if st.button("🗑️", key=f"del_{src_info['source']}", help=f"Remove {src_info['source']}"):
                        removed = delete_source(src_info['source'])
                        st.session_state.indexed_chunks = get_collection().count()
                        st.success(f"Removed {removed} chunks from {src_info['source']}")
                        st.rerun()
            st.divider()
            if st.button("🗑️ Clear All", use_container_width=True, type="secondary"):
                from app.store import reset_collection
                reset_collection()
                st.session_state.indexed_chunks = 0
                st.success("All documents cleared.")
                st.rerun()

    st.divider()

    # ── Memory Section ───────────────────────────────────────────────────────
    st.header("Memory")

    user_mem_path = Path("USER_MEMORY.md")
    company_mem_path = Path("COMPANY_MEMORY.md")

    user_entries = []
    if user_mem_path.exists():
        for line in user_mem_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("- ") and "|" in line:
                user_entries.append(line[2:])

    company_entries = []
    if company_mem_path.exists():
        for line in company_mem_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("- ") and "|" in line:
                company_entries.append(line[2:])

    if user_entries or company_entries:
        if user_entries:
            st.markdown("**👤 User Memory**")
            for entry in user_entries[-5:]:  # show last 5
                st.markdown(f'<div class="memory-entry">{entry}</div>', unsafe_allow_html=True)
        if company_entries:
            st.markdown("**🏢 Company Memory**")
            for entry in company_entries[-5:]:
                st.markdown(f'<div class="memory-entry">{entry}</div>', unsafe_allow_html=True)
        if st.button("🗑️ Clear Memory", use_container_width=True, key="clear_mem"):
            _hdr_u = "# USER MEMORY\n\n<!--\nAppend only high-signal, user-specific facts worth remembering.\nDo NOT dump raw conversation.\nAvoid secrets or sensitive information.\n-->\n"
            _hdr_c = "# COMPANY MEMORY\n\n<!--\nAppend reusable org-wide learnings that could help colleagues too.\nDo NOT dump raw conversation.\nAvoid secrets or sensitive information.\n-->\n"
            user_mem_path.write_text(_hdr_u, encoding="utf-8")
            company_mem_path.write_text(_hdr_c, encoding="utf-8")
            st.success("Memory cleared.")
            st.rerun()
    else:
        st.caption("No memories stored yet. Chat to build memory.")

    st.divider()

    # ── Weather Tool (Feature C) ─────────────────────────────────────────────
    with st.expander("🌤️ Weather Analytics (Feature C)"):
        wx_location = st.text_input("Location", value="San Francisco", key="wx_loc")
        wx_start = st.date_input("Start", value=date.today() - timedelta(days=2), key="wx_start_d")
        wx_end = st.date_input("End", value=date.today(), key="wx_end_d")
        if st.button("Run Analysis", use_container_width=True, key="wx_run"):
            with st.spinner("Fetching weather data..."):
                try:
                    result = analyze_weather(wx_location, str(wx_start), str(wx_end))
                    st.json(result)
                except Exception as exc:
                    st.error(f"Weather analysis failed: {exc}")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN CHAT AREA
# ═══════════════════════════════════════════════════════════════════════════════
st.title("🤖 RAG Chatbot")
st.caption("Gemini · Hybrid Retrieval · Selective Memory · Grounded Citations")

# ── Render chat history ──────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("citations"):
            with st.expander("View Citations"):
                for i, c in enumerate(msg["citations"], 1):
                    source = c.get("source", "unknown")
                    locator = c.get("locator", "")
                    snippet = c.get("snippet", "")
                    st.markdown(f"**[{i}] From {source}** ({locator})")
                    st.caption(snippet[:300])
        if msg.get("memory_write"):
            st.caption(f"💾 Memory saved: {msg['memory_write']}")

# ── Chat input ───────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        # Retrieve relevant chunks
        chunks = retrieve_hybrid(prompt, top_k=5)

        if not chunks:
            answer_text = "I couldn't find relevant information in the uploaded documents. Please upload and ingest a document first."
            st.markdown(answer_text)
            citations_json = []
        else:
            # Build prompt + stream the answer from Gemini
            from app.rag import _sanitize_query, _build_prompt, _citations_from_chunks

            safe_query = _sanitize_query(prompt)
            rag_prompt = _build_prompt(safe_query, chunks)
            citations = _citations_from_chunks(chunks)
            citations_json = [
                {"source": c.source, "locator": c.locator, "snippet": c.snippet}
                for c in citations
            ]

            # Stream with st.write_stream for real-time token display
            answer_text = st.write_stream(llm.stream(rag_prompt, fallback="Based on the uploaded documents: please see the cited passages below."))

            # Ensure citations are appended if the model didn't inline them
            if "[" not in answer_text:
                answer_text += "\n\nSources: " + ", ".join(
                    f"[{i}]" for i in range(1, min(4, len(citations_json) + 1))
                )

            # Show citations in expander
            with st.expander("View Citations"):
                for i, c in enumerate(citations_json, 1):
                    st.markdown(f"**[{i}] From {c['source']}** ({c['locator']})")
                    st.caption(c["snippet"][:300])

        # Memory decision
        mem = memory.decide_and_write(prompt, answer_text, llm)
        mem_msg = ""
        if mem:
            mem_msg = f"{mem.target}: {mem.summary}"
            st.caption(f"💾 Memory written → {mem.target}_MEMORY.md")

    # Save to session
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer_text,
            "citations": citations_json,
            "memory_write": mem_msg,
        }
    )
