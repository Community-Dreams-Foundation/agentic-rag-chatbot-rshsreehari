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
from app.store import get_collection, list_indexed_sources, delete_source, set_active_collection
from app.user_store import (
    authenticate_user,
    register_user,
    load_chat_history,
    save_chat_history,
    UserProfile,
)

load_dotenv()

SAMPLE_DIR = Path("sample_docs")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")

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

    /* Auth card centering */
    .auth-header {
        text-align: center;
        color: #4fc3f7;
    }
    .auth-sub {
        text-align: center;
        color: #b0bec5;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  AUTH GATE — Login / Register before anything else
# ═══════════════════════════════════════════════════════════════════════════════

def _show_auth_screen() -> None:
    """Render the login / register screen. Sets session_state.user on success."""

    st.markdown("<h1 class='auth-header'>🤖 RAG Chatbot</h1>", unsafe_allow_html=True)
    st.markdown("<p class='auth-sub'> Hybrid Retrieval · Selective Memory · Multi-User</p>", unsafe_allow_html=True)

    # Centre the form using columns
    _left, center, _right = st.columns([1, 2, 1])

    with center:
        tab_login, tab_register = st.tabs(["🔑 Login", "📝 Register"])

        with tab_login:
            with st.form("login_form"):
                username = st.text_input("Username", key="login_user")
                password = st.text_input("Password", type="password", key="login_pass")
                submitted = st.form_submit_button("Login", use_container_width=True, type="primary")
                if submitted:
                    if not username or not password:
                        st.error("Please fill in both fields.")
                    else:
                        profile = authenticate_user(username, password)
                        if profile:
                            _login(profile)
                            st.rerun()
                        else:
                            st.error("Invalid username or password.")

        with tab_register:
            with st.form("register_form"):
                new_user = st.text_input("Choose a username", key="reg_user")
                new_display = st.text_input("Display name (optional)", key="reg_display")
                new_pass = st.text_input("Choose a password", type="password", key="reg_pass")
                new_pass2 = st.text_input("Confirm password", type="password", key="reg_pass2")
                submitted = st.form_submit_button("Create Account", use_container_width=True, type="primary")
                if submitted:
                    if not new_user or not new_pass:
                        st.error("Username and password are required.")
                    elif new_pass != new_pass2:
                        st.error("Passwords don't match.")
                    elif len(new_pass) < 3:
                        st.error("Password must be at least 3 characters.")
                    else:
                        profile = register_user(new_user, new_display or new_user, new_pass)
                        if profile:
                            st.success(f"Account created! Logging in as **{profile.display_name}** …")
                            _login(profile)
                            st.rerun()
                        else:
                            st.error("Username already taken. Pick another.")


def _login(profile: UserProfile) -> None:
    """Hydrate session state from the user profile + disk-persisted data."""
    st.session_state.user = {
        "uid": profile.uid,
        "username": profile.username,
        "display_name": profile.display_name,
        "collection_name": profile.collection_name,
    }
    # Activate the user's ChromaDB collection
    set_active_collection(profile.collection_name)
    # Restore persisted chat history from disk
    st.session_state.messages = load_chat_history(profile)
    # Refresh chunk count
    try:
        st.session_state.indexed_chunks = get_collection().count()
    except Exception:
        st.session_state.indexed_chunks = 0


def _logout() -> None:
    """Persist state and clear session."""
    _persist_chat()
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    set_active_collection(None)


def _persist_chat() -> None:
    """Save current chat history to disk for the logged-in user."""
    user = st.session_state.get("user")
    if user and "messages" in st.session_state:
        from app.user_store import _build_profile
        profile = _build_profile(user["uid"], user["username"], user["display_name"])
        save_chat_history(profile, st.session_state.messages)


def _get_user_profile() -> UserProfile:
    """Reconstruct a UserProfile from session state."""
    from app.user_store import _build_profile
    u = st.session_state.user
    return _build_profile(u["uid"], u["username"], u["display_name"])


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN APP — only shown after login
# ═══════════════════════════════════════════════════════════════════════════════

def _show_main_app() -> None:
    user = st.session_state.user
    profile = _get_user_profile()
    UPLOAD_DIR = profile.upload_dir
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    # Ensure the user's collection is active for this script run
    set_active_collection(user["collection_name"])

    if "indexed_chunks" not in st.session_state:
        try:
            st.session_state.indexed_chunks = get_collection().count()
        except Exception:
            st.session_state.indexed_chunks = 0

    # ── Initialize LLM + Memory (per-user paths) ────────────────────────────
    llm = GeminiClient()
    memory = MemoryManager(
        user_path=str(profile.user_memory_path),
        company_path=str(profile.company_memory_path),
    )

    # ═══════════════════════════════════════════════════════════════════════════
    #  SIDEBAR
    # ═══════════════════════════════════════════════════════════════════════════
    with st.sidebar:
        # ── User badge + logout ──────────────────────────────────────────────
        col_user, col_logout = st.columns([3, 1])
        with col_user:
            st.markdown(f"👤 **{user['display_name']}**")
            st.caption(f"@{user['username']}")
        with col_logout:
            if st.button("🚪", help="Logout", key="logout_btn"):
                _logout()
                st.rerun()

        st.divider()

        # ── Document Upload ──────────────────────────────────────────────────
        st.header("Document Upload")
        st.markdown("**Upload documents**")
        st.caption("Drag and drop files here\nLimit 200MB file · TXT, PDF, MD")

        files = st.file_uploader(
            "Browse files",
            type=["txt", "md", "pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        if files:
            for f in files:
                st.markdown(f"📄 **{f.name}** — {f.size / 1024:.1f}KB")

        # ── Ingest Documents button ──────────────────────────────────────────
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

        # ── Load Sample Documents ────────────────────────────────────────────
        if st.button("Load Sample Documents", use_container_width=True):
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

        # ── Indexed Chunks counter ───────────────────────────────────────────
        st.markdown("**Indexed Chunks**")
        st.markdown(
            f'<p class="chunk-counter">{st.session_state.indexed_chunks}</p>',
            unsafe_allow_html=True,
        )

        # ── Indexed Files panel (file management) ────────────────────────────
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

        # ── Memory Section ───────────────────────────────────────────────────
        st.header("Memory")

        user_mem_path = profile.user_memory_path
        company_mem_path = profile.company_memory_path

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
                for entry in user_entries[-5:]:
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

        # ── Weather Tool (Feature C) ─────────────────────────────────────────
        with st.expander("🌤️ Weather Analytics (Feature C)"):
            wx_location = st.text_input("Location", value="San Francisco", key="wx_loc")
            wx_start = st.date_input("Start", value=date.today() - timedelta(days=2), key="wx_start_d")
            wx_end = st.date_input("End", value=date.today(), key="wx_end_d")
            if st.button("Run Analysis", use_container_width=True, key="wx_run"):
                with st.spinner("Fetching weather data..."):
                    try:
                        result = analyze_weather(wx_location, str(wx_start), str(wx_end))

                        # ── Pretty weather card ──────────────────────────
                        st.markdown(f"### 🌍 {result['location']}")
                        st.caption(f"{result['start_date']}  →  {result['end_date']}")

                        c1, c2, c3 = st.columns(3)
                        c1.metric("🌡️ Mean Temp", f"{result['mean_temperature']} °C")
                        c2.metric("📊 Volatility", f"{result['volatility']} °C")
                        c3.metric("📏 Data Points", result["points"])

                        c4, c5 = st.columns(2)
                        c4.metric("⚠️ Anomalies", result["anomaly_count"])
                        c5.metric("❓ Missing", result["missing_count"])

                        # Rolling average trend
                        rolling = result.get("rolling_avg_tail", [])
                        if rolling:
                            st.markdown("**📈 Rolling Avg (last 5 hours)**")
                            st.line_chart(rolling, height=120)

                        # Anomaly details
                        if result["anomaly_count"] > 0:
                            st.warning(
                                f"Detected **{result['anomaly_count']}** anomalous readings "
                                f"(>2σ from mean) at hour indices: {result['anomaly_indices']}"
                            )
                        else:
                            st.success("✅ No anomalous readings detected.")

                    except Exception as exc:
                        st.error(f"Weather analysis failed: {exc}")

    # ═══════════════════════════════════════════════════════════════════════════
    #  MAIN CHAT AREA
    # ═══════════════════════════════════════════════════════════════════════════
    st.title("🤖 RAG Chatbot")
    st.caption(f"Logged in as **{user['display_name']}** · Gemini · Hybrid Retrieval · Selective Memory · Grounded Citations")

    # ── Render chat history ──────────────────────────────────────────────────
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

    # ── Chat input ───────────────────────────────────────────────────────────
    if prompt := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            chunks = retrieve_hybrid(prompt, top_k=5)

            if not chunks:
                answer_text = "I couldn't find relevant information in the uploaded documents. Please upload and ingest a document first."
                st.markdown(answer_text)
                citations_json = []
            else:
                from app.rag import _sanitize_query, _build_prompt, _citations_from_chunks

                safe_query = _sanitize_query(prompt)
                rag_prompt = _build_prompt(safe_query, chunks)
                citations = _citations_from_chunks(chunks)
                citations_json = [
                    {"source": c.source, "locator": c.locator, "snippet": c.snippet}
                    for c in citations
                ]

                answer_text = st.write_stream(llm.stream(rag_prompt, fallback="Based on the uploaded documents: please see the cited passages below."))

                if "[" not in answer_text:
                    answer_text += "\n\nSources: " + ", ".join(
                        f"[{i}]" for i in range(1, min(4, len(citations_json) + 1))
                    )

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

        # Save to session state
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer_text,
                "citations": citations_json,
                "memory_write": mem_msg,
            }
        )
        # Persist chat history to disk after every exchange
        _persist_chat()


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRYPOINT — Route to auth screen or main app
# ═══════════════════════════════════════════════════════════════════════════════

if "user" not in st.session_state:
    _show_auth_screen()
else:
    _show_main_app()
