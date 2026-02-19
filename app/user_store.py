"""Lightweight multi-user backend backed by SQLite.

Each user gets:
- Isolated ChromaDB collection (rag_chunks_v1_<uid>)
- Per-user memory files  (data/users/<uid>/USER_MEMORY.md etc.)
- Per-user upload dir     (data/users/<uid>/uploads/)
- Persistent chat history (data/users/<uid>/chat_history.json)
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

DB_PATH = Path("data/users.db")
USERS_DIR = Path("data/users")

# ── Ensure dirs exist ────────────────────────────────────────────────────────
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
USERS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class UserProfile:
    uid: str
    username: str
    display_name: str
    collection_name: str
    user_memory_path: Path
    company_memory_path: Path
    upload_dir: Path
    chat_history_path: Path


# ── Password hashing (SHA-256 + per-user salt — good enough for demo) ───────

def _hash_password(password: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{password}".encode()).hexdigest()


# ── SQLite helpers ───────────────────────────────────────────────────────────

def _get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            uid          TEXT PRIMARY KEY,
            username     TEXT UNIQUE NOT NULL,
            display_name TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            salt         TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn


def register_user(username: str, display_name: str, password: str) -> Optional[UserProfile]:
    """Create a new user. Returns UserProfile on success, None if username taken."""
    username = username.strip().lower()
    if not username or not password:
        return None

    conn = _get_db()
    # Check if exists
    row = conn.execute("SELECT uid FROM users WHERE username = ?", (username,)).fetchone()
    if row:
        conn.close()
        return None

    uid = uuid.uuid4().hex[:12]
    salt = uuid.uuid4().hex
    pw_hash = _hash_password(password, salt)

    conn.execute(
        "INSERT INTO users (uid, username, display_name, password_hash, salt) VALUES (?, ?, ?, ?, ?)",
        (uid, username, display_name or username, pw_hash, salt),
    )
    conn.commit()
    conn.close()

    profile = _build_profile(uid, username, display_name or username)
    _init_user_dirs(profile)
    return profile


def authenticate_user(username: str, password: str) -> Optional[UserProfile]:
    """Verify credentials. Returns UserProfile on success, None on failure."""
    username = username.strip().lower()
    conn = _get_db()
    row = conn.execute(
        "SELECT uid, display_name, password_hash, salt FROM users WHERE username = ?",
        (username,),
    ).fetchone()
    conn.close()

    if not row:
        return None

    uid, display_name, pw_hash, salt = row
    if _hash_password(password, salt) != pw_hash:
        return None

    profile = _build_profile(uid, username, display_name)
    _init_user_dirs(profile)
    return profile


# ── Profile builder ──────────────────────────────────────────────────────────

def _build_profile(uid: str, username: str, display_name: str) -> UserProfile:
    user_dir = USERS_DIR / uid
    return UserProfile(
        uid=uid,
        username=username,
        display_name=display_name,
        collection_name=f"rag_chunks_v1_{uid}",
        user_memory_path=user_dir / "USER_MEMORY.md",
        company_memory_path=user_dir / "COMPANY_MEMORY.md",
        upload_dir=user_dir / "uploads",
        chat_history_path=user_dir / "chat_history.json",
    )


def _init_user_dirs(profile: UserProfile) -> None:
    """Create user directories and seed memory files if missing."""
    profile.upload_dir.mkdir(parents=True, exist_ok=True)

    _hdr_u = (
        "# USER MEMORY\n\n"
        "<!--\n"
        "Append only high-signal, user-specific facts worth remembering.\n"
        "Do NOT dump raw conversation.\n"
        "Avoid secrets or sensitive information.\n"
        "-->\n"
    )
    _hdr_c = (
        "# COMPANY MEMORY\n\n"
        "<!--\n"
        "Append reusable org-wide learnings that could help colleagues too.\n"
        "Do NOT dump raw conversation.\n"
        "Avoid secrets or sensitive information.\n"
        "-->\n"
    )

    if not profile.user_memory_path.exists():
        profile.user_memory_path.write_text(_hdr_u, encoding="utf-8")
    if not profile.company_memory_path.exists():
        profile.company_memory_path.write_text(_hdr_c, encoding="utf-8")


# ── Persistent chat history ─────────────────────────────────────────────────

def load_chat_history(profile: UserProfile) -> list[dict]:
    """Load persisted chat history from disk."""
    if not profile.chat_history_path.exists():
        return []
    try:
        data = json.loads(profile.chat_history_path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return []


def save_chat_history(profile: UserProfile, messages: list[dict]) -> None:
    """Persist chat history to disk."""
    try:
        profile.chat_history_path.write_text(
            json.dumps(messages, ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
    except Exception:
        pass
