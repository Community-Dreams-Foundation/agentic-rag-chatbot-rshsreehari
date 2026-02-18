from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from app.llm_client import GeminiClient


@dataclass
class MemoryWrite:
    target: str
    summary: str
    confidence: float


class MemoryManager:
    def __init__(self, user_path: str = "USER_MEMORY.md", company_path: str = "COMPANY_MEMORY.md") -> None:
        self.user_path = Path(user_path)
        self.company_path = Path(company_path)

    def _is_sensitive(self, text: str) -> bool:
        patterns = [
            r"api[_-]?key",
            r"password",
            r"secret",
            r"token",
            r"ssn",
            r"credit card",
        ]
        low = text.lower()
        return any(re.search(p, low) for p in patterns)

    def _extract_rule_based(self, user_message: str) -> dict:
        msg = user_message.strip()

        preference_patterns = [
            r"\bi prefer\b",
            r"\bmy role is\b",
            r"\bi am\b",
            r"\bi'm\b",
            r"\bplease remember\b",
        ]

        if any(re.search(p, msg.lower()) for p in preference_patterns) and not self._is_sensitive(msg):
            return {
                "should_write": True,
                "target": "user",
                "summary": msg,
                "confidence": 0.9,
            }

        org_patterns = [r"team", r"company", r"workflow", r"process", r"bottleneck"]
        if any(re.search(p, msg.lower()) for p in org_patterns) and not self._is_sensitive(msg):
            return {
                "should_write": True,
                "target": "company",
                "summary": msg,
                "confidence": 0.85,
            }

        return {
            "should_write": False,
            "target": "user",
            "summary": "",
            "confidence": 0.0,
        }

    def _dedupe(self, path: Path, summary: str) -> bool:
        """Reject if an existing entry is substantially similar (fuzzy match)."""
        if not path.exists():
            return False
        existing = path.read_text(encoding="utf-8", errors="ignore")
        existing_lower = existing.lower()
        summary_lower = summary.lower().strip()

        # Exact substring match
        if summary_lower in existing_lower:
            return True

        # Fuzzy token-overlap check: if >70% of tokens already exist in memory, skip
        summary_tokens = set(re.findall(r"[a-z0-9]+", summary_lower))
        if not summary_tokens:
            return True
        # Check against each existing memory entry
        for line in existing.splitlines():
            line = line.strip()
            if not line.startswith("- ") or "|" not in line:
                continue
            entry_part = line.split("|", 1)[-1].strip().lower()
            entry_tokens = set(re.findall(r"[a-z0-9]+", entry_part))
            if not entry_tokens:
                continue
            overlap = len(summary_tokens & entry_tokens) / len(summary_tokens)
            if overlap > 0.70:
                return True
        return False

    def _append(self, path: Path, summary: str) -> None:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        with path.open("a", encoding="utf-8") as f:
            f.write(f"\n- {ts} | {summary.strip()}\n")

    def decide_and_write(self, user_message: str, assistant_message: str, llm: GeminiClient) -> MemoryWrite | None:
        default = self._extract_rule_based(user_message)

        prompt = f"""Analyze this conversation and decide if a DURABLE memory should be stored.
Return strict JSON: {{"should_write": bool, "target": "user"|"company", "summary": string, "confidence": 0..1}}

STRICT RULES — follow exactly:
- Write ONLY short, reusable, profile-level facts (e.g. "User prefers dark mode", "Company uses 2-week sprints").
- NEVER summarize the assistant's answer or the document content.
- NEVER store RAG retrieval results, document summaries, or Q&A transcripts.
- Only write if the USER explicitly states a preference, role, or org fact.
- If the conversation is just a question about a document and an answer, set should_write=false.
- confidence must be >= 0.85 to write. If uncertain, set should_write=false.
- summary must be ONE concise sentence, max 20 words.

User: {user_message}
Assistant: {assistant_message}""".strip()

        decision = llm.json_decision(prompt, default_obj=default)

        should_write = bool(decision.get("should_write", False))
        target = str(decision.get("target", "user")).lower()
        summary = str(decision.get("summary", "")).strip()
        confidence = float(decision.get("confidence", 0.0) or 0.0)

        if not should_write or confidence < 0.85 or not summary:
            return None
        if self._is_sensitive(summary):
            return None
        # Reject overly long summaries — likely document dumps, not durable facts
        if len(summary.split()) > 30:
            return None

        if target not in {"user", "company"}:
            target = "user"

        path = self.user_path if target == "user" else self.company_path
        if self._dedupe(path, summary):
            return None

        self._append(path, summary)
        return MemoryWrite(target=target.upper(), summary=summary, confidence=confidence)
